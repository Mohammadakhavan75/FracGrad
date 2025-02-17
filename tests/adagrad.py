import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from frac_optim.optimizers import AdaGrad

class TestAdaGrad(unittest.TestCase):
    def test_standard_update(self):
        # Create a simple parameter tensor.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        # Use MSELoss on the element-wise square of p.
        criterion = nn.MSELoss()
        lr = 0.1
        eps = 1e-10  # Assuming AdaGrad uses an epsilon of 1e-10.
        optimizer = AdaGrad([p], lr=lr)
        
        # Save a copy of the initial parameter.
        old_data = p.data.clone()
        # Compute a value and loss.
        val = p ** 2
        loss = criterion(val, torch.tensor(0.))
        loss.backward()
        # For MSELoss applied to p**2 vs. 0, the backward pass computes a gradient;
        # Sum of squared gradients is [ (2*1)^2, (2*2)^2 ] = [4, 16]
        sum_sq_grad = torch.tensor([4.0, 16.0])
        expected = p.data + ( -lr * (2 * p.data) / (torch.sqrt(sum_sq_grad) + eps) )


        optimizer.step()
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-6),
                        msg=f"Expected {expected}, but got {p.data}")

    def test_operator_update(self):
        # Define a dummy operator similar to the one used in the SGD tests.
        # This operator takes the current parameter's gradient,
        # doubles it, then adds the previous parameter value and a second-order term.
        operator = lambda p, old_param, second_order: p.grad * 2 + old_param + second_order
        lr = 0.1
        eps = 1e-10
        
        # Create a parameter with gradient tracking.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        old_p = p.data.clone()
        optimizer = AdaGrad([p], operator=operator, lr=lr, eps=eps)
        
        # --- First step: standard AdaGrad update (operator not used yet) ---
        # Use a simple function so that grad = 2*p.
        f = (p ** 2).sum()
        f.backward(create_graph=True)
        optimizer.step()
        # For p = [1.0, 2.0], grad becomes [2, 4] and the accumulated squared gradients are [4, 16].
        # Hence, the update is:
        #   p_new = p - lr/sqrt([4,16] + eps)*[2,4] = [1,2] - [0.1/2*2, 0.1/4*4] = [0.9, 1.9]
        sum_sq = torch.tensor([4.0, 16.0])
        expected_first = old_p - lr * (torch.tensor([2.0, 4.0]) / (torch.sqrt(sum_sq) + eps))
        self.assertTrue(torch.allclose(p.data, expected_first, atol=1e-5),
                        msg=f"After first step, expected {expected_first}, but got {p.data}")
        
        # --- Second step: operator update branch ---
        optimizer.zero_grad()  # Clear previous gradients.
        f = (p ** 2).sum()
        f.backward(create_graph=True)
        optimizer.step()
        # In this branch the operator is applied.
        # At p = expected_first, p.grad = 2*expected_first.
        # The operator returns: 2*(2*expected_first) + old_p + second_order = 4*expected_first + old_p + second_order.
        second_order = 2
        op_val = 4 * expected_first + old_p + second_order
        sum_sq += op_val ** 2 # Accumulated squared gradients now include the operator value.
        expected_second = expected_first - lr * op_val / (torch.sqrt(sum_sq) + eps)
        self.assertTrue(torch.allclose(p.data, expected_second, atol=1e-5),
                        msg=f"After second step, expected {expected_second}, but got {p.data}")

    def test_closure(self):
        # Test that a closure properly computes the loss and updates the parameter.
        lr = 0.03
        eps = 1e-10
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = AdaGrad([p], lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = (p ** 2).sum()  # f = 1^2 + 2^2 = 5 initially.
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        # For f = (p**2).sum(), grad = 2*p, accumulated squared grad = [4,16] so sqrt = [2,4].
        # Expected update: p_new = p - lr/sqrt([4,16] + eps)*[2,4] = [1,2] - [0.03, 0.03] = [0.97, 1.97]
        expected = torch.tensor([1.0, 2.0]) - lr * (torch.tensor([2.0, 4.0]) / (torch.sqrt(torch.tensor([4.0, 16.0])) + eps))
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-5),
                        msg=f"After closure step, expected {expected}, but got {p.data}")
        self.assertAlmostEqual(loss_val.item(), 5.0, places=5,
                               msg=f"Expected loss 5.0, but got {loss_val.item()}")

    def test_invalid_operator(self):
        # Passing a non-callable operator should raise a ValueError.
        p = nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            _ = AdaGrad([p], operator=123, lr=0.03)

    def test_none_grad(self):
        # If a parameter's gradient is None, its value should not change.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = AdaGrad([p], lr=0.1)
        old_data = p.data.clone()
        # Do not compute any gradients.
        optimizer.step()
        self.assertTrue(torch.allclose(p.data, old_data, atol=1e-6),
                        msg="Parameter data changed even though gradient was None.")

if __name__ == '__main__':
    unittest.main()
