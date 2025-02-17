import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from frac_optim.optimizers import RMSProp

class TestRMSProp(unittest.TestCase):
    def test_standard_update(self):
        # Create a simple parameter tensor.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        # Use MSELoss on the element-wise square of p.
        criterion = nn.MSELoss()
        lr = 0.1
        eps = 1e-10
        # For simplicity, use alpha = 0 so that the running average equals the squared gradient.
        alpha = 0.99
        optimizer = RMSProp([p], lr=lr, eps=eps, alpha=alpha)
        
        # Save a copy of the initial parameter.
        old_data = p.data.clone()
        # Compute a value and loss.
        val = (p ** 2).sum()
        val.backward()
        # With MSELoss applied in this way, we assume p.grad becomes 2*p.
        # Thus, for p = [1, 2] the gradient is [2, 4] and its squared values are [4, 16].
        v0 = torch.tensor([0.0, 0.0])
        vt = alpha * v0 + (1 - alpha) * (2 * p) ** 2  # vt = [0.04, 0.16]
        avg = vt.sqrt() + eps  # sqrt(vt) = [0.2, 0.4]
        expected = old_data - lr * (torch.tensor([2.0, 4.0]) / avg)
        
        optimizer.step()
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-6),
                        msg=f"Expected {expected}, but got {p.data}")

    def test_operator_update(self):
        # Define a dummy operator similar to the one used in the SGD and AdaGrad tests.
        # This operator takes the current parameter's gradient, doubles it,
        # then adds the previous parameter value and the second-order gradient.
        operator = lambda p, old_param, second_order: p.grad * 2 + old_param + second_order
        lr = 0.1
        eps = 1e-10
        alpha = 0.99
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        old_p = p.data.clone()
        optimizer = RMSProp([p], operator=operator, lr=lr, eps=eps, alpha=alpha)
        
        # --- First step: standard RMSProp update (operator not used yet) ---
        # Use a simple sum function so that grad = 2*p.
        # Expected vt: (1 - alpha) * grad^2 = [0.04, 0.16]
        f = (p ** 2).sum()
        f.backward(create_graph=True)
        
        v0 = torch.tensor([0.0, 0.0])
        vt = alpha * v0 + (1 - alpha) * (2 * p) ** 2  # vt = [0.04, 0.16]
        avg = vt.sqrt() + eps  # sqrt(vt) = [0.2, 0.4]
        # For p = [1.0, 2.0], p.grad becomes [2, 4] and vt is updated to [4, 16].
        expected_first = old_p - lr * (torch.tensor([2.0, 4.0]) / avg)

        optimizer.step()
        self.assertTrue(torch.allclose(p.data, expected_first, atol=1e-5),
                        msg=f"After first step, expected {expected_first}, but got {p.data}")
        
        # --- Second step: operator update branch ---
        optimizer.zero_grad()
        f = (p ** 2).sum()
        # This backward call computes:
        #   p.grad = 2*p, i.e. 2*expected_first, and
        #   second_order_grads = autograd.grad(p.grad.sum(), p, create_graph=True)[0],
        # which will be [2, 2] for this function.
        f.backward(create_graph=True)
        optimizer.step()
        # The operator returns: 2*p.grad + old_p + second_order_grads.
        # Since p.grad is 2*expected_first, the operator yields:
        #   op_val = 2*(2*expected_first) + old_p + [2, 2] = 4*expected_first + old_p + [2, 2]
        op_val = 4 * expected_first + old_p + torch.tensor([2.0, 2.0])
        # Update vt: vt was [4, 16] and now we add op_val^2.
        new_vt = alpha * vt + (1 - alpha) * op_val ** 2
        expected_second = expected_first - lr * op_val / (torch.sqrt(new_vt) + eps)
        self.assertTrue(torch.allclose(p.data, expected_second, atol=1e-5),
                        msg=f"After second step, expected {expected_second}, but got {p.data}")

    def test_closure(self):
        # Test that a closure properly computes the loss and updates the parameter.
        lr = 0.03
        eps = 1e-10
        alpha = 0
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = RMSProp([p], lr=lr, eps=eps, alpha=alpha)

        def closure():
            optimizer.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        # For f = (p**2).sum(), grad = 2*p so that vt becomes [4, 16].
        expected = torch.tensor([1.0, 2.0]) - lr * (torch.tensor([2.0, 4.0]) / (torch.sqrt(torch.tensor([4.0, 16.0])) + eps))
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-5),
                        msg=f"After closure step, expected {expected}, but got {p.data}")
        self.assertAlmostEqual(loss_val.item(), 5.0, places=5,
                               msg=f"Expected loss 5.0, but got {loss_val.item()}")

    def test_invalid_operator(self):
        # Passing a non-callable operator should raise a ValueError.
        p = nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            _ = RMSProp([p], operator=123, lr=0.03)

    def test_none_grad(self):
        # If a parameter's gradient is None, its value should not change.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        alpha = 0
        optimizer = RMSProp([p], lr=0.1, eps=1e-10, alpha=alpha)
        old_data = p.data.clone()
        # Do not compute any gradients.
        optimizer.step()
        self.assertTrue(torch.allclose(p.data, old_data, atol=1e-6),
                        msg="Parameter data changed even though gradient was None.")

if __name__ == '__main__':
    unittest.main()
