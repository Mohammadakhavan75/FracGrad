import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from frac_optim.optimizers import SGD

class TestSGD(unittest.TestCase):
    def test_standard_update(self):
        # Create a simple parameter tensor.
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        # Define a simple loss function and optimizer.
        criterion = nn.MSELoss()
        lr = 0.1
        optimizer = SGD([p], lr=lr)
        
        # Save a copy of the old parameter data.
        old_data = p.data.clone()
        # Compute a value.
        val = p ** 2
        # Compute the loss.
        loss = criterion(val, torch.tensor(0.))
        # Backpropagate the loss .
        loss.backward()
        optimizer.step()
        # Expected update: p_new = p_old - lr * grad
        expected = old_data - lr * torch.tensor([2.0, 16.0])
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-6),
                        msg=f"Expected {expected}, but got {p.data}")

    def test_operator_update(self):
        # Define a dummy operator that doubles the gradient.
        # Note: This operator ignores the 'old' parameter and second-order grads.
        operator = lambda p, old_param, second_order: p.grad * 2 + old_param + second_order
        lr = 0.1

        # Create a parameter with gradient tracking.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        old_p = p.data.clone()
        optimizer = SGD([p], operator=operator, lr=lr)

        # --- First step: standard update (since old_params is not set) ---
        # Use a differentiable function to compute a nontrivial gradient.
        # f = sum(p^2)  => grad = 2*p.
        f = (p ** 2).sum()
        # Use create_graph=True so that gradients are differentiable (needed for second-order).
        f.backward(create_graph=True)
        optimizer.step()

        # Expected: p_new = [1,2] - lr * (2*[1,2]) = [1 - 0.2, 2 - 0.4] = [0.8, 1.6]
        expected_first = torch.tensor([1.0, 2.0]) - lr * (2 * torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.allclose(p.data, expected_first, atol=1e-5),
                        msg=f"After first step, expected {expected_first}, but got {p.data}")

        # --- Second step: operator update branch ---
        optimizer.zero_grad()  # Clear previous gradients.
        # Compute a new loss.
        f = (p ** 2).sum()
        f.backward(create_graph=True)
        optimizer.step()
        # The second order of gradient from p ^ 2 is 2,
        # so the second step gradient is 2 * expected_first (updated p)  + old_p (last step p) + second_order
        second_order = 2
        second_step_grad = 4 * expected_first + old_p + second_order
        expected_second = expected_first - lr * second_step_grad
        self.assertTrue(torch.allclose(p.data, expected_second, atol=1e-5),
                        msg=f"After second step, expected {expected_second}, but got {p.data}")

    def test_closure(self):
        # Test that providing a closure returns the loss.
        lr = 0.03
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = SGD([p], lr=lr)

        def closure():
            # Zero gradients (if any)
            optimizer.zero_grad()
            loss = (p ** 2).sum()  # f = 1^2 + 2^2 = 5 initially.
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        # After the step, standard SGD update: p_new = p_old - lr*(2*p_old)
        expected = torch.tensor([1.0, 2.0]) - lr * (2 * torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-5),
                        msg=f"After closure step, expected {expected}, but got {p.data}")
        self.assertAlmostEqual(loss_val.item(), 5.0, places=5,
                               msg=f"Expected loss 5.0, but got {loss_val.item()}")

    def test_invalid_operator(self):
        # Passing a non-callable operator should raise a ValueError.
        p = nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            _ = SGD([p], operator=123, lr=0.03)

    def test_none_grad(self):
        # If a parameter's gradient is None, its value should not change.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = SGD([p], lr=0.1)
        old_data = p.data.clone()
        # Do not set any gradient.
        optimizer.step()
        self.assertTrue(torch.allclose(p.data, old_data, atol=1e-6),
                        msg="Parameter data changed even though gradient was None.")

if __name__ == '__main__':
    unittest.main()
