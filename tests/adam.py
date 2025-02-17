import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from frac_optim.optimizers import Adam

class TestAdam(unittest.TestCase):
    def test_standard_update(self):
        lr = 0.1
        eps = 1e-8
        betas = (0.9, 0.999)
        # Create a simple parameter tensor.
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = Adam([p], lr=lr, eps=eps, betas=betas)
        
        # Use the sum-of-squares function so that the gradient is 2*p.
        loss = (p ** 2).sum()
        loss.backward(create_graph=True)
        optimizer.step()
        
        # For t = 1, the Adam update computes:
        #   grad = 2 * [1, 2] = [2, 4]
        #   mt = (1 - beta1) * grad = 0.1 * [2, 4] = [0.2, 0.4]
        #   vt = (1 - beta2) * grad^2 = 0.001 * [4, 16] = [0.004, 0.016]
        #   Bias correction:
        #       mt_hat = mt / (1 - beta1^1) = [0.2/0.1, 0.4/0.1] = [2, 4]
        #       vt_hat = vt / (1 - beta2^1) = [0.004/0.001, 0.016/0.001] = [4, 16]
        #   Effective update = [2/√4, 4/√16] = [1, 1]
        #   Thus, expected p = [1, 2] - 0.1 * [1, 1] = [0.9, 1.9].
        grad = 2 * torch.tensor([1.0, 2.0])
        mt = (1 - betas[0]) * grad
        vt = (1 - betas[1]) * (grad ** 2)
        mt_hat = mt / (1 - betas[0] ** 1)
        vt_hat = vt / (1 - betas[1] ** 1)
        update = mt_hat / (vt_hat.sqrt() + eps)
        expected = torch.tensor([1.0, 2.0]) - lr * update
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-6),
                        msg=f"Expected {expected}, but got {p.data}")

    def test_operator_update(self):
        lr = 0.1
        eps = 1e-8
        betas = (0.9, 0.999)
        # Define a dummy operator that doubles the gradient and adds the previous parameter
        # value and the second-order gradient.
        operator = lambda p, old, second: p.grad * 2 + old + second
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = Adam([p], operator=operator, lr=lr, eps=eps, betas=betas)
        
        # --- First step: standard Adam update to initialize operator state ---
        old_p = p.data.clone()  # Should be [1.0, 2.0]
        loss = (p ** 2).sum()
        loss.backward(create_graph=True)
        optimizer.step()
        # As in test_standard_update, after the first step p should be [0.9, 1.9].
        expected_first = torch.tensor([0.9, 1.9])
        self.assertTrue(torch.allclose(p.data, expected_first, atol=1e-5),
                        msg=f"After first step, expected {expected_first}, but got {p.data}")
        
        # --- Second step: operator update branch ---
        optimizer.zero_grad()
        loss = (p ** 2).sum()
        loss.backward(create_graph=True)
        optimizer.step()
        # For the second step:
        #   p (before second step) is expected_first = [0.9, 1.9]
        #   grad = 2 * p = 2 * [0.9, 1.9] = [1.8, 3.8]
        #   second_order_grads = [2, 2] (since derivative of 2*p is 2)
        #   The operator returns:
        #       op_val = 2 * grad + old_p + second_order
        #              = 2*[1.8, 3.8] + [1,2] + [2,2]
        #              = [3.6, 7.6] + [1,2] + [2,2] = [6.6, 11.6]
        grad_second = 2 * expected_first
        second_order = torch.tensor([2.0, 2.0])
        op_val = 2 * grad_second + old_p + second_order  # = [6.6, 11.6]
        
        # The Adam state from the first step:
        mt_prev = torch.tensor([0.2, 0.4])       # (1-beta1)*[2,4]
        vt_prev = torch.tensor([0.004, 0.016])     # (1-beta2)*[4,16]
        # Update the moving averages:
        mt_new = betas[0] * mt_prev + (1 - betas[0]) * op_val
        vt_new = betas[1] * vt_prev + (1 - betas[1]) * (op_val ** 2)
        t = 2
        mt_hat = mt_new / (1 - betas[0] ** t)
        vt_hat = vt_new / (1 - betas[1] ** t)
        update_second = mt_hat / (vt_hat.sqrt() + eps)
        expected_second = expected_first - lr * update_second
        self.assertTrue(torch.allclose(p.data, expected_second, atol=1e-5),
                        msg=f"After second step, expected {expected_second}, but got {p.data}")

    def test_closure(self):
        lr = 0.1
        eps = 1e-8
        betas = (0.9, 0.999)
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = Adam([p], lr=lr, eps=eps, betas=betas)

        def closure():
            optimizer.zero_grad()
            loss = (p ** 2).sum()
            loss.backward(create_graph=True)
            return loss

        loss_val = optimizer.step(closure=closure)
        # Expected update as in test_standard_update.
        grad = 2 * torch.tensor([1.0, 2.0])
        mt = (1 - betas[0]) * grad
        vt = (1 - betas[1]) * (grad ** 2)
        mt_hat = mt / (1 - betas[0])
        vt_hat = vt / (1 - betas[1])
        update = mt_hat / (vt_hat.sqrt() + eps)
        expected = torch.tensor([1.0, 2.0]) - lr * update
        self.assertTrue(torch.allclose(p.data, expected, atol=1e-5),
                        msg=f"After closure step, expected {expected}, but got {p.data}")
        self.assertAlmostEqual(loss_val.item(), 5.0, places=5,
                               msg=f"Expected loss 5.0, but got {loss_val.item()}")

    def test_invalid_operator(self):
        p = nn.Parameter(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            _ = Adam([p], operator=123, lr=0.1)

    def test_none_grad(self):
        lr = 0.1
        eps = 1e-8
        betas = (0.9, 0.999)
        p = nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
        optimizer = Adam([p], lr=lr, eps=eps, betas=betas)
        old_data = p.data.clone()
        # Do not compute any gradients.
        optimizer.step()
        self.assertTrue(torch.allclose(p.data, old_data, atol=1e-6),
                        msg="Parameter data changed even though gradient was None.")

if __name__ == '__main__':
    unittest.main()
