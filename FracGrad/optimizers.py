from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np
import random
import torch
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with a fractional gradient operator).

    This optimizer implements the classic stochastic gradient descent algorithm.
    If an operator function is provided, it applies a custom gradient modification that
    incorporates second-order gradient information. This can be used to implement fractional
    gradient descent methods. When no operator is provided, the optimizer defaults to standard SGD.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        operator (callable, optional): A function that accepts the current parameter tensor,
            its previous state, and the second-order gradients, returning a modified gradient tensor.
            If None, a standard SGD update is applied.
        lr (float, optional): Learning rate.
        momentum (float, optional): Momentum factor (not implemented for fractional updates).
        weight_decay (float, optional): Weight decay (L2 penalty) (not implemented for fractional updates).
        nesterov (bool, optional): Enables Nesterov momentum (not implemented for fractional updates).
        maximize (bool, optional): Maximizes the objective instead of minimizing (not implemented).
    """
    def __init__(self, params, operator=None, lr=0.03, momentum=0, weight_decay=0, nesterov=False, maximize=False):
        if operator is not None and not callable(operator):
            raise ValueError("operator must be a callable or None")
        defaults = dict(lr=lr, operator=operator, old_params={})
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss computed by the closure, if provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None: # check if any grad value are available
                    continue
                
                if group['operator'] is None:
                    # Standard SGD update.
                    p.data.add_(p.grad, alpha=-group['lr'])
                else:
                    if l not in group['old_params']: 
                        # This will run only for first iteration to store the current parameter state inside old_params.
                        group['old_params'][l] = p.data.clone().detach()
                        p.data.add_(p.grad, alpha=-group['lr'])

                    else:
                        # Compute second-order gradients.
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        # Compute gradients using the provided operator.
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        # Update the stored parameter state.
                        group['old_params'][l] = p.data.clone().detach()
                        # Updating the parameters using gradient values.
                        p.data.add_(grad_values, alpha=-group['lr'])
        return loss

class AdaGrad(Optimizer):
    r"""Implements the AdaGrad optimization algorithm with an optional fractional gradient operator.

    This optimizer adapts the learning rate for each parameter based on the historical sum
    of squared gradients. Optionally, if a custom operator is provided, the optimizer will
    modify the gradient using second-order information, enabling techniques such as fractional
    gradient descent.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        operator (callable or None): A function that takes three arguments: the current parameter tensor,
            its previous state, and the second-order gradients, returning a modified gradient tensor.
            If None, the standard AdaGrad update is used.
        lr (float, optional): Learning rate.
        eps (float, optional): Term added to the denominator to improve numerical stability.
    """
    def __init__(self, params, operator=None, lr=0.03, eps=1e-10):
        if operator is not None and not callable(operator):
            raise ValueError("operator must be a callable or None")
        defaults = dict(lr=lr, operator=operator, eps=eps, sum_of_squared_grads={}, old_params={})
        super(AdaGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            The loss computed by the closure, if provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
        
                if group['operator'] is None:
                    # Standard AdaGrad update.
                    grad_values = p.grad
                    if l not in group['sum_of_squared_grads']:
                        group['sum_of_squared_grads'][l] = torch.zeros_like(p.data.detach().cpu())
                    # Accumulate squared gradients.
                    group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1)
                    # Compute the adjusted denominator: sqrt(sum) + eps.
                    avg = group['sum_of_squared_grads'][l].sqrt().add_(group['eps']).to(grad_values.device)
                    # Update the parameter.
                    p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct.
                
                else:
                    # For the first iteration with the operator, initialize old_params.
                    if l not in group['old_params']:
                        group['old_params'][l] = p.data.clone().detach()
                        grad_values = p.grad
                        group['sum_of_squared_grads'][l] = torch.zeros_like(p.data.detach().cpu())
                        group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1)
                        avg = group['sum_of_squared_grads'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct.
                    else:
                        # Compute second-order gradients.
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        # Compute gradients using the provided operator.
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        # Update the stored parameter state.
                        group['old_params'][l] = p.data.clone().detach()
                        # Accumulate the squared gradients.
                        group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1)
                        avg = group['sum_of_squared_grads'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        # Update the parameter using the gradient.
                        p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct.

        return loss
        

class RMSProp(Optimizer):
    r"""Implements the RMSProp optimization algorithm with an optional fractional gradient operator.

    RMSProp is an adaptive learning rate method which divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.
    Optionally, if a custom operator is provided, the optimizer applies a modified gradient update using second-order gradient information,
    allowing for techniques such as fractional gradient descent.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        operator (callable or None): A function that accepts the current parameter tensor, its previous state, and the second-order gradients,
            returning a modified gradient tensor. If None, the standard RMSProp update is used.
        lr (float, optional): Learning rate (default: 0.01).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        alpha (float, optional): Smoothing constant (default: 0.99).
    """
    def __init__(self, params, operator=None, lr=0.01, eps=1e-8, alpha=0.99):
        if operator is not None and not callable(operator):
            raise ValueError("operator must be a callable or None")
        defaults = dict(lr=lr, operator=operator, eps=eps, alpha=alpha, vt={}, old_params={})
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        
        Returns:
            The loss computed by the closure, if provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # No custom operator: standard RMSProp update.
                if group['operator'] is None:
                    grad_values = p.grad
                    if l not in group['vt']:
                        group['vt'][l] = torch.zeros_like(p.data.detach().cpu())
                    # Update running average of squared gradients.
                    group['vt'][l].mul_(group['alpha']).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - group['alpha'])
                    avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                    p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct. p.addcdiv_(grad, avg, value=-lr)
                else:
                    # With a custom operator, initialize previous parameters if needed.
                    if l not in group['old_params']:
                        group['old_params'][l] = p.data.clone().detach()
                        group['vt'][l] = torch.zeros_like(p.data.detach().cpu())
                        grad_values = p.grad
                        group['vt'][l].mul_(group['alpha']).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - group['alpha'])
                        avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct. p.addcdiv_(grad, avg, value=-lr)
                    else:
                        # Compute second-order gradients.
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        # Compute second-order gradients.
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        # Update the stored parameter state.
                        group['old_params'][l] = p.data.clone().detach()
                        group['vt'][l].mul_(group['alpha']).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - group['alpha'])
                        avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg) # TODO: Check if this is correct. p.addcdiv_(grad, avg, value=-lr)
        return loss


class Adam(Optimizer):
    r"""Implements the Adam optimization algorithm with an optional fractional gradient operator.

    Adam computes adaptive learning rates for each parameter from estimates of first and second moments of the gradients.
    Optionally, if a custom operator is provided, the optimizer applies a modified gradient update based on second-order
    gradient information. This can be useful for implementing fractional gradient descent or other custom schemes.

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        operator (callable or None): A function that accepts the current parameter tensor, its previous state, and the second-order
            gradients, returning a modified gradient tensor. If None, the standard Adam update is used.
        lr (float, optional): Learning rate (default: 0.001).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square
            (default: (0.9, 0.999)).
    """
    def __init__(self, params, operator=None, lr=0.001, eps=1e-8, betas=(0.9, 0.999)):
        if operator is not None and not callable(operator):
            raise ValueError("operator must be a callable or None")
        defaults = dict(lr=lr, operator=operator, eps=eps, betas=betas,
                        mt={}, vt={}, t=0, old_params={})
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss computed by the closure, if provided; otherwise, None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            group['t'] += 1
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                else:
                    # Standard Adam update.
                    if group['operator'] is None:
                        grad_values = p.grad
                        if l not in group['mt']:
                            group['mt'][l] = torch.zeros_like(p.data.detach().cpu())
                            group['vt'][l] = torch.zeros_like(p.data.detach().cpu())
                        
                        group['mt'][l].mul_(beta1).add_(grad_values.detach().cpu() * (1 - beta1))
                        group['vt'][l].mul_(beta2).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - beta2)
                    
                        mt_hat = group['mt'][l].to(grad_values.device) / (1 - beta1 ** group['t'])
                        vt_hat = group['vt'][l].to(grad_values.device) / (1 - beta2 ** group['t'])
                        grad_values =  mt_hat / (vt_hat.sqrt().add_(group['eps']))
                        p.data.add_(grad_values, alpha=-group['lr'])

                    else:
                        # First iteration with custom operator: initialize state.
                        if l not in group['old_params']:
                            grad_values = p.grad
                            group['old_params'][l] = p.data.clone().detach()
                            group['old_params'][l].grad = p.grad.clone()
                            group['mt'][l] = torch.zeros_like(p.data.detach().cpu())
                            group['vt'][l] = torch.zeros_like(p.data.detach().cpu())
                            group['mt'][l].mul_(beta1).add_(grad_values.detach().cpu() * (1 - beta1))
                            group['vt'][l].mul_(beta2).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - beta2)

                            mt_hat = group['mt'][l].to(grad_values.device) / (1 - beta1 ** group['t'])
                            vt_hat = group['vt'][l].to(grad_values.device) / (1 - beta2 ** group['t'])
                            grad_values =  mt_hat / (vt_hat.sqrt().add_(group['eps']))
                            p.data.add_(grad_values, alpha=-group['lr'])
                        else:
                            # Compute second-order gradients and apply the custom operator.
                            second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                            grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                            group['old_params'][l] = p.data.clone().detach()
                            group['old_params'][l].grad = p.grad.clone()
                            group['mt'][l].mul_(beta1).add_(grad_values.detach().cpu() * (1 - beta1))
                            group['vt'][l].mul_(beta2).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - beta2)

                            mt_hat = group['mt'][l].to(grad_values.device) / (1 - beta1 ** group['t'])
                            vt_hat = group['vt'][l].to(grad_values.device) / (1 - beta2 ** group['t'])
                            grad_values =  mt_hat / (vt_hat.sqrt().add_(group['eps']))
                            p.data.add_(grad_values, alpha=-group['lr'])

        return loss