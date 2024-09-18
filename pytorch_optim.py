from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np
import random
import torch

from torch.optim.optimizer import Optimizer


class grad_generator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input 

    @staticmethod

    def backward(ctx, grad_output):
        # Add a constant to the gradient
        # return grad_output * 1000
        return grad_output


class SGD(Optimizer):
    def __init__(self, params, operator=None, lr=0.03):
        self.start_epoch = True
        defaults = dict(lr=lr, operator=operator, old_params={})
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                if group['operator'] is None:
                    p.data.add_(p.grad, alpha=-group['lr'])
                
                else:
                    if l not in group['old_params']:# FIRST_ITERATION
                        group['old_params'][l] = p.data.clone().detach()
                        p.data.add_(p.grad, alpha=-group['lr'])

                    else: # continue_of_the_iterations
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        group['old_params'][l] = p.data.clone().detach()
                        p.data.add_(grad_values, alpha=-group['lr'])
                    

class AdaGrad(Optimizer):
    def __init__(self, params, operator, lr=0.03, eps=1e-10):
        defaults = dict(lr=lr, operator=operator, eps=eps, sum_of_squared_grads={}, old_params={})
        super(AdaGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
        
                if group['operator'] is None:
                    p.data.add_(p.grad, alpha=-group['lr'])   
                
                else:
                    if l not in group['old_params']:
                        group['old_params'][l] = p.data.clone().detach()
                        grad_values = p.grad
                        group['sum_of_squared_grads'][l] = torch.pow(grad_values.detach().cpu(), 2)
                        adjusted_lr = 1 / (group['sum_of_squared_grads'][l].sqrt() + group['eps'])
                        grad_values = grad_values * adjusted_lr.to(grad_values.device)
                        p.data.add_(grad_values, alpha=-group['lr'])
                    else:
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        group['old_params'][l] = p.data.clone().detach()
                        group['sum_of_squared_grads'][l].add_(torch.pow(grad_values.detach().cpu(), 2))
                        adjusted_lr = 1 / (group['sum_of_squared_grads'][l].sqrt() + group['eps'])
                        grad_values = grad_values * adjusted_lr.to(grad_values.device)
                        p.data.add_(grad_values, alpha=-group['lr'])
    

class RMSProp(Optimizer):
    def __init__(self, params, operator, lr=0.01, eps=1e-8, alpha=0.99):
        defaults = dict(lr=lr, operator=operator, eps=eps, alpha=alpha, accumulated_grad={}, old_params={})
        super(RMSProp, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                else:
                    if group['operator'] is None:
                        p.data.add_(p.grad, alpha=-group['lr'])   
                        
                    else:
                        if l not in group['old_params']:
                            group['old_params'][l] = p.data.clone().detach()
                            grad_values = p.grad
                            group['accumulated_grad'][l] = (1 - group['alpha']) * torch.pow(grad_values.detach().cpu(), 2)
                            scaled_grad = grad_values / (group['accumulated_grad'][l].sqrt().to(grad_values.device) + group['eps'])
                            p.data.add_(scaled_grad, alpha=-group['lr'])
                        else:
                            second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                            grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                            group['old_params'][l] = p.data.clone().detach()
                            group['accumulated_grad'][l].add_(group['alpha'] * group['accumulated_grad'][l] + (1 - group['alpha']) * torch.pow(grad_values.detach().cpu(), 2))

                            scaled_grad = grad_values/(group['accumulated_grad'][l].sqrt().to(grad_values.device) + group['eps'])
                            
                            p.data.add_(scaled_grad, alpha=-group['lr'])


class Adam(Optimizer):
    def __init__(self, params, operator, lr=0.001, eps=1e-8, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, operator=operator, eps=eps, betas=betas,
                        moment1={}, moment2={}, t=0, old_params={})
        super(Adam, self).__init__(params, defaults)


    def step(self, closure=None):
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
                    if group['operator'] is None:
                        p.data.add_(p.grad, alpha=-group['lr'])   
                        
                    else:
                        if l not in group['old_params']:
                            group['old_params'][l] = p.data.clone().detach()
                            group['old_params'][l].grad = p.grad.clone()
                            group['moment1'][l] = torch.zeros_like(p.data)
                            group['moment2'][l] = torch.zeros_like(p.data)
                            grad_values = p.grad
                            
                            group['moment1'][l] = beta1 * group['moment1'][l].detach().cpu() + (1 - beta1) * grad_values.detach().cpu()
                            group['moment2'][l] = beta2 * group['moment2'][l].detach().cpu() + (1 - beta2) * torch.pow(grad_values.detach().cpu(), 2)

                            m1_hat = group['moment1'][l].cuda() / (1 - beta1 ** group['t'])
                            m2_hat = group['moment2'][l].cuda() / (1 - beta2 ** group['t'])

                            scaled_grad =  m1_hat / (m2_hat.sqrt() + group['eps'])
                            p.data.add_(scaled_grad, alpha=-group['lr'])
                            
                        else:
                            second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                            grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                            group['old_params'][l] = p.data.clone().detach()
                            group['old_params'][l].grad = p.grad.clone()
                            
                            group['moment1'][l] = beta1 * group['moment1'][l].detach().cpu() + (1 - beta1) * grad_values.detach().cpu()
                            group['moment2'][l] = beta2 * group['moment2'][l].detach().cpu() + (1 - beta2) * torch.pow(grad_values.detach().cpu(), 2)

                            m1_hat = group['moment1'][l].to(grad_values.device) / (1 - beta1 ** group['t'])
                            m2_hat = group['moment2'][l].to(grad_values.device) / (1 - beta2 ** group['t'])

                            scaled_grad =  m1_hat / (m2_hat.sqrt() + group['eps'])
                            p.data.add_(scaled_grad, alpha=-group['lr'])
        
        group['t'] = 0