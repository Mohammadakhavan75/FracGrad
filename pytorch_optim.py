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
    def __init__(self, params, operator=None, lr=0.03, momentum=0, weight_decay=0, nesterov=False, maximize=False):
        defaults = dict(lr=lr, operator=operator, old_params={})
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None: # check if any grad value are available
                    continue
                
                if group['operator'] is None:
                    p.data.add_(p.grad, alpha=-group['lr'])
                
                else:
                    if l not in group['old_params']: # this will run only for first iteration to fill the old_params value
                        group['old_params'][l] = p.data.clone().detach()
                        p.data.add_(p.grad, alpha=-group['lr'])

                    else: # continue_of_the_iterations
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads) # calculating grads using new operators
                        group['old_params'][l] = p.data.clone().detach() # updating old parameters
                        p.data.add_(grad_values, alpha=-group['lr']) # updating the params based on grad values


class AdaGrad(Optimizer):
    def __init__(self, params, operator, lr=0.03, eps=1e-10):
        defaults = dict(lr=lr, operator=operator, eps=eps, sum_of_squared_grads={}, old_params={})
        super(AdaGrad, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
        
                if group['operator'] is None:
                    grad_values = p.grad
                    if l not in group['sum_of_squared_grads']:
                        group['sum_of_squared_grads'][l] = torch.zeros_like(p.data.detach().cpu())

                    group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=0)
                    # multiply the adjustmnet of lr into grad values
                    avg = group['sum_of_squared_grads'][l].sqrt().add_(group['eps']).to(grad_values.device)
                    p.data.addcdiv_(-group['lr'], grad_values, avg)
                
                else:
                    if l not in group['old_params']:
                        group['old_params'][l] = p.data.clone().detach()
                        grad_values = p.grad
                        group['sum_of_squared_grads'][l] = torch.zeros_like(p.data.detach().cpu())
                        group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=0)
                        p.data.addcdiv_(-group['lr'], grad_values, avg)
                    else:
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        group['old_params'][l] = p.data.clone().detach()
                        group['sum_of_squared_grads'][l].addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=0)
                        avg = group['sum_of_squared_grads'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg)
        

class RMSProp(Optimizer):
    def __init__(self, params, operator, lr=0.01, eps=1e-8, alpha=0.99):
        defaults = dict(lr=lr, operator=operator, eps=eps, alpha=alpha, vt={}, old_params={})
        super(RMSProp, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                if group['operator'] is None:
                    grad_values = p.grad
                    if l not in group['vt']:
                        group['vt'][l] = torch.zeros_like(p.data.detach().cpu())

                    group['vt'][l].mul_(group['alpha']).addcmul_(grad_values, grad_values, value=1 - group['alpha'])
                    avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                    p.data.addcdiv_(-group['lr'], grad_values, avg)
                    
                else:
                    if l not in group['old_params']:
                        group['old_params'][l] = p.data.clone().detach()
                        group['vt'][l] = torch.zeros_like(p.data.detach().cpu())
                        grad_values = p.grad
                        group['vt'][l].mul_(group['alpha']).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - group['alpha'])
                        avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg)
                    else:
                        second_order_grads = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        group['old_params'][l] = p.data.clone().detach()
                        group['vt'][l].mul_(group['alpha']).addcmul_(grad_values.detach().cpu(), grad_values.detach().cpu(), value=1 - group['alpha'])
                        avg = group['vt'][l].sqrt().add_(group['eps']).to(grad_values.device)
                        p.data.addcdiv_(-group['lr'], grad_values, avg)
        return loss


class Adam(Optimizer):
    def __init__(self, params, operator, lr=0.001, eps=1e-8, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, operator=operator, eps=eps, betas=betas,
                        mt={}, vt={}, t=0, old_params={})
        super(Adam, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            group['t'] += 1
            for l, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                else:
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