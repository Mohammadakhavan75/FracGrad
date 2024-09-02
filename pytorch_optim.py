from torch.optim.optimizer import Optimizer
from operators import operators
import torch.nn.functional as F
import numpy as np
import random
import torch
import copy


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
                        # group['old_params'][l].grad = p.grad.clone()
                        p.data.add_(p.grad, alpha=-group['lr'])
                    else: # continue_of_the_iterations
                        # flat_grads = p.grad.contiguous().view(-1)
                        # flat_params = p.contiguous().view(-1)
                        
                        # vector = torch.ones_like(p.grad)
                        # second_order_grads = torch.zeros_like(p.grad)
                        # for i, (flat_grad, flat_param) in enumerate(zip(p.grad, p)):
                        #     print(p.grad_fn, p.requires_grad, p.retain_grad())
                        #     print(flat_param.grad_fn, flat_param.requires_grad, flat_param.retain_grad())
                        #     vector = torch.ones_like(flat_grad)
                        #     second_order_grads[i] = torch.autograd.grad(flat_grad.dot(vector), flat_param, retain_graph=True)[0]
                        
                        # vector = [torch.randn_like(p) for p in model.parameters()]
                        vector = torch.ones_like(p)
                        grad_vector_product = sum(torch.sum(g * v) for g, v in zip(p.grad, vector))
                        # Compute Hessian-vector product
                        second_order_grads = torch.autograd.grad(grad_vector_product, p, retain_graph=True)[0]

                        # second_order_grads = torch.autograd.grad(p.grad.dot(vector), p)[0]
                        grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                        # print(torch.mean(grad_values))
                        group['old_params'][l] = p.data.clone().detach()
                        # group['old_params'][l].grad = p.grad.clone()
                        p.data.add_(grad_values, alpha=-group['lr'])
                    
        return loss


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
                
                if l not in group['old_params']:
                    group['old_params'][l] = p.data.clone().detach()
                    p.data.add_(p.grad, alpha=-group['lr'])
                else:
                    vector = torch.ones_like(p)
                    grad_vector_product = sum(torch.sum(g * v) for g, v in zip(p.grad, vector))
                    second_order_grads = torch.autograd.grad(grad_vector_product, p, retain_graph=True)[0]
                    grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                    
                    group['old_params'][l] = p.data.clone().detach()
                    group['sum_of_squared_grads'][l].add_(torch.pow(grad_values, 2))
                    adjusted_lr = 1 / (group['sum_of_squared_grads'][l].sqrt() + group['eps'])
                    grad_values = grad_values * adjusted_lr
                    p.data.add_(grad_values, alpha=-group['lr'])
        return loss
    
import torch
from torch.optim.optimizer import Optimizer

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

                if l not in group['old_params']:
                    group['old_params'][l] = p.data.clone().detach()
                    p.data.add_(p.grad, alpha=-group['lr'])
                else:
                    vector = torch.ones_like(p)
                    grad_vector_product = sum(torch.sum(g * v) for g, v in zip(p.grad, vector))
                    # Compute Hessian-vector product
                    second_order_grads = torch.autograd.grad(grad_vector_product, p, retain_graph=True)[0]
                    grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                    group['old_params'][l] = p.data.clone().detach()
                    group['accumulated_grad'][l] = group['accumulated_grad'][l] + group['alpha'] * group['accumulated_grad'][l] + (1 - group['alpha']) * (grad_values ** 2)

                scaled_grad = grad_values/(group['accumulated_grad'][l].sqrt() + group['eps'])
                p.data.add_(scaled_grad, alpha=-group['lr'])

        return loss


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

                
                if l not in group['old_params']:
                    group['old_params'][l] = p.data.clone().detach()
                    p.data.add_(p.grad, alpha=-group['lr'])
                    group['moment1'][l] = torch.zeros_like(p.data)
                    group['moment2'][l] = torch.zeros_like(p.data)

                vector = torch.ones_like(p)
                grad_vector_product = sum(torch.sum(g * v) for g, v in zip(p.grad, vector))
                group['old_params'][l] = p.data.clone().detach()
                second_order_grads = torch.autograd.grad(grad_vector_product, p, retain_graph=True)[0]
                grad_values = group['operator'](p, group['old_params'][l], second_order_grads)
                
                group['moment1'][l] = beta1 * group['moment1'][l] + (1 - beta1) * grad_values
                group['moment2'][l] = beta2 * group['moment2'][l] + (1 - beta2) * (grad_values ** 2)

                m1_hat = group['moment1'][l] / (1 - beta1 ** group['t'])
                m2_hat = group['moment2'][l] / (1 - beta2 ** group['t'])

                scaled_grad =  m1_hat / (m2_hat.sqrt() + group['eps'])
                p.data.add_(scaled_grad, alpha=-group['lr'])

        return loss

                
# class SimpleModel(torch.nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = torch.nn.Linear(784, 128)
#         self.fc2 = torch.nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = SimpleModel()

# # Example data
# num_epochs = 5
# batch_size = input_size = 784
# num_classes = 10
# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)

# data = torch.randn(batch_size, input_size)
# target = torch.randint(0, num_classes, (batch_size,))

# criterion = torch.nn.CrossEntropyLoss()
# operator_instance = operators(grad_func=torch.autograd.grad)
# optimizer = SGD(model.parameters(), lr=0.01, operator=operator_instance.fractional)  # Use the integer method for example

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()

#     # Forward pass
#     output = model(data) 
#     loss = criterion(output, target)

#     # Backward pass
#     loss.backward()

#     # Step with custom gradient function
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")