import torch
from torch.optim.optimizer import Optimizer, required


# class CustomGradientFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         gg = torch.ones_like(grad_input) * 10
#         # return grad_input
#         return gg
from grads import grads
from operator import operators
G = grads.grad
OPT = operators

class FiniteDifferenceGradient(torch.autograd.Function):
    def __init__(self, operator):
        self.operator = operator
    
    @staticmethod
    def forward(ctx, input, model, loss_fn, target, epsilon=1e-4):
        ctx.model = model
        ctx.loss_fn = loss_fn
        ctx.target = target
        ctx.epsilon = epsilon
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(self, ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)

        # Compute the numerical gradient
        with torch.no_grad():
            for i in range(input.size(0)):
                original_value = input[i].item()
                
                grad_input[i] = self.operator(original_value)
                # # Evaluate loss at w + epsilon
                # input[i] = original_value + ctx.epsilon
                # loss1 = ctx.loss_fn(ctx.model(input), ctx.target)

                # # Evaluate loss at w - epsilon
                # input[i] = original_value - ctx.epsilon
                # loss2 = ctx.loss_fn(ctx.model(input), ctx.target)

                # # Compute numerical gradient
                # grad_input[i] = (loss1 - loss2) / (2 * ctx.epsilon) # Operator!
                
                # Restore the original value
                input[i] = original_value

        return grad_input * grad_output

    def jvp(ctx, grad_inputs):
        # Grad_function







import torch
from torch.optim.optimizer import Optimizer, required

class CustomSGD(Optimizer):
    def __init__(self, params, model, loss_fn, target, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CustomSGD, self).__init__(params, defaults)
        self.model = model
        self.loss_fn = loss_fn
        self.target = target

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = FiniteDifferenceGradient.apply(p.grad.data, self.model, self.loss_fn, self.target)  # Apply custom gradient function
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha=-group['lr'])

        return loss



import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_optim import CustomSGD
import numpy as np
import random
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Example data
num_epochs = 10
batch_size = 64
input_size = 784
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

data = torch.randn(batch_size, input_size)
target = torch.randint(0, num_classes, (batch_size,))

criterion = nn.CrossEntropyLoss()
optimizer = CustomSGD(model.parameters(), model, criterion, target, lr=0.01, momentum=0.9)

# Generate random data for example purposes


# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data)
    loss = criterion(output, target)

    # Backward pass
    loss.backward()

    # Step with custom gradient function
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
