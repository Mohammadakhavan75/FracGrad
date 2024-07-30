from torch.optim.optimizer import Optimizer
from operators import operators
import torch.nn.functional as F
import numpy as np
import random
import torch
import itertools
import copy
class grad_generator(torch.autograd.Function):
    # def __init__(self, operator, pm_1=None, lr=None):
        # self.operator = operator
        # self.pm_1 = pm_1
        # self.lr = lr
    
    @staticmethod
    def forward(ctx, input, operator):
        ctx.save_for_backward(input)
        ctx.operator = operator
        
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        # Compute the numerical gradient
        with torch.no_grad():
            for i in range(input.size(0)):
                original_value = input[i].item()
                grad_input[i] = ctx.operator(original_value)
                # Restore the original value
                input[i] = original_value

        return grad_input * grad_output


def deep_copy_generator(gen):
    for item in gen:
        yield copy.deepcopy(item)

# Use itertools.tee on the new generator

def my_generator(params):
    for p in list(params):
        yield copy.deepcopy(p)


class SGD(Optimizer):
    def __init__(self, params, operator, lr=0.03):
        # Create a generator
        params_ = my_generator(params)

        # To "copy" the generator, simply create a new one
        self.params_1 = my_generator(params)
        

        defaults = dict(lr=lr, operator=operator)
        super(SGD, self).__init__(params_, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            print("#1")
            for p, t in zip(group['params'], np.arange(10)):
            # for p in group['params']:
                if p.grad is None:
                    continue
                print("#3")
                grad_values = grad_generator.apply(p, group['operator'])
                p.data.add_(grad_values, alpha=-group['lr'])
                # print(p.data[0])#, pm_1[0][0])
                
                
        return loss





class CustomSGD(Optimizer):
    def __init__(self, params, operator, lr=0.03, alpha1=0.9, alpha2=1.1, N=50, history_size=2):
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)
        self.operator = operators(self.grad_func, alpha1, alpha2, N)
        self.operator_name = operator
        self.history = {}
        self.history_size = history_size

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                param_id = id(p)
                if param_id not in self.history:
                    self.history[param_id] = [p.data.clone()]
                else:
                    if len(self.history[param_id]) >= self.history_size:
                        self.history[param_id].pop(0)
                    self.history[param_id].append(p.data.clone())

                if len(self.history[param_id]) >= self.history_size:
                    history = self.history[param_id]
                    idx = -2
                    if self.operator_name == "integer":
                        update = self.operator.integer(history, idx, lr)
                    elif self.operator_name == "fractional":
                        update = self.operator.fractional(history, idx, lr)
                    elif self.operator_name == "multi_fractional":
                        update = self.operator.multi_fractional(history, idx, lr)
                    elif self.operator_name == "distributed_fractional":
                        update = self.operator.distributed_fractional(history, idx, lr)
                    else:
                        raise ValueError(f"Unknown operator: {self.operator_name}")

                    p.data.add_(-lr, torch.tensor(update, dtype=p.data.dtype, device=p.data.device))

        return loss


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Example data
num_epochs = 2
batch_size = 64
input_size = 784
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

data = torch.randn(batch_size, input_size)
target = torch.randint(0, num_classes, (batch_size,))

criterion = torch.nn.CrossEntropyLoss()
operator_instance = operators(grad_func=torch.autograd.grad)
optimizer = SGD(model.parameters(), lr=0.01, operator=operator_instance.integer)  # Use the integer method for example

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






