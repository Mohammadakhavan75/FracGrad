from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
import numpy as np
import random
import torch

class grad_generator(torch.autograd.Function):
    def __init__(self, operator):
        self.operator = operator
    
    @staticmethod
    def forward(ctx, input):
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

                # Restore the original value
                input[i] = original_value

        return grad_input * grad_output


class SGD(Optimizer):
    def __init__(self, params, operator, lr=0.03):
        defaults = dict(lr=lr, operator=operator, params_1=params, params_2=params)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            operator = group['operator']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_values = grad_generator.apply(p.grad.data)  # Apply custom gradient function
                p.data.add_(grad_values, alpha=-group['lr'])

        return loss


class SimpleModel(nn.Module):
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
num_epochs = 10
batch_size = 64
input_size = 784
num_classes = 10
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

data = torch.randn(batch_size, input_size)
target = torch.randint(0, num_classes, (batch_size,))

criterion = torch.nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, operator=None)

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
