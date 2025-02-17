
# Fractional Gradient Descent Optimizers for PyTorch

This package implements a novel approach to gradient descent by incorporating fractional derivatives into the update rules of popular optimization algorithms. Built on top of PyTorch, the optimizers in this package (SGD, AdaGrad, RMSProp, and Adam) can leverage both CPU and CUDA devices, making them versatile for a wide range of applications.

## Overview

Fractional derivatives extend the traditional concept of differentiation, offering a more generalized framework that can capture memory and hereditary properties of complex systems. In this package, a custom fractional gradient operator is provided that modifies the gradient computation based on a user-defined fractional order (alpha). This operator can be optionally integrated into any of the available optimizers, allowing for experimental research into fractional gradient descent methods.

## Features

- **Multiple Optimizers:** Custom implementations for SGD, AdaGrad, RMSProp, and Adam.

- **Fractional Derivative Operator:** Modify gradient updates using a fractional derivative, with an adjustable parameter alpha.

- **Seamless Integration:** Easily swap between standard and fractional gradient descent by providing (or omitting) the operator.

- **PyTorch-Based:** Built on top of PyTorch, ensuring compatibility with existing models and the autograd system.

- **CPU and CUDA Support:** Run your experiments on both CPU and GPU.

## Installation

Ensure you have PyTorch installed. You can install PyTorch by following the instructions at [PyTorch.org](https://pytorch.org/).

Clone this repository and add it to your Python path:
```bash
pip install FracGrad
```

## Usage

To use the fractional optimizers, import the desired optimizer and the fractional operator, then pass your model parameters and operator to the optimizer.

### Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from FracGrad import SGD, AdaGrad, RMSProp, Adam
from operators import fractional

# Define a simple model
model = nn.Linear(10, 1)

# Choose a fractional operator with a specific order (alpha)
frac_operator = fractional(alpha=0.9)

# Initialize an optimizer; here we use SGD with the fractional operator
optimizer = SGD(model.parameters(), operator=frac_operator, lr=0.03)

# Training loop example
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

```

## File Structure

- **operators.py:** Contains the implementation of the fractional class, which defines the fractional derivative operator. This operator adjusts the gradient based on the fractional order.

- **optimizers.py:** Implements custom versions of standard optimizers (SGD, AdaGrad, RMSProp, and Adam). Each optimizer is designed to optionally use the fractional operator for modified gradient updates.

## Contributing

Contributions, suggestions, and bug reports are welcome! Feel free to open an issue or submit a pull request.