from scipy.special import gamma
import torch
import numpy as np

class operators():
    def __init__(self, grad_func, alpha=0.9) -> None:
        self.grad_func = grad_func
        self.alpha = alpha
        
    def integer(self, p):
        return self.grad_func(p)


    def fractional(self, p, pm_1):
        return (1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha1)))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - self.alpha1)
    