import torch

class fractional():
    def __init__(self, alpha=0.9) -> None:
        self.alpha = alpha

    def gradient(self, p, pm_1):
        return (1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha)))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - self.alpha)
    
    def __call__(self, p, pm_1):
        return self.gradient(p, pm_1)