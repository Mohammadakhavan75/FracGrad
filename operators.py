from scipy.special import gamma
import numpy as np

class operators():
    def __init__(self, grad_func, alpha1=0.9, alpha2=1.1, N=50) -> None:
        self.grad_func = grad_func
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.N = N
        
    def integer(self, p, pm_1, lr):
        return self.grad_func(p)


    def fractional(self, p, pm_1, lr):
        return pm_1 - (lr / gamma(2 - self.alpha1)) * self.grad_func(p) * np.abs(pm_1 - p) ** (1 - self.alpha1)


    def multi_fractional(self, p, pm_1, lr):
        t1 = (1/gamma(2 - self.alpha1)) * self.grad_func(p) * np.abs(pm_1 - p) ** (1 - self.alpha1)
        t2 = (1/gamma(2 - self.alpha2)) * self.grad_func(p) * np.abs(pm_1 - p) ** (1 - self.alpha2)

        return 0.5 * t1 + 0.5 * t2


    def distributed_fractional(self, p, pm_1, lr):
        d_alpha = (self.alpha2 - self.alpha1) / self.N
        integral = lambda alpha: ((2 * (alpha - self.alpha1)) / (gamma(2 - alpha) * (self.alpha2 - self.alpha1) ** 2)) * self.grad_func(p) * np.abs(pm_1 - p) ** (1 - alpha)
        
        delta = 0.5 * integral(self.alpha1)
        for n in range(1, self.N):
            delta += integral(self.alpha1 + n * d_alpha)
        
        delta += 0.5 * integral(self.alpha2)

        return delta
