import numpy as np
from scipy.special import gamma

class grads():
    def __init__(self, loss_func) -> None:
        self.loss_func = loss_func
    
    # Simple Grad
    def grad(self, x, epsilon=1e-9):
        return ((self.loss_func(x + epsilon) - self.loss_func(x)) / epsilon) 
    
    def Ggamma(self, p, q):
        return gamma(p + 1)/(gamma(q + 1) * gamma(p - q + 1))
    
    # Solution 2
    def Glearning_rate(self, xk, alpha=0.9, c=0, epsilon=0.0001):
        return (1/gamma(2-alpha)) * (np.abs(xk - c) + epsilon) ** (1 - alpha)

    def Reimann_Liouville(self, x, alpha=0.9, N=1, c=0):
        result = []
        for i in range(N):
            result.append((self.loss_func(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)

        return np.sum(np.asarray(result))

    def Caputo(self, x, alpha=0.9, n=0, N=1, c=0):
        result = []
        for i in range(n, N):
            result.append((self.loss_func(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)
        
        return np.sum(np.asarray(result))

    def Reimann_Liouville_fromG(self, x, alpha=0.9, N=1, c=0):
        result = []
        for i in range(N):
            result.append(self.Ggamma(p=alpha, q=i) * (self.loss_func(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)

        return np.sum(np.asarray(result))

    def Caputo_fromG(self, x, alpha=0.9, n=0, N=1, c=0):
        result = []
        for i in range(n, N):
            result.append(self.Ggamma(p=alpha-n, q=i-n) * (self.loss_func(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)
        
        return np.sum(np.asarray(result))