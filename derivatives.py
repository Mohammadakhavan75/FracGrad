import numpy as np
from scipy.special import gamma

class local:
    def __init__(self):
        pass
    
    def K(self, x):
        return x

    def Kprim(self, x):
        return int(1)

    def grad(self, f, x, epsilon=0.1 ** 4):
        return (f(x + epsilon) - f(x)) / epsilon
        
    def chen(f, x, alpha=0.9):
        out = x.copy()
        s = x[x>0] - 0.1**4
        out[x>0] = (f(x[x>0]) - f(s)) / (x[x>0] ** alpha - s ** alpha)
        s = x[x<0] + 0.1**4
        out[x<0] = (f(x[x<0]) - f(s)) / (s ** alpha - x[x<0] ** alpha)

        return out


    def conformable(self, f, x, alpha=0.9, epsilon=0.1 ** 4):
        out = (f(x + epsilon * x ** (1-alpha)) - f(x)) / epsilon
        return out

    def katugampola(self, f, x, alpha=0.9, epsilon=0.1 ** 4):
        out = (f(x * np.exp(epsilon * x ** (-alpha))) - f(x)) / epsilon
        return out

    def deformable(self, f, x, beta=0.1, alpha=0.9, epsilon=0.1 ** 4):
        out = ((1 + epsilon * beta) * f(x + epsilon * alpha) - f(x)) / epsilon
        return out

    def beta(self, f, x, alpha=0.9, epsilon=0.1 ** 4):
        out = (f(x + epsilon * (x + (1 / gamma(alpha))) ** (1 - alpha)) - f(x)) / epsilon
        return out

    def AGO(self, f, x, K, alpha=0.9, epsilon=0.1 ** 4):
        out = (f(x + epsilon * K(x) ** (1 - alpha)) - f(x)) / epsilon
        return out

    def Generalized(self, f, x, K, Kprim, alpha=0.9, epsilon=0.1 ** 4):
        out = (f(x - K(x) + K(x) * np.exp((epsilon * K(x) ** (- alpha))/Kprim(x))) - f(x)) / epsilon 
        return out

    # def general_conformable(self,f, x, alpha, epsilon=0.1 ** 4):
    #     out = (f(x + epsilon * SI(x, alpha)) - f(x)) / epsilon
    #     return out

class non_singular:
    def __init__(self):
        pass