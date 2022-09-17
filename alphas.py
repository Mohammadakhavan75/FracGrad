import numpy as np
from scipy.special import gamma

# Solution 3
class alpha:
    def __init__(self):
        pass

    def case1(self, x, J, beta=0.5):
        return 1/(1 + beta * J(x))

    def case2(self, x, J, beta=0.5):
        return 2/(1 + np.exp(beta * J(x)))

    def case3(self, x, J, beta=0.5):
        return 1/(np.cosh(beta * J(x)))

    def case4(self, x, J, beta=0.5):
        return 1 - (2 / np.pi) * np.arctan(beta * J(x))
    
    def case5(self, x, J, beta=0.5):
        return 1 - np.tanh(beta * J(x))
