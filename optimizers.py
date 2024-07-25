import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt


class SGD():
    def __init__(self, operator):
        self.operator = operator
    
    def optimize(self, w_old, model, lr=0.03, max_iter=10, return_history=False):
        history = [w_old]
        for i in range(max_iter):
            w_new = w_old - lr * self.operator(history, i, lr)
            w_old = w_new
            model.w_flatten = w_new.copy()
       

class Adagrad():
    def __init__(self, operator):
        self.operator = operator
    
    def optimize(self, w_old, model, lr=0.03, max_iter=10, epsilon=1e-8, return_history=False):
        history = [w_old]
        grad_squared_accum = np.zeros_like(w_old)  # Initialize the accumulator
        
        for i in range(max_iter):
            grad = self.operator(history, i, lr)
            grad_squared_accum += grad ** 2  # Accumulate the squared gradients
            adjusted_lr = lr / (np.sqrt(grad_squared_accum) + epsilon)  # Adjust learning rate
            
            w_new = w_old - adjusted_lr * grad
            w_old = w_new
            model.w_flatten = w_new.copy()


class RMSProp():
    def __init__(self, operator):
        self.operator = operator
    
    def optimize(self, w_old, model, lr=0.001, beta=0.9, epsilon=1e-8, max_iter=10, return_history=False):
        history = [w_old]
        grad_squared_accum = np.zeros_like(w_old)  # Initialize the accumulator
        
        for i in range(max_iter):
            grad = self.operator(history, i, lr)
            grad_squared_accum = beta * grad_squared_accum + (1 - beta) * grad ** 2  # Exponentially decay the average of squared gradients
            adjusted_lr = lr / (np.sqrt(grad_squared_accum) + epsilon)  # Adjust learning rate
            
            w_new = w_old - adjusted_lr * grad
            w_old = w_new
            model.w_flatten = w_new.copy()

class Adam():
    def __init__(self, operator):
        self.operator = operator
    
    def optimize(self, w_old, model, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=10, return_history=False):
        history = [w_old]
        m = np.zeros_like(w_old)  # Initialize the first moment vector
        v = np.zeros_like(w_old)  # Initialize the second moment vector
        
        for i in range(1, max_iter + 1):
            grad = self.operator(history, i, lr)
            
            m = beta1 * m + (1 - beta1) * grad  # Update biased first moment estimate
            v = beta2 * v + (1 - beta2) * grad ** 2  # Update biased second moment estimate
            
            m_hat = m / (1 - beta1 ** i)  # Correct bias in first moment
            v_hat = v / (1 - beta2 ** i)  # Correct bias in second moment
            
            w_new = w_old - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            w_old = w_new
            model.w_flatten = w_new.copy()
            



def gen_frac_opt(self, f, x, mdoel, D, lr=0.3, max_iter=10):
    history = [x]
    for _ in range(max_iter):
        w_new = x - lr * D
        if f(w_new) < f(x):
            x = w_new
            mdoel.w_flatten = w_new.copy()
        else:
            lr = 0.8*lr
            if isinstance(lr, list):
                if np.mean(lr) < 0.1 ** 12:
                    break
            elif lr < 0.1 ** 12:
                break
            
        history.append(w_new)

def easy_frac_opt(self, f, x, D, lr=0.3, max_iter=10):
    for _ in range(max_iter):
        w_new = x.copy()
        for i in range(x.shape[0]):
            w_new[i] = x[i] - lr * D(f, x[i])

        if f(w_new) < f(x):
            x = w_new
        else:
            lr = 0.8 * lr
            if isinstance(lr, list):
                if np.mean(lr) < 0.1 ** 12:
                    break
            elif lr < 0.1 ** 12:
                break
