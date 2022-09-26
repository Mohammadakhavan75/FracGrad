import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self):
        pass
    
    def grad(self, f, x, epsilon=0.0001):
        g = np.zeros((x.shape[0],) )

        x_temp = x.copy()

        for i in range(x.shape[0]):
            x_temp[i] += epsilon
            if i>0:
                x_temp[i-1] -= epsilon
            g[i] = ((f(x_temp) - f(x)) / epsilon) 
        return g


    def optimizer(self, f, x0, model, lr=0.1, max_iter=500, return_history=False):
        x = x0
        history = [x]
        for i in range(max_iter):
            x_new = x - lr * self.grad(f, x)

            if f(x_new) < f(x):
                x = x_new
                model.w_flatten = x_new.copy()
            else:
                # print(f"Updating learning rate: {lr}")
                lr = 0.8*lr
                if lr < 0.1 ** 12:
                    # print("Early STOP!!!")
                    break
                
            history.append(x)
        # if return_history:
        #     return x, history
        # else:
        #     return x

    def frac_optimizer(self, f, x0, model, lr=0.5, alpha=0.98, max_iter=1000, return_history=False):
        x = x0
        history = [x]
        x_new = x - lr * self.grad(f, x)
        history.append(x_new)
        indx = 0
        for i in range(max_iter):
        
            x_new_new = history[indx+1] - (lr/gamma(2-alpha)) * self.grad(f, history[indx]) * np.abs(history[indx+1] - history[indx]) ** (1-alpha)
            if f(x_new_new) < f(history[indx+1]):
                history.append(x_new_new)
                model.w_flatten = x_new_new.copy()
                indx += 1
            else:
                
                # print(f"Updating learning rate: {lr}")
                lr = 0.8*lr
                if lr < 0.1 ** 12:
                    # print("Early STOP!!!")
                    break
            # history.append(x)
            # print(x)
        # if return_history:
        #     return history[-1], history
        # else:
        #     return history[-1]

    def multi_frac_optimizer(self, f, x0, model, lr=0.5, alpha1=0.9, alpha2=1.1, max_iter=1000, return_history=False):
        x = x0
        history = [x]
        x_new = x - lr * self.grad(f, x)
        history.append(x_new)
        indx = 0
        for i in range(max_iter):

            t1 = (1/gamma(2-alpha1)) * self.grad(f, history[indx]) * np.abs(history[indx+1] - history[indx]) ** (1- alpha1)
            t2 = (1/gamma(2-alpha2)) * self.grad(f, history[indx]) * np.abs(history[indx+1] - history[indx]) ** (1- alpha2)
        
            x_new_new = history[indx+1] - lr*(0.5*t1 + 0.5*t2)
            if f(x_new_new) < f(history[indx+1]):
                history.append(x_new_new)
                model.w_flatten = x_new_new.copy()
                indx += 1
            else:
                
                # print(f"Updating learning rate: {lr}")
                lr = 0.8*lr
                if lr < 0.1 ** 12:
                    # print("Early STOP!!!")
                    break
            # history.append(x)
            # print(x)
        # if return_history:
        #     return history[-1], history
        # else:
        #     return history[-1]

    def dist_frac_optimizer(self, f, x0, model, lr=0.5, alpha1=0.9, alpha2=1.1, N=50, max_iter=1000, return_history=False):
        x = x0
        history = [x]
        x_new = x - lr * self.grad(f, x)
        history.append(x_new)
        indx = 0

        for i in range(max_iter):

            d_alpha = (alpha2-alpha1)/N
            
            # d = lambda alpha: (1/gamma(2-alpha)) * grad(f, history[indx]) * np.abs(history[indx+1] - history[indx]) ** (1-alpha)
            d = lambda alpha: ((2*(alpha-alpha1))/(gamma(2-alpha)*(alpha2-alpha1)**2)) * self.grad(f, history[indx]) * np.abs(history[indx+1] - history[indx]) ** (1-alpha)

            delta = 0.5*d(alpha1)
            for n in range(1, N):
                delta = delta + d(alpha1 + n*d_alpha)
            delta = 0.5*d(alpha2)

            x_new_new = history[indx+1] - lr*delta
            if f(x_new_new) < f(history[indx+1]):
                history.append(x_new_new)
                model.w_flatten = x_new_new.copy()
                indx += 1
            else:
                # print(f"Updating learning rate: {lr}")
                lr = 0.8*lr
                if lr < 0.1 ** 12:
                    # print("Early STOP!!!")
                    break
            # history.append(x)
            # print(x)
        # if return_history:
        #     return history[-1], history
        # else:
        #     return history[-1]

    def gen_frac_opt(self, f, x, mdoel, D, lr=0.3, max_iter=10):
        history = [x]
        for _ in range(max_iter):
            x_new = x - lr * D(f, x)
            if f(x_new) < f(x):
                x = x_new
                mdoel.w_flatten = x_new.copy()
            else:
                lr = 0.8*lr
                if isinstance(lr, list):
                    if np.mean(lr) < 0.1 ** 12:
                        break
                elif lr < 0.1 ** 12:
                    break
                
            history.append(x_new)

    def Ggamma(self, p, q):
        return gamma(p + 1)/(gamma(q + 1) * gamma(p - q + 1))
    
    # Solution 2
    def Glearning_rate(self, xk, alpha=0.9, c=0, epsilon=0.0001):
        return (1/gamma(2-alpha)) * (np.abs(xk - c) + epsilon) ** (1 - alpha)

    def Reimann_Liouville(self, fi, x, alpha=0.9, N=1, c=0):
        result = []
        for i in range(N):
            result.append((fi(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)

        return np.sum(np.asarray(result))

    def Caputo(self, fi, x, alpha=0.9, n=0, N=1, c=0):
        result = []
        for i in range(n, N):
            result.append((fi(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)
        
        return np.sum(np.asarray(result))

    def Reimann_Liouville_fromG(self, fi, x, alpha=0.9, N=1, c=0):
        result = []
        for i in range(N):
            result.append(self.Ggamma(p=alpha, q=i) * (fi(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)

        return np.sum(np.asarray(result))

    def Caputo_fromG(self, fi, x, alpha=0.9, n=0, N=1, c=0):
        result = []
        for i in range(n, N):
            result.append(self.Ggamma(p=alpha-n, q=i-n) * (fi(x) /gamma(i + 1 - alpha)) * (x - c) ** i-alpha)
        
        return np.sum(np.asarray(result))