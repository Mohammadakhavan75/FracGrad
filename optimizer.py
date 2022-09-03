from  scipy.integrate import quad
from scipy.special import gamma
import differint.differint as df
import matplotlib.pyplot as plt
from sympy import *
import numpy as np

def f(x):
    return x ** 2 + 1

def grad(x, f, epsilon=0.0001):
    return (f(x + epsilon) - f(x)) / epsilon

def optimizer(f,x0, lr, max_iter, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        x_new = x - lr * grad(x, f)

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
            
        history.append(x)
    if return_history:
        return x, history
    else:
        return x

def frac_optimizer(f,x0, lr, alpha, max_iter, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        if x>0:
            x_new = x - lr * df.RL(alpha, f, 0, float(x))[-1]
        else:
            x_new = x + lr * df.RL(alpha, f, float(x), 0)[-1]

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
        history.append(x)
    if return_history:
        return x, history
    else:
        return x

def dist_frac_optimizer(f,x0, lr, max_iter, a=0.3, b=0.95, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        if x>0:
            x_new = x - lr * quad(lambda alpha: (2*(alpha-a))/(b-a)**2 * df.RL(alpha, f, 0, float(x))[-1], a=a, b=b)[0]
        else:
            x_new = x + lr * quad(lambda alpha: (2*(alpha-a))/(b-a)**2 * df.RL(alpha, f, float(x), 0)[-1], a=a, b=b)[0]

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
        history.append(x)
    if return_history:
        return x, history
    else:
        return x

x_g, hist_g = optimizer(f, x0=10, lr=0.05, max_iter=200, return_history=True)
x_fg, hist_fg = frac_optimizer(f, x0=10, lr=0.05, alpha=0.9, max_iter=200, return_history=True)
x_dfg, hist_dfg = dist_frac_optimizer(f, x0=10, lr=0.05, max_iter=200, return_history=True)

plt.plot(hist_g)
plt.plot(hist_fg)
plt.plot(hist_dfg)


#############################################
#############################################
### Change some part of differint library ###
####### Now it can run in array mode ########
#############################################
#############################################
def f(x):
    return np.sum(x ** 2) + 1

def grad(x, f, epsilon=0.0001):
    g = np.zeros((x.shape[0],) )
    I = np.eye(x.shape[0])
    for i in range(x.shape[0]):
        g[i] = ((f(x + I[:,i]*epsilon) - f(x)) / epsilon)
        
    return g


def optimizer(f,x0, lr, max_iter, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        x_new = x - lr * grad(x, f)

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
            
        history.append(x)
    if return_history:
        return x, history
    else:
        return x

def frac_optimizer(f,x0, lr, alpha, max_iter, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        if all(xx> 0 for xx in x):
            x_new = x - lr * df.RL(alpha, f, np.zeros((x.shape[0], )), x)[-1]
        else:
            x_new = x + lr * df.RL(alpha, f, x, np.zeros((x.shape[0], )))[-1]

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
        history.append(x)
        print(x)
    if return_history:
        return x, history
    else:
        return x

def dist_frac_optimizer(f,x0, lr, max_iter, a=0.3, b=0.95, return_history=False):
    x = x0
    history = [x]
    for i in range(max_iter):
        if all(xx> 0 for xx in x) > 0:
            x_new = x - lr * quad(lambda alpha: (2*(alpha-a))/(b-a)**2 * df.RL(alpha, f, np.zeros((x.shape[0], )), x)[-1], a=a, b=b)[0]
        else:
            x_new = x + lr * quad(lambda alpha: (2*(alpha-a))/(b-a)**2 * df.RL(alpha, f, x, np.zeros((x.shape[0], )))[-1], a=a, b=b)[0]

        if f(x_new) < f(x):
            x = x_new
        else:
            lr = 0.1*lr
        history.append(x)
        print(x)
    if return_history:
        return x, history
    else:
        return x


x=np.array([10 ,10])
x0=x
x_g, hist_g = optimizer(f, x0=x, lr=0.05, max_iter=200, return_history=True)
x_fg, hist_fg = frac_optimizer(f, x0=x, lr=0.05, alpha=0.9, max_iter=200, return_history=True)
x_dfg, hist_dfg = dist_frac_optimizer(f, x0=x, lr=0.05, max_iter=200, return_history=True)