from scipy.special import gamma
import torch
import numpy as np

class operators():
    def __init__(self, grad_func, alpha1=0.9, alpha2=1.1, N=50) -> None:
        self.grad_func = grad_func
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.N = N
        
    def integer(self, p, pm_1, second_order_grad):
        return self.grad_func(p)


    def fractional(self, p, pm_1, second_order_grad):
        return (1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha1)))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - self.alpha1)
    

    def multi_fractional(self, p, pm_1, second_order_grad):
        t1 = (1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha1)))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - self.alpha1)
        
        if 1 - self.alpha2 > 0 :
            t2 = (1 / torch.exp(torch.lgamma(torch.tensor(2 - self.alpha2)))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - self.alpha2)
        else:
            t2 = (1 / (torch.exp(torch.lgamma(torch.tensor(3 - self.alpha2)))))  * second_order_grad * torch.abs(p.data.detach() - pm_1.data.detach()) ** (2 - self.alpha2)
        
        return 0.5 * t1 + 0.5 * t2


    def distributed_fractional(self, p, pm_1, second_order_grad):
        d_alpha = (self.alpha2 - self.alpha1) / self.N
        delta_alpha = (self.alpha2 - self.alpha1)

        integral_left = lambda alpha: ((2 * (alpha - self.alpha1)) / (torch.exp(torch.lgamma(torch.tensor(2 - alpha))))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - alpha)
        integral_right = lambda alpha: ((2 * (alpha - self.alpha1)) / (torch.exp(torch.lgamma(torch.tensor(3 - alpha))))) * second_order_grad * torch.abs(p.data.detach() - pm_1.data.detach()) ** (2 - alpha) # we cannot ignore the abs because of negetive under square
        
        delta = delta_alpha * integral_left(self.alpha1) * 0.5
        for n in range(1, self.N):
            if (1 - (self.alpha1 + n * d_alpha)) > 0 :
                delta += delta_alpha * integral_left(self.alpha1 + n * d_alpha)
            else:
                delta += delta_alpha * integral_right(self.alpha1 + n * d_alpha)

        delta += delta_alpha * integral_right(self.alpha2) * 0.5

        return delta

# TODO: Make the for loop multiprocessed
# import torch.multiprocessing as mp

# def worker(self, n, alpha1, d_alpha, queue):
#     integral = lambda alpha: ((2 * (alpha - self.alpha1)) / (torch.exp(torch.lgamma(torch.tensor(2 - alpha))) * (self.alpha2 - self.alpha1) ** 2)) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - alpha)
#     result = integral(alpha1 + n * d_alpha)
#     queue.put(result)

# def parallel_loop(N, alpha1, d_alpha):
#     delta = 0
#     queue = mp.Queue()
#     processes = []

#     for n in range(1, N):
#         p = mp.Process(target=worker, args=(n, alpha1, d_alpha, queue))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     while not queue.empty():
#         delta += queue.get()

#     return delta