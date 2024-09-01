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
            # second_order_grad = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]
            # t2 = (1 / torch.exp(torch.lgamma(torch.tensor(3 - self.alpha2))))  * second_order_grad * torch.abs(p.data.detach() - pm_1.data.detach()) ** (2 - self.alpha2)
            t2 = (1 / (torch.exp(torch.lgamma(torch.tensor(3 - self.alpha2)))))  * second_order_grad * torch.abs(p.data.detach() - pm_1.data.detach()) ** (2 - self.alpha2)
        
        # if torch.any(torch.isnan(t1)) or torch.any(torch.isinf(t1)):
        #     t1 = torch.where(torch.isnan(t1), torch.tensor(epsilon), t1)
        #     t1 = torch.where(torch.isinf(t1), torch.tensor(epsilon), t1)

        # if torch.any(torch.isnan(t2)) or torch.any(torch.isinf(t2)):
        #     t2 = torch.where(torch.isnan(t2), torch.tensor(epsilon), t2)
        #     t2 = torch.where(torch.isinf(t2), torch.tensor(epsilon), t2)

        return 0.5 * t1 + 0.5 * t2


    def distributed_fractional(self, p, pm_1, second_order_grad):
        d_alpha = (self.alpha2 - self.alpha1) / self.N
        # second_order_grad = torch.autograd.grad(p.grad.sum(), p, create_graph=True)[0]

        # integral = lambda alpha: ((2 * (alpha - self.alpha1)) / (torch.exp(torch.lgamma(torch.tensor(2 - alpha))) * (self.alpha2 - self.alpha1) ** 2)) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - alpha)
        integral_left = lambda alpha: (1 / (torch.exp(torch.lgamma(torch.tensor(2 - alpha))))) * p.grad.detach() * torch.abs(p.data.detach() - pm_1.data.detach()) ** (1 - alpha)
        integral_right = lambda alpha: (1 / (torch.exp(torch.lgamma(torch.tensor(3 - alpha))))) * second_order_grad * torch.abs(p.data.detach() - pm_1.data.detach()) ** (2 - alpha) # we cannot ignore the abs because of negetive under square
        delta = integral_left(self.alpha1) * (1 / self.N)
        # delta = 0.5 * integral_left(self.alpha1)
        # epsilon = 0
        # if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
        #     delta = torch.where(torch.isnan(delta), torch.tensor(epsilon), delta)
        #     delta = torch.where(torch.isinf(delta), torch.tensor(epsilon), delta)


        # NS = [n for n in range(1, self.N)]
        # print(0, torch.mean(delta))
        for n in range(1, self.N):
            # delta += integral(self.alpha1 + n * d_alpha)
            if (1 - (self.alpha1 + n * d_alpha)) > 0 :
                BOBO = integral_left(self.alpha1 + n * d_alpha) * (1 / self.N)
                # print(n, torch.mean(BOBO))
            else:
                
                # if second_order_grad.sum == 0:
                #     second_order_grad = epsilon
                BOBO = integral_right(self.alpha1 + n * d_alpha) * (1 / self.N)
                # print(n, torch.mean(BOBO), second_order_grad.mean(), p.grad.mean())

            # if torch.any(torch.isnan(BOBO)) or torch.any(torch.isinf(BOBO)):
            #     BOBO = torch.where(torch.isnan(BOBO), torch.tensor(epsilon), BOBO)
            #     BOBO = torch.where(torch.isinf(BOBO), torch.tensor(epsilon), BOBO)

            delta = delta + BOBO
        # if second_order_grad.sum == 0:
        #     second_order_grad = epsilon
        # BOBO = 0.5 * integral_right(self.alpha2)
        BOBO = integral_right(self.alpha2) * (1 / self.N)
        # if torch.any(torch.isnan(BOBO)) or torch.any(torch.isinf(BOBO)):
        #     BOBO = torch.where(torch.isnan(BOBO), torch.tensor(epsilon), BOBO)
        #     BOBO = torch.where(torch.isinf(BOBO), torch.tensor(epsilon), BOBO)
        
        delta = delta + BOBO
        # print(self.N, torch.mean(delta))
        # delta = parallel_loop(self.N, self.alpha1, d_alpha)

        return delta

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
