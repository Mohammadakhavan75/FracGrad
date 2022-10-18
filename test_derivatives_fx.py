from network import Net
from optimizers import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import time
from derivatives import local

def logs(hist, name, hist_total, Labels):

    for i, dd in enumerate(hist_total):
        sns.lineplot(data=dd, label=Labels[i])

    plt.savefig(f"Loss_{len(hist_total)}.png", dpi=500)
    plt.close()

    with open(f'hist_{Labels[len(hist_total) - 1]}.pkl', 'wb') as f:
        pickle.dump(hist, f)


X = np.ones((200, 10)) + np.random.random((200, 10))

oper = local()
opt = Optimizer()
hist_total=[]
Labels = ['GD', 'chen', 'conformable', 'katugampola', 'deformable', 'beta', 'AGO', 'Generalized']

def fun(x):
    return np.sum(x ** 2)

F=fun

################################
############## GD ##############
################################
s=time.time()
epoch = 5
name = 'GD'
hist = []

print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    local.optimizer(F, X)
    temp_loss = F(X)
    hist.append(temp_loss)
    print(f"GD, EPOCH: {ep}, Loss: {temp_loss}")

d=time.time()
print(f"time is: {d-s}")
logs(hist, name, hist_total, Labels)

hist_total.append(hist.copy())

################################
############# chen #############
################################

################################
######### conformable ##########
################################


################################
######### katugampola ##########
################################


################################
######### deformable ##########
################################

################################
######### beta ##########
################################


################################
######### AGO ##########
################################


################################
######### Generalized ##########
################################
