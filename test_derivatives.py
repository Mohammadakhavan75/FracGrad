from tkinter import Label
from network import Net
from optimizers import Optimizer
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pickle 
import time
from derivatives import local

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, [-1, 28*28])
    x_test = np.reshape(x_test, [-1, 28*28])
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    x_train = x_train/np.max(x_train)
    x_test = x_test/np.max(x_test)
    return x_train, y_train, x_test, y_test

def run_tf_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("\nFitting Model...")
    model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
    print("\nEvaluating Model...")
    model.evaluate(x_test, y_test)


def logs(model, x_test, y_test, hist, name, hist_total, Labels):
    model.save_model(name=name)
    print(f"ACC model: {name} is {model.eval(w=model.w_flatten, x=x_test, y=y_test)}")

    for i, dd in enumerate(hist_total):
        sns.lineplot(data=dd, label=Labels[i])

    plt.savefig(f"Loss_{len(hist_total)}.png", dpi=500)
    plt.close()

    with open(f'hist_{Labels[len(hist_total) - 1]}.pkl', 'wb') as f:
        pickle.dump(hist, f)


x_train, y_train, x_test, y_test= load_mnist()
pca = PCA(10)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
opt=Optimizer()
hist_total=[]
Labels = ['GD', 'chen', 'conformable', 'katugampola', 'deformable', 'beta', 'AGO', 'Generalized']

################################
############## GD ##############
################################
s=time.time()
epoch = 5
name = 'GD'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten

print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.optimizer(F, X, model)
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())


################################
############# chen #############
################################
s=time.time()
epoch = 5
name = 'chen'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.chen(F, X))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())


################################
######### conformable ##########
################################
s=time.time()
epoch = 5
name = 'conformable'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.conformable(F, X))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())



################################
######### katugampola ##########
################################
s=time.time()
epoch = 5
name = 'katugampola'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.katugampola(F, X))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())


################################
######### deformable ##########
################################
s=time.time()
epoch = 5
name = 'deformable'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.deformable(F, X))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())


################################
######### beta ##########
################################
s=time.time()
epoch = 5
name = 'beta'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.beta(F, X))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())


################################
######### AGO ##########
################################
s=time.time()
epoch = 5
name = 'AGO'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.AGO(F, X, local.K))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())



################################
######### Generalized ##########
################################
s=time.time()
epoch = 5
name = 'Generalized'
model = Net(x_train, y_train, [10, 10], batch_size=64)
hist = []
F = model.categorical_cross_entropy
X = model.w_flatten


print(f"Starting {name}")
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.gen_frac_opt(F, X, model, local.Generalized(F, X, local.K, local.Kprim))
        temp_loss = F(X)
        hist.append(temp_loss)
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
print(f"time is: {d-s}")
logs(model, x_test, y_test, hist, name, hist_total, Labels)

hist_total.append(hist.copy())
