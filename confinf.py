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

opt = Optimizer()

x_train, y_train, x_test, y_test= load_mnist()
pca = PCA(10)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


model = Net(x_train, y_train, [10, 10], batch_size=64)


print("Starting GD")
s=time.time()
epoch = 5

hist_int = []
acc_int = []
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.optimizer(model.categorical_cross_entropy, model.w_flatten, model,lr=0.03, max_iter=10)
        temp_loss = model.categorical_cross_entropy(model.w_flatten)
        hist_int.append(temp_loss)
        acc_int.append(model.eval(w=model.w_flatten, x=x_test, y=y_test))
        print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
model.save_model(name="model_GD")
print(f"time is: {d-s}")
valval=model.eval(w=model.w_flatten, x=x_test, y=y_test)
print(valval)
with open('acc.txt', 'a') as acctxt:
    acctxt.write(str(valval))

sns.lineplot(hist_int, label="int", linestyle='solid')
plt.savefig("Int_Loss.png", dpi=500)
plt.close()

with open('hist_int.pkl', 'wb') as f:
    pickle.dump(hist_int, f)

with open('acc_int.pkl', 'wb') as f:
    pickle.dump(acc_int, f)


print("Starting Fractional")
s=time.time()
epoch = 5
model.reset_to_default()
hist_frac = []
acc_frac = []
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha=0.9, max_iter=10)
        temp_loss = model.categorical_cross_entropy(model.w_flatten)
        hist_frac.append(temp_loss)
        acc_frac.append(model.eval(w=model.w_flatten, x=x_test, y=y_test))
        print(f"Fractional, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")

        model.batch_counter += model.batch_size

d=time.time()
model.save_model(name="model_Frac")
print(f"time is: {d-s}")
valval=model.eval(w=model.w_flatten, x=x_test, y=y_test)
print(valval)
with open('acc.txt', 'a') as acctxt:
    acctxt.write(str(valval))

sns.lineplot(hist_int, label="int", linestyle='solid')
sns.lineplot(hist_frac, label="frac", linestyle='dashed')
plt.savefig("Int_Frac.png", dpi=500)
plt.close()

with open('hist_frac.pkl', 'wb') as f:
    pickle.dump(hist_frac, f)
with open('acc_frac.pkl', 'wb') as f:
    pickle.dump(acc_frac, f)

print("Starting Multi Fractional")
s=time.time()
epoch = 5
model.reset_to_default()
hist_multi = []
acc_multi = []
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.multi_frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha1=0.9, alpha2=1.1, max_iter=10)
        temp_loss = model.categorical_cross_entropy(model.w_flatten)
        hist_multi.append(temp_loss)
        acc_multi.append(model.eval(w=model.w_flatten, x=x_test, y=y_test))
        print(f"Multi, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
model.save_model(name="model_Frac_Multi")
print(f"time is: {d-s}")
valval=model.eval(w=model.w_flatten, x=x_test, y=y_test)
print(valval)
with open('acc.txt', 'a') as acctxt:
    acctxt.write(str(valval))

sns.lineplot(hist_int, label="int", linestyle='solid')
sns.lineplot(hist_frac, label="frac", linestyle='dashed')
sns.lineplot(hist_multi, label="multi", linestyle='dotted')
plt.savefig("Int_Frac_Multi.png", dpi=500)
plt.close()

with open('hist_multi.pkl', 'wb') as f:
    pickle.dump(hist_multi, f)
with open('acc_multi.pkl', 'wb') as f:
    pickle.dump(acc_multi, f)

print("Starting Distribute Fractional")
s=time.time()
epoch = 5
model.reset_to_default()
hist_dist = []
acc_dist = []
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        opt.dist_frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha1=0.9, alpha2=1.1, max_iter=10, N=10)
        temp_loss = model.categorical_cross_entropy(model.w_flatten)
        hist_dist.append(temp_loss)
        acc_dist.append(model.eval(w=model.w_flatten, x=x_test, y=y_test))
        print(f"Distribute, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
        model.batch_counter += model.batch_size

d=time.time()
model.save_model(name="model_Frac_dist")
print(f"time is: {d-s}")
valval=model.eval(w=model.w_flatten, x=x_test, y=y_test)
print(valval)
with open('acc.txt', 'a') as acctxt:
    acctxt.write(str(valval))

with open('hist_dist.pkl', 'wb') as f:
    pickle.dump(hist_dist, f)
with open('acc_dist.pkl', 'wb') as f:
    pickle.dump(acc_dist, f)

sns.lineplot(data=hist_int, label="int", linestyle='solid')
sns.lineplot(data=hist_frac, label="frac", linestyle='dashed')
sns.lineplot(data=hist_multi, label="multi", linestyle='dotted')
sns.lineplot(data=hist_dist, label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Loss Value")
plt.savefig("Int_Frac_Multi_Dist.png", dpi=500)
plt.close()

sns.lineplot(data=hist_int[:50], label="int", linestyle='solid')
sns.lineplot(data=hist_frac[:50], label="frac", linestyle='dashed')
sns.lineplot(data=hist_multi[:50], label="multi", linestyle='dotted')
sns.lineplot(data=hist_dist[:50], label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Loss Value")
plt.savefig("Int_Frac_Multi_Dist_50.png", dpi=500)
plt.close()

sns.lineplot(data=hist_int[:100], label="int", linestyle='solid')
sns.lineplot(data=hist_frac[:100], label="frac", linestyle='dashed')
sns.lineplot(data=hist_multi[:100], label="multi", linestyle='dotted')
sns.lineplot(data=hist_dist[:100], label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Loss Value")
plt.savefig("Int_Frac_Multi_Dist_100.png", dpi=500)
plt.close()

##########

sns.lineplot(data=acc_int, label="int", linestyle='solid')
sns.lineplot(data=acc_frac, label="frac", linestyle='dashed')
sns.lineplot(data=acc_multi, label="multi", linestyle='dotted')
sns.lineplot(data=acc_dist, label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Accuracy")
plt.savefig("acc_Full.png", dpi=500)
plt.close()

sns.lineplot(data=acc_int[:50], label="int", linestyle='solid')
sns.lineplot(data=acc_frac[:50], label="frac", linestyle='dashed')
sns.lineplot(data=acc_multi[:50], label="multi", linestyle='dotted')
sns.lineplot(data=acc_dist[:50], label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Accuracy")
plt.savefig("acc_50.png", dpi=500)
plt.close()


sns.lineplot(data=acc_int[:100], label="int", linestyle='solid')
sns.lineplot(data=acc_frac[:100], label="frac", linestyle='dashed')
sns.lineplot(data=acc_multi[:100], label="multi", linestyle='dotted')
sns.lineplot(data=acc_dist[:100], label="dist", linestyle='dashdot').set(xlabel="Iteration", ylabel="Accuracy")
plt.savefig("acc_100.png", dpi=500)
plt.close()
