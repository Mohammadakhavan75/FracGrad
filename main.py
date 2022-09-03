from network import Net
from optimizers import Optimizer
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

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


x_train, y_train, x_test, y_test= load_mnist()
model = Net(x_train, y_train, [256, 10], batch_size=8)
opt = Optimizer()

hist_all = []
epoch = 5
for ep in range(epoch):
    print("EPOCH: ", ep)
    for b in range(int(x_test.shape[0]/model.batch_size)):
        print("Batch step: ", b)
        x, history_int = opt.optimizer(model.categorical_cross_entropy, model.w_flatten, lr=0.03, return_history=True)
        x, history_frac = opt.frac_optimizer(model.categorical_cross_entropy, model.w_flatten,  lr=0.03, alpha=0.9, return_history=True)
        x, history_multi_frac = opt.multi_frac_optimizer(model.categorical_cross_entropy, model.w_flatten,  lr=0.03, alpha1=1.1, alpha2=0.9, return_history=True)
        x, history_dist_frac = opt.dist_frac_optimizer(model.categorical_cross_entropy, model.w_flatten,  lr=0.03, alpha1=0.1, alpha2=1.1, return_history=True)
        
        plt.savefig("batch_" + str(b) + ".png", dpi=500)

        hist_all.append([history_int, history_frac, history_multi_frac, history_dist_frac])
        model.batch_counter += model.batch_size