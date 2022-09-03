from  scipy.integrate import quad
import differint.differint as df
import itertools
import copy
import numpy as np
import time 

class Net:
    def __init__(self, x, y, layers, batch_size=32):
        self.x = x
        self.y = y
        self.layers = layers
        self.w = self.init_weight(self.x, self.layers)
        self.w_shape = np.asarray([x.shape for x in self.w])
        self.w_flatten = self.flattener(self.w, self.w_shape)
        self.step=0
        self.batch_size = batch_size
        self.batch_counter = 0

    def init_weight(self, x, layers):
        w = []
        tmp = np.random.normal(0, 1, (x.shape[1], layers[0]))
        w.append(tmp / np.linalg.norm(tmp))
        for i in range(1, len(layers)):
            tmp = np.random.normal(0, 1, (layers[i-1], layers[i]))
            w.append(tmp / np.linalg.norm(tmp))

        return np.asarray(w)

    def flattener(self, w, w_shape):
        temp = [[xx for x in w[i] for xx in x] for i in range(len(w_shape))]
        flat_w = [xx for x in temp for xx in x]
        return np.asarray(flat_w)

    def forward(self, w):
        # print("forward calculate")
        H = self.x[self.batch_counter: self.batch_counter + self.batch_size] @ np.asarray(np.reshape(w[:self.w_shape[0][0] * self.w_shape[0][1]], self.w_shape[0]))
        for i in range(1, len(self.layers)):
            H = H @ np.asarray(np.reshape(w[self.w_shape[i-1][0] * self.w_shape[i-1][1]:(self.w_shape[i-1][0] * self.w_shape[i-1][1]) + (self.w_shape[i][0] * self.w_shape[i][1])], self.w_shape[i]))

        return H

    def forward_less(self):
        H = self.x[self.batch_counter: self.batch_counter + self.batch_size] @ np.asarray(np.reshape(self.w_flatten[:self.w_shape[0][0] * self.w_shape[0][1]], self.w_shape[0]))
        for i in range(1, len(self.layers)):
            H = H @ np.asarray(np.reshape(self.w_flatten[self.w_shape[i-1][0] * self.w_shape[i-1][1]:(self.w_shape[i-1][0] * self.w_shape[i-1][1]) + (self.w_shape[i][0] * self.w_shape[i][1])], self.w_shape[i]))

        return H

    def softmax(self, output):
        # print("softmax calculate")
        return (np.exp(output.T)/np.exp(output.T).sum(axis=0)).T
        # return np.asarray([np.asarray([np.exp(one)/ np.sum(np.exp(one_output)) for one in one_output]) for one_output in output])
    
    def MSE(self, w):
        return np.sum((self.y[self.batch_counter: self.batch_counter + self.batch_size] - self.softmax(self.forward())) ** 2)

    def categorical_cross_entropy(self, w):
        # we use this ([predict[i][np.argmax(target)]if predict[i][np.argmax(target)] != 0 else 10 ** 6]) for handling log(0) and preventing -Inf error.
        # print("loss calculate")
        predict = self.softmax(self.forward(w))
        # predict = self.softmax(self.forward_less())
        loss = -(1/self.batch_size) * np.sum([target * np.log([predict[i][np.argmax(target)]if predict[i][np.argmax(target)] != 0 else 0.1 ** 14]) for i, target in enumerate(self.y[self.batch_counter: self.batch_counter + self.batch_size])])
        # self.batch_counter += self.batch_size
        # print("batch_couter: ", self.batch_counter)
        return  loss
