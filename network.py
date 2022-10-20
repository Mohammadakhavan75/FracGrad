from  scipy.integrate import quad
import numpy as np
import time 
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score

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

        self.W_F = self.w_flatten.copy()
        self.W_I = self.w.copy()

    def reset_to_default(self):
        self.w = self.W_I.copy()
        self.w_flatten = self.W_F.copy()
        self.step=0
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

    def update_batch_counter(self):
        self.batch_counter += self.batch_size

    def reset_batch_counter(self):
        self.batch_counter = 0

    def forward(self, w):
        H = self.x[self.batch_counter: self.batch_counter + self.batch_size] @ np.asarray(np.reshape(w[:self.w_shape[0][0] * self.w_shape[0][1]], self.w_shape[0]))
        for i in range(1, len(self.layers)):
            H = H @ np.asarray(np.reshape(w[self.w_shape[i-1][0] * self.w_shape[i-1][1]:(self.w_shape[i-1][0] * self.w_shape[i-1][1]) + (self.w_shape[i][0] * self.w_shape[i][1])], self.w_shape[i]))

        return H

    def forward_less(self):
        H = self.x[self.batch_counter: self.batch_counter + self.batch_size] @ np.asarray(np.reshape(self.w_flatten[:self.w_shape[0][0] * self.w_shape[0][1]], self.w_shape[0]))
        for i in range(1, len(self.layers)):
            H = H @ np.asarray(np.reshape(self.w_flatten[self.w_shape[i-1][0] * self.w_shape[i-1][1]:(self.w_shape[i-1][0] * self.w_shape[i-1][1]) + (self.w_shape[i][0] * self.w_shape[i][1])], self.w_shape[i]))

        return H

    def fit(self, w, optimizer):
        pass

    def prediction(self, w, x, b_c):
        H = x[b_c: b_c + self.batch_size] @ np.asarray(np.reshape(w[:self.w_shape[0][0] * self.w_shape[0][1]], self.w_shape[0]))
        for i in range(1, len(self.layers)):
            H = H @ np.asarray(np.reshape(w[self.w_shape[i-1][0] * self.w_shape[i-1][1]:(self.w_shape[i-1][0] * self.w_shape[i-1][1]) + (self.w_shape[i][0] * self.w_shape[i][1])], self.w_shape[i]))        
        
        return self.softmax(H)

    def eval(self, w, x, y):
        total_loss = []
        preds = []
        self.reset_batch_counter()
        for _ in range(int(x.shape[0]/self.batch_size)):
            predict = self.prediction(w, x, self.batch_counter)
            for p in predict:
                preds.append(np.argmax(p))
            
            loss = -(1/self.batch_size) * np.sum([target * np.log([predict[i][np.argmax(target)]if predict[i][np.argmax(target)] != 0 else 0.1 ** 14]) for i, target in enumerate(y[self.batch_counter: self.batch_counter + self.batch_size])])
            total_loss.append(loss)
            self.update_batch_counter()
        
        preds = tf.keras.utils.to_categorical(preds, 10)
        return np.mean(total_loss), accuracy_score(y_true=y[:len(preds)], y_pred=preds)


    def softmax(self, output):
        return (np.exp(output.T)/np.exp(output.T).sum(axis=0)).T
        # return np.asarray([np.asarray([np.exp(one)/ np.sum(np.exp(one_output)) for one in one_output]) for one_output in output])
    
    def MSE(self, w):
        return np.sum((self.y[self.batch_counter: self.batch_counter + self.batch_size] - self.softmax(self.forward())) ** 2)

    def categorical_cross_entropy(self, w):
        # we use this ([predict[i][np.argmax(target)]if predict[i][np.argmax(target)] != 0 else 10 ** 6]) for handling log(0) and preventing -Inf error.
        predict = self.softmax(self.forward(w))
        loss = -(1/self.batch_size) * np.sum([target * np.log([predict[i][np.argmax(target)]if predict[i][np.argmax(target)] != 0 else 0.1 ** 14]) for i, target in enumerate(self.y[self.batch_counter: self.batch_counter + self.batch_size])])
        return  loss

    def save_model(self, path='./', name="model"):
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(self.w_flatten, f)

        with open(path + name + '_shape.pkl', 'wb') as f:
            pickle.dump(self.w_shape, f)

    def load_model(self, path_model, path_shape):
        self.w_flatten = pickle.load(open(path_model,'rb'))
        self.w_shape = pickle.load(open(path_shape,'rb'))