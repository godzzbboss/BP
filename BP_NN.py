import math
import random
import numpy as np


# generate a rand num between a and b
def rand(a, b):
    return (b - a) * random.random() + a


# define the activate function
def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# y is a sigmoid function
def SigmoidDerivate(y):
    return y * (1 - y)


def Tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def TanhDerivate(y):
    return 1 - y * y

'''
    BP Network

'''
class BP:
    def __init__(self):
        # node number each layer
        self.i_n = 0
        self.h_n = 0
        self.o_n = 0
        # output values for each layer
        self.i_v = []
        self.h_v = []
        self.o_v = []
        # parameters, ih_w is the parameters between input layer and hidden layer
        self.ih_w = []
        self.ho_w = []
        # threshold for each neuron
        self.h_t = []
        self.o_t = []

        # activate function and its derivation
        self.fun = {
            "Sigmoid": Sigmoid,
            "SigmoidDerivate": SigmoidDerivate,
            "Tanh": Tanh,
            "TanhDerivate": TanhDerivate,
            # for more, add here

        }

    # actfun is a string
    def creatNN(self, ni, nh, no, actfun):
        self.i_n = ni
        self.h_n = nh
        self.o_n = no

        # initial values of output for each layer
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(self.o_n)

        # initial weights for each link randomly
        self.ih_w = np.zeros([self.i_n, self.h_n])
        self.ho_w = np.zeros([self.h_n, self.o_n])
        for i in range(self.i_n):
            for h in range(self.h_n):
                self.ih_w[i][h] = rand(0, 1)
        for h in range(self.h_n):
            for j in range(self.o_n):
                self.ho_w[h][j] = rand(0, 1)

        # initial threshold for each neuron
        self.h_t = np.zeros(self.h_n)
        self.o_t = np.zeros(self.o_n)
        for h in range(self.h_n):
            self.h_t[h] = rand(0,1)
        for j in range(self.o_n):
            self.o_t[j] = rand(0, 1)

        # initial activative function
        self.af = self.fun[actfun]
        self.afd = self.fun[actfun + "Derivate"]

    # forward propagate
    def forwardPropagate(self, x):
        # activate input layer
        for i in range(self.i_n):
            self.i_v[i] = x[i]

        # activate hidden layer
        for h in range(self.h_n):
            # the input values of each node in hidden layer
            total = 0.0
            for i in range(self.i_n):
                total += self.ih_w[i][h] * self.i_v[i]
            self.h_v[h] = self.af(total + self.h_t[h])

        # activate output layer
        for j in range(self.o_n):
            # the input values of each node in output layer
            total = 0.0
            for h in range(self.h_n):
                total += self.ho_w[h][j] * self.h_v[h]

            self.o_v[j] = self.af(total + self.o_t[j])

    # backforward propagate, forwardPropagate before backforward
    def backPropagate(self, x, y, learning_rate):
        # get the output of the network
        self.forwardPropagate(x)
        # caculate the gradient based on output
        o_grid = np.zeros(self.o_n)
        for j in range(self.o_n):
            # J=1/2(y[j]-o_v[j])^2
            # self.afd(self.o_v[j]) is the derivation of self.o_v[j] with respect to w^T*h_v
            o_grid[j] = -(y[j] - self.o_v[j]) * self.afd(self.o_v[j])

        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * self.afd(self.h_v[h])

        # updating parameters
        for h in range(self.h_n):
            for j in range(self.o_n):
                #
                self.ho_w[h][j] -= learning_rate * o_grid[j] * self.h_v[h]

        for i in range(self.i_n):
            for h in range(self.h_n):
                self.ih_w[i][h] -= learning_rate * h_grid[h] * self.i_v[i]

        # updating the threshold
        for j in range(self.o_n):
            self.o_t[j] -= learning_rate * o_grid[j]
        for h in range(self.h_n):
            self.h_t[h] -= learning_rate * h_grid[h]
        

    '''
        train the nn and return the error
    
    '''
    def train(self, train_X, train_y, learning_rate):
        # the cost list of all sample
        e = []
        # SGD
        for k in range(train_X.shape[0]):
            x = np.array(train_X[k])
            y = np.array([train_y[k]])
            # print(x,y)
            self.backPropagate(x, y, learning_rate)

            # the cost of each sample
            e_iter = 0.0
            for j in range(self.o_n):
                e_iter += (self.o_v[j] - y[j])**2
            # the error of each sample
            e.append(e_iter/2)
        error = sum(e)/len(e)
        # the cost
        return error


    def predict(self, test_X):
        pred_y = []
        for n in range(test_X.shape[0]):
            self.forwardPropagate(test_X[n])

            if self.o_v[0] > 0.5:
                pred_y.append(1)
            else:
                pred_y.append(0)
        return np.array(pred_y)
