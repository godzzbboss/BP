#-*- coding: utf-8 -*

from BP import BP_NN
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np


X, y = make_circles(100, noise=0.05, random_state=10)  # 2 input 1 output
print(len(X), X.shape[0])

f1 = plt.figure(1)
# print(X.shape[0])
plt.scatter(X[:,0], X[:,1], s=20, c=y)
plt.title("circles data")
plt.show()


'''
BP implementation
'''


nn = BP_NN.BP()  # build a BP network class
nn.creatNN(2, 6, 1, 'Sigmoid')  # build the network

e = []
# the num of interate is 1000
for i in range(1000):
    error = nn.train(X, y, learning_rate=0.1)
    e.append(error)

f2 = plt.figure(2)
plt.xlabel("epochs")
plt.ylabel("accumulated error")
plt.title("circles convergence curve")
plt.plot(e)
plt.show()

'''
draw decision boundary
'''
h = 0.01
x0_min, x0_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
x1_min, x1_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))

f3 = plt.figure(3)
# x0跟x1都是一个矩阵
z = nn.predict(np.c_[x0.ravel(), x1.ravel()])
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap = plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], s=20, c=y)
plt.title("circles classification")
plt.show()
