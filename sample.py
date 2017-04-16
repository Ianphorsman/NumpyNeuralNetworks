import numpy as np
from sklearn.datasets import make_moons


class NeuralNetwork(object):

    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)





X, y = make_moons(200, noise=0.2, random_state=32)

w0 = 2 * np.random.random((2, 3)) - 1
w1 = 2 * np.random.random((3, 1)) - 1

print(y[0:5])

print(w0.shape, w1.shape)

nn = NeuralNetwork()