import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class LSTMCell(object):

    def __init__(self, X, y, filename='lstm_cell', inspect_rate=50, iterations=1000, learning_rate=0.000025):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def feedforward(self, x):
        pass

    def backpropagate(self, y):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x)**2

    def save(self):
        pass

    def load(self):
        pass