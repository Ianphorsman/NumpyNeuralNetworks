import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class RNN(object):

    def __init__(self, X, y, filename='lstm_cell', inspect_rate=50, iterations=1000, learning_rate=0.000025, input_nodes=3, hidden_nodes=3, output_nodes=1):
        self.X = X
        self.y = y
        self.filename = filename
        self.inspect_rate = inspect_rate
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes



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
        x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x)**2

    def save(self):
        pickle.dump(self, open("{}.p".format(self.filename), 'wb'))

    def load(self):
        return pickle.load(open("{}.p".format(self.filename), 'rb'))