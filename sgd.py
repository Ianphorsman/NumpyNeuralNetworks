import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import _pickle as picklerick
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
import pdb


class SGD(object):

    def __init__(self, X, y, filename='sgd', inspect_rate=50, iterations=1000, learning_rate=0.000025, nodes=(2,3,1), batch_size=50):
        # store input and expected output data
        self.X = X
        self.y = y

        # saved network name
        self.filename = filename

        # inspect rate to print and store current error and accuracy
        self.inspect_rate = inspect_rate

        # number of iterations
        self.iterations = iterations

        # learning hyperparameters
        self.learning_rate = learning_rate

        # network layers
        self.nodes = nodes

        # store costs and accuracy over iterations
        self.costs = []
        self.accuracies = []

        # initialize placeholder nodes
        self.initialize_nodes()

        # initialize placeholder weights
        self.initialize_weights()


    def initialize_nodes(self):
        for i in range(len(self.nodes)):
            setattr(self, "layer_{}".format(i), np.atleast_2d(np.ones(self.nodes[i])))

    def initialize_weights(self):
        for i in range(len(self.nodes)-1):
            pass

    def train(self, X):
        for i in range(self.iterations):
            cost = 0
            if i % self.inspect_rate == 0:
                self.costs.append(cost)
                accuracy = self.test_accuracy()
                self.accuracies.append(accuracy)
                print(self.inspect_performance(i, cost, accuracy))

    def test_accuracy(self):
        pass

    def next_batch(self):
        pass

    def feedforward(self, X):
        pass

    def backpropagate(self, y):
        pass

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def d_sigmoid(self, X):
        return X * (1 - X)

    def tanh(self, X):
        return np.tanh(X)

    def d_tanh(self, X):
        return 1 - np.tanh(X) ** 2

    def save(self):
        picklerick.dump(self, open("{}.p".format(self.filename), 'wb'))

    def load(self):
        return picklerick.load(open("{}.p".format(self.filename), 'rb'))

    def plot_decision_boundary(self):
        pass

    def plot_performance(self, cost=True, accuracy=True):
        pass

    def inspect_performance(self,iteration, cost, accuracy):
        return "Iteration: {} , Cost: {} , Accuracy: {}".format(iteration, cost, accuracy)


X, y = make_moons(1000, noise=0.2, random_state=333)
X_train, X_test, y_train, y_test = train_test_split(X, y)

sgd = SGD(X_train, y_train)
