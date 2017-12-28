import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import _pickle as picklerick
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
import pdb


class SGD(object):

    def __init__(self, X, y, filename='sgd', inspect_rate=50, iterations=1000, learning_rate=0.000025, nodes=(2,3,1), batch_size=50, activation_function='sigmoid'):
        # store input and expected output data
        self.X = X
        self.y = np.atleast_2d(y)

        # saved network name
        self.filename = filename

        # inspect rate to print and store current error and accuracy
        self.inspect_rate = inspect_rate

        # number of iterations
        self.iterations = iterations

        # learning hyperparameters
        self.learning_rate = learning_rate

        # activation function
        self.activation_function = activation_function

        # network layers
        self.nodes = nodes
        # store costs and accuracy over iterations
        self.costs = []
        self.accuracies = []

        # initialize placeholder layers
        self.layers = self.initialize_layers()

        # initialize placeholder weights and deltas
        self.weights = self.initialize_weights()
        #self.deltas = self.initialize_deltas()

    def initialize_layers(self):
        layers = []
        return layers

    def initialize_weights(self):
        weights = []
        for i in range(1, len(self.nodes)):
            weights.append(np.random.randn(self.nodes[i - 1] + 1, self.nodes[i]))
        return weights

    def train(self, X):
        for i in range(self.iterations):
            hypothesis = self.feedforward(X)
            self.backpropagate(hypothesis)
            cost = np.average(0.5 * ((self.y - hypothesis) ** 2))
            if i % self.inspect_rate == 0:
                self.costs.append(cost)
                accuracy = self.test_accuracy(X, self.y)
                self.accuracies.append(accuracy)
                print(self.inspect_performance(i, cost, accuracy))

    def test_accuracy(self, X_test, y_test):
        hypothesis = self.feedforward(X_test)
        #pdb.set_trace()
        return np.sum(np.round(hypothesis) == y_test.T) / X_test.shape[0] * 100


    def next_batch(self):
        pass

    def add_bias(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def feedforward(self, X):
        num_weight_matrices = len(self.weights)
        layer = self.add_bias(X)
        self.layers.append(layer)
        for i in range(num_weight_matrices):
            #pdb.set_trace()
            if i == num_weight_matrices - 1:
                w = self.weights[i]
                layer = self.activate(np.dot(layer, w))
            else:
                w = self.weights[i]
                layer = self.add_bias(self.activate(np.dot(layer, w)))
            self.layers.append(layer)
        return layer

    def backpropagate(self, hypothesis):
        num_layer_matrices = len(self.layers)
        deltas = [self.y.T - hypothesis]
        for layer, weights in zip(reversed(self.layers[:-1]), reversed(self.weights)):
            prev_delta = deltas[-1]
            #pdb.set_trace()
            if prev_delta.shape[1] - 1 != weights.shape[1]:
                #print("not clipped")
                delta = prev_delta.dot(weights.T) * self.activate(layer, d=True)
            else:
                #print("clipped")
                delta = prev_delta[:, 1:].dot(weights.T) * self.activate(layer, d=True)
            deltas.append(delta)
        for i in range(1, len(deltas)):
            delta = deltas[i - 1]
            layer = self.layers[:-1][-i]
            #print("update")
            #pdb.set_trace()
            if i == 1:
                self.weights[-i] += self.learning_rate * delta.T.dot(layer).T
            else:
                self.weights[-i] += self.learning_rate * delta[:, 1:].T.dot(layer).T

        return deltas




    def activate(self, X, d=False):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(X, d)
        elif self.activation_function == 'tanh':
            return self.tanh(X, d)
        else:
            return self.relu(X, d)

    def sigmoid(self, X, d=False):
        if d:
            return X * (1 - X)
        return 1 / (1 + np.exp(-X))

    def tanh(self, X, d=False):
        if d:
            return 1 - np.tanh(X) ** 2
        return np.tanh(X)

    def relu(self, X, d=False):
        if d:
            return np.multiply(1., (X > 0))
        return np.maximum(X, 0, X)

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

    def inspect_architecture(self):
        return [w.shape for w in self.weights] == [d.shape for d in self.deltas]


X, y = make_moons(1000, noise=0.15, random_state=333)
X_train, X_test, y_train, y_test = train_test_split(X, y)

sgd = SGD(X_train, y_train, iterations=2000, learning_rate=0.01)

sgd.train(sgd.X)
print(sgd.test_accuracy(X_test, np.atleast_2d(y_test)))
