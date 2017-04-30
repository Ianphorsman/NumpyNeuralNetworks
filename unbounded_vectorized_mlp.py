import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import _pickle as pickle

class NeuralNetwork(object):

    def __init__(
        self, X, y,
        filename='neural_network_2',
        inspect_rate=50,
        iterations=1,
        learning_rate=0.00002,
        layers=(
                (3, 1),
                (3, 1),
                (1, 1)
        )
    ):
        self.X = X
        self.y = y
        self.filename = filename
        self.inspect_rate = inspect_rate
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.layers = layers

        self.nodes = []


        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i][0], layers[i+1][0]))

        self.deltas = []




    def train(self):
        for i in range(self.iterations):
            error = 0.0
            for j in range(self.X.shape[0]):
                x = self.X[j]
                y = self.y[j]
                self.feedforward(x)
                error += self.backpropagate(y)
            if i % self.inspect_rate == 0:
                print(error, i)

    def test(self, x):
        self.feedforward(x)
        #print(self.output)

    def test_accuracy(self, x):
        correct = 0
        for i in range(X.shape[0]):
            self.feedforward(X[i])
            if np.round(self.output[0]) == y[i]:
                correct += 1

        return (correct / X.shape[0]) * 100

    def feedforward(self, x):
        pass

    def backpropagate(self, y):
        pass

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def d_sigmoid(self, X):
        return X * (1 - X)

    def inspect(self):
        pass

    def inspect_shape(self):
        return self.layers

X, y = make_moons(1000, noise=0.2, random_state=314)
X = np.column_stack((X, np.ones(X.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=314)

nn = NeuralNetwork(X_train, y_train, filename="vectorized_neural_network")

print(nn.inspect_shape())