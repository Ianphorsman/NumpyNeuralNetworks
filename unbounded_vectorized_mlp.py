import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import _pickle as pickle
from pprint import pprint as pp

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
        for l in layers:
            self.nodes.append(np.ones(l))

        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i][0], layers[i+1][0]))

        self.deltas = []
        for i in range(len(layers)-1):
            self.deltas.append(np.random.randn(layers[i][0], layers[i+1][0]))



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
        self.nodes[0] = x
        for i in range(len(self.nodes)-1):
            self.nodes[i+1] = self.sigmoid(np.dot(self.nodes[i], self.weights[i]))

    def backpropagate(self, y):
        # calculate deltas
        self.deltas[-1] = self.d_sigmoid(self.nodes[-1]) * -(y - self.nodes[-1])
        print(self.deltas[-1])

        for i in reversed(range(len(self.deltas)-1)):
            self.deltas[i] = np.multiply(self.d_sigmoid())



        for i in reversed(range(len(self.deltas)-1)):
            self.deltas[i] = self.d_sigmoid(self.nodes[i-1]) * np.dot(self.deltas[i+1], self.weights[i+1].T)
            print(self.deltas[i])
        # update weights
        for i in reversed(range(len(self.weights))):
            change = np.multiply(self.deltas[i], self.nodes[i-1])
            #print(self.deltas[-i].shape, self.nodes[-i-1].shape)
            update = self.learning_rate * change + self.deltas[i]
            self.weights[i] -= update
            self.deltas[i] = change

        error = 0.5 * ((y - self.nodes[-1])**2)
        return error

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def d_sigmoid(self, X):
        return X * (1 - X)

    def inspect_shape(self):
        pass

    def inspect(self):
        print("Intended Layers: ")
        pp(self.layers)
        print()
        print()
        print("Generated Nodes: ")
        pp(self.nodes)
        print()
        print()
        print("Weights: ")
        pp(self.weights)
        print()
        print()
        print("Deltas: ")
        pp(self.deltas)

    def save(self):
        pickle.dump(self, open("{}.p".format(self.filename), 'wb'))

    def load(self):
        return pickle.load(open("{}.p".format(self.filename), 'rb'))

X, y = make_moons(1000, noise=0.2, random_state=314)
X = np.column_stack((X, np.ones(X.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=314)

nn = NeuralNetwork(X_train, y_train, filename="vectorized_neural_network")
nn.train()
nn.save()
mn = nn.load()
#nn.inspect()