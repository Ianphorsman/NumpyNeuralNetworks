import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import _pickle as pickle

class NeuralNetwork(object):

    def __init__(self, X, y, filename='neural_network', inspect_rate=50, iterations=10000, learning_rate=0.000025, input_nodes=3, hidden_nodes=3, output_nodes=1):
        # initialize training data
        self.X = X
        self.y = y
        self.filename = filename
        self.inspect_rate = inspect_rate
        # initialize hyperparameters
        self.iterations = iterations
        self.learning_rate = learning_rate

        # declare model architecture (layer shapes)
        self.input_nodes = np.int64(input_nodes)
        self.hidden_nodes = np.int64(hidden_nodes)
        self.output_nodes = np.int64(output_nodes)

        # initialize placeholder nodes
        self.activation_input = np.ones(self.input_nodes)
        self.activation_hidden = np.ones(self.hidden_nodes)
        self.activation_output = np.ones(self.output_nodes)

        # initialize weights
        self.ih_weights = np.random.randn(self.activation_input.shape[0], self.activation_hidden.shape[0])
        self.ho_weights = np.random.randn(self.activation_hidden.shape[0], self.activation_output.shape[0])

        # initialize deltas
        self.ih_deltas = np.zeros((self.activation_input.shape[0], self.activation_hidden.shape[0]))
        self.ho_deltas = np.zeros((self.activation_hidden.shape[0], self.activation_output.shape[0]))

    def train(self):
        for i in range(self.iterations):
            error = 0.0
            for j in range(self.X.shape[0]):
                x = self.X[j]
                y = self.y[j]
                self.feedforward(x)
                error += self.vbackpropagate(y)
            if i % self.inspect_rate == 0:
                print(error, i)


    def test(self, x):
        self.feedforward(x)
        print(self.activation_output)

    def test_accuracy(self, X, y):
        correct = 0.0
        for i in range(X.shape[0]):
            self.feedforward(X[i])
            if np.round(self.activation_output[0]) == y[i]:
                correct += 1

        return (correct / X.shape[0]) * 100

    def feedforward(self, x):
        self.activation_input = x
        self.activation_hidden = self.sigmoid(np.dot(self.activation_input, self.ih_weights))
        self.activation_output = self.sigmoid(np.dot(self.activation_hidden, self.ho_weights))

    def vbackpropagate(self, y):
        output_deltas = self.d_sigmoid(self.activation_output) * -(y - self.activation_output)
        hidden_deltas = self.d_sigmoid(self.activation_hidden) * np.dot(output_deltas, self.ho_weights.T)

        change = np.multiply(output_deltas, self.activation_hidden).reshape(3,1)
        update = self.learning_rate * change + self.ho_deltas
        self.ho_weights -= update
        self.ho_deltas = change

        change = np.multiply(hidden_deltas, self.activation_input)
        update = self.learning_rate * change + self.ih_deltas.T
        self.ih_weights -= update.T
        self.ih_deltas = self.activation_input.reshape(3,1)*hidden_deltas

        error = 0.5 * ((y - self.activation_output)**2)
        return error

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def d_sigmoid(self, X):
        return X * (1 - X)

    def inspect(self):
        print("Training Iterations to Perform: ", self.iterations)
        print("Learning Rate: ", self.learning_rate)
        print("Neural Network Structure: ", np.array([self.input_nodes, self.hidden_nodes, self.output_nodes]))
        print()
        print("Input Nodes: ", self.activation_input)
        print("Input Weights: ", self.ih_weights)
        print()
        print("Hidden Layer 1: ", self.activation_hidden)
        print("Hidden Weights: ", self.ho_weights)
        print()
        print("Output Nodes: ", self.activation_output)

    def inspect_shape(self):
        print("Input Nodes: ", self.activation_input.shape)
        print("Hidden Layer 1: ", self.activation_hidden.shape)
        print("Output Nodes: ", self.activation_output.shape)

        print("Input to Hidden 1 weights: ", self.ih_weights.shape)
        print("Hidden 1 to Output weights: ", self.ho_weights.shape)

    def visualize(self):
        pass

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

print(mn.test_accuracy(X_test, y_test))
