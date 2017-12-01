import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from lstm_cell import LSTMCell

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

        # initialize placeholder nodes
        self.activation_input = np.atleast_2d(np.ones(self.input_nodes))
        self.activation_hidden = np.apply_along_axis(lambda x: LSTMCell(5,5,5,0.000025), 0, np.atleast_2d(np.ones(self.hidden_nodes)))
        self.activation_output = np.atleast_2d(np.ones(self.output_nodes))

        # initialize weights
        self.ih_weights = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.ho_weights = np.random.randn(self.hidden_nodes, self.output_nodes)

        # initialize placeholder deltas
        self.ih_deltas = np.zeros_like(self.ih_weights)
        self.ho_deltas = np.zeros_like(self.ho_weights)

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
        print(self.activation_output)

    def test_accuracy(self, X, y):
        correct = 0.0
        for i in range(X.shape[0]):
            self.feedforward(X[i])
            if np.round(self.activation_output[0]) == y[i]:
                correct += 1

        return (correct / X.shape[0]) * 100

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

    def inspect_shape(self):
        pass


#def gen_test_data(self, range=250, magnitude=8):
 #   y = np.cos(np.arange(range)*(magnitude*np.pi/range))[:,None]
  #  x = np.arange(range)
   # return x, y


#X, y = gen_test_data()
#rnn = RNN(X=X, y=y, filename='lstm')