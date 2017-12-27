import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
import _pickle as pickle
import pdb
class NeuralNetwork(object):

    def __init__(self, X, y, filename='neural_network', inspect_rate=50, iterations=1000, learning_rate=0.000025, input_nodes=3, hidden_nodes=4, output_nodes=1, optional_features=[]):
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
        self.activation_input = np.atleast_2d(np.ones(self.input_nodes))
        self.activation_hidden = np.atleast_2d(np.ones(self.hidden_nodes))
        self.activation_output = np.atleast_2d(np.ones(self.output_nodes))



        # initialize weights
        self.ih_weights = np.random.randn(self.activation_input.shape[1] + 1, self.activation_hidden.shape[1])
        self.ho_weights = np.random.randn(self.activation_hidden.shape[1] + 1, self.activation_output.shape[1])

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
        return self.activation_output

    def test_accuracy(self, X, y):
        correct = 0.0
        for i in range(X.shape[0]):
            self.feedforward(X[i])
            if np.round(self.activation_output[0]) == y[i]:
                correct += 1

        return (correct / X.shape[0]) * 100

    def feedforward(self, x):
        self.activation_input = np.append(1, x)
        self.activation_hidden = np.append(1, self.sigmoid(np.dot(self.activation_input, self.ih_weights)))
        self.activation_output = self.sigmoid(np.dot(self.activation_hidden, self.ho_weights))

    def vbackpropagate(self, y):
        output_deltas = y - self.activation_output
        #output_deltas = error * self.d_sigmoid(self.activation_output[1:])
        hidden_deltas = np.dot(output_deltas, self.ho_weights[1:].T) * self.d_sigmoid(self.activation_hidden[1:])

        self.ho_weights += self.learning_rate * np.dot(np.atleast_2d(self.activation_hidden).T, np.atleast_2d(output_deltas))
        self.ih_weights += self.learning_rate * np.dot(np.atleast_2d(self.activation_input).T, np.atleast_2d(hidden_deltas))

        error = 0.5 * ((y - self.activation_output) ** 2)
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

    def visualize_decision_boundary(self, x, y, c):
        plt.figure(0)
        xbg, ybg, cbg = [], [], []
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        for x_ in np.linspace(x_min - 0.1, x_max + 0.1, 300):
            for y_ in np.linspace(y_min - 0.1, y_max + 0.1, 200):
                xbg.append(x_)
                ybg.append(y_)
                cbg.append(np.round(self.test([x_, y_, x_ * y_])))
        plt.scatter(xbg, ybg, cbg, cmap=plt.get_cmap('Blues'))
        plt.scatter(x, y, c=c)
        plt.show()

    def save(self):
        pickle.dump(self, open("{}.p".format(self.filename), 'wb'))

    def load(self):
        return pickle.load(open("{}.p".format(self.filename), 'rb'))

#from make_recurrent_moons import Moons
#moons = Moons()
#X, y = moons.recurrent(n_moons=5, degradation=1.05)
X, y = make_moons(1000, noise=0.15, random_state=500)
X = np.column_stack((X, np.multiply(X[:, 0], X[:, 1])))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=314)

nn = NeuralNetwork(X_train, y_train, filename="vectorized_neural_network", learning_rate=0.005, iterations=2000)
nn.train()
nn.save()

nn.visualize_decision_boundary(X_test[:, 0], X_test[:, 1], y_test)

my_score = nn.test_accuracy(X_test, y_test)
print(my_score)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(alpha=0.0025, random_state=1)
mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(my_score, score*100)

if my_score > (score*100):
    print("You are awesome!")
