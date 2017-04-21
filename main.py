import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import _pickle as pickle

class NeuralNetwork(object):

    def __init__(self, X, y, iterations=50000, learning_rate=0.000001, input_nodes=3, hidden_nodes=3, output_nodes=1):
        self.X = X
        self.y = y
        self.iterations = iterations
        self.learning_rate = learning_rate

        self.input_nodes = np.int64(input_nodes)
        self.hidden_nodes = np.int64(hidden_nodes)
        self.output_nodes = np.int64(output_nodes)

        self.activation_input = np.ones(self.input_nodes)
        self.activation_hidden = np.ones(self.hidden_nodes)
        self.activation_output = np.ones(self.output_nodes)

        self.input_weights = np.random.randn(self.activation_input.shape[0], self.activation_hidden.shape[0])
        self.hidden_weights = np.random.randn(self.activation_hidden.shape[0], self.activation_output.shape[0])

        self.input_changes = np.random.randn(self.activation_input.shape[0], self.activation_hidden.shape[0])
        self.hidden_changes = np.random.randn(self.activation_hidden.shape[0], self.activation_output.shape[0])



    def train(self):
        for i in range(self.iterations):
            error = 0.0
            for j in range(self.X.shape[0]):
                x = self.X[j]
                y = self.y[j]
                self.feedforward(x)
                error += self.backpropagate(y)
            if i % 50 == 0:
                print(error)

    def test(self, x):
        self.feedforward(x)
        print(self.activation_output)

    def test_accuracy(self, X, y):
        correct = 0.0
        incorrect = 0.0
        for i in range(X.shape[0]):
            self.feedforward(X[i])
            if np.round(self.activation_output[0]) == y[i]:
                correct += 1
            else:
                incorrect += 1

        return (correct / X.shape[0]) * 100

    def visualize(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def feedforward(self, x): # i = input instance
        self.activation_input = x
        for i in range(self.hidden_nodes): # iterate over hidden layer
            s = 0.0
            for j in range(self.input_nodes): # iterate over input node and weight pairs
                s += self.activation_input[j] * self.input_weights[j][i]
            self.activation_hidden[i] = self.sigmoid(s)
        for i in range(self.output_nodes): # iterate over output layer
            s = 0.0
            for j in range(self.hidden_nodes): # iterate over hidden node and weight pairs
                s += self.activation_hidden[j] * self.hidden_weights[j][i]
            self.activation_output[i] = self.sigmoid(s)

    def backpropagate(self, y):
        output_deltas = self.d_sigmoid(self.activation_output) * -(y - self.activation_output)

        hidden_deltas = np.zeros(self.hidden_nodes)
        for i in range(self.hidden_nodes):
            error = 0.0
            for j in range(self.output_nodes):
                error += output_deltas[j] * self.hidden_weights[i][j]
            hidden_deltas[i] = self.d_sigmoid(self.activation_hidden[i]) * error

        for i in range(self.hidden_nodes):
            for j in range(self.output_nodes):
                change = output_deltas[j] * self.activation_hidden[i]
                self.hidden_weights[i][j] -= self.learning_rate * change + self.hidden_changes[i][j]
                self.hidden_changes[i][j] = change
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                change = hidden_deltas[j] * self.activation_input[i]
                self.input_weights[i][j] -= self.learning_rate * change + self.input_changes[i][j]
                self.input_changes[i][j] = change

        error = 0.5 * ((y - self.activation_output)**2)
        return error

    def inspect(self):
        print("Training Iterations to Perform: ", self.iterations)
        print("Learning Rate: ", self.learning_rate)
        print("Neural Network Structure: ", np.array([self.input_nodes, self.hidden_nodes, self.output_nodes]))
        print()
        print("Input Nodes: ", self.activation_input)
        print("Input Weights: ", self.input_weights)
        print()
        print("Hidden Layer 1: ", self.activation_hidden)
        print("Hidden Weights: ", self.hidden_weights)
        print()
        print("Output Nodes: ", self.activation_output)

    def inspect_shape(self):
        print("Input Nodes: ", self.activation_input.shape)
        print("Hidden Layer 1: ", self.activation_hidden.shape)
        print("Output Nodes: ", self.activation_output.shape)

        print("Input to Hidden 1 weights: ", self.input_weights.shape)
        print("Hidden 1 to Output weights: ", self.hidden_weights.shape)

    def save(self, filename='neural_network'):
        pickle.dump(self, open("{}.p".format(filename), 'wb'))

    def load(self, filename='neural_network'):
        return pickle.load(open("{}.p".format(filename), 'rb'))


# create data with outputs
X, y = make_moons(400, noise=0.2, random_state=314159)
X = np.column_stack((X, np.ones(X.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=314)

# plot data
#plt.figure()
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Blues)
#plt.show()

nn = NeuralNetwork(X=X_train, y=y_train)
nn.train()
nn.save()

mn = nn.load()

print(mn.test_accuracy(X_test, y_test))


