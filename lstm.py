import numpy as np
import matplotlib.pyplot as plt

np.random.seed(314)

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

class LSTM(object):

    def __init__(self, X, y, iterations=10000, learning_rate=0.000025, alpha=0.1, input_nodes=2, hidden_nodes=16, output_nodes=1):
        self.X = X
        self.y = y

        # initialize hyperparameters
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate

        # initialize model architecture, specifies layer shapes
        self.input_nodes = np.int64(input_nodes)
        self.hidden_nodes = np.int64(hidden_nodes)
        self.output_nodes = np.int64(output_nodes)

        # initialize placeholder nodes
        self.activation_input = np.ones(self.input_nodes)
        self.activation_hidden = np.ones(self.hidden_nodes)
        self.activation_output = np.ones(self.output_nodes)

        # initialize weights
        self.synapse_0 = 2 * np.random.random((self.input_nodes, self.hidden_nodes)) - 1
        self.synapse_1 = 2 * np.random.random((self.hidden_nodes, self.output_nodes)) - 1
        self.synapse_h = 2 * np.random.random((self.hidden_nodes, self.hidden_nodes)) - 1

        # initialize update placeholders
        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)



        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def train(self):
        for j in range(self.iterations):
            # generate an addition problem a + b = c
            a_int = np.random.randint(largest_number / 2)
            a = int2bin[a_int]
            b_int = np.random.randint(largest_number / 2)
            b = int2bin[b_int]
            c_int = a_int + b_int
            c = int2bin[c_int]
            d = np.zeros_like(c)
            total_error = 0

            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(self.hidden_nodes))

            for position in binary_dim:
                X = np.array([
                    [
                        a[binary_dim - position - 1],
                        b[binary_dim - position - 1]
                    ]
                ])

                y = np.array([
                    
                ])

            



    def test(self):
        pass

    def feedforward(self):
        pass

    def backpropagate(self):
        pass

    def inspect(self):
        pass

    def visualize(self):
        pass




# generate training dataset

int2bin = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1
)

for i in range(largest_number):
    int2bin[i] = binary[i]

print(binary)