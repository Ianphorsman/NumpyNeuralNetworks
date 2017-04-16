import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import _pickle as pickle

# This network is similar to a typical fully connected multilayer neural network,
# but evaluates error based on modulus operations instead of arithmetic operations

# The purpose of this design is to examine its effectiveness at generalizing its
# predictions given data with recurring patterns

class NeuralNetwork(object):

    def __init__(self):
        pass


    def train(self):
        pass

    def test(self):
        pass

    def sigmoid(self):
        pass

    def d_sigmoid(self):
        pass

    def visualize(self):
        pass

    def inspect(self):
        pass

    def inspect_shape(self):
        pass

    def save(self):
        pickle.dump(self, open('modulo_net.p', 'wb'))

    def load(self):
        return pickle.load(open('modulo_net.p', 'rb'))


