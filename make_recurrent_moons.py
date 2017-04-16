import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# this file provides functions to create datasets to train
#
#   - fully connected multilayer neural networks
#   - recurrent neural networks

class Moons(object):

    def __init__(self, datapoints=100, noise=0.25, seed=31415):
        self.datapoints = datapoints
        self.noise = noise
        self.seed = seed


    def linear(self):
        self.X, self.y = make_moons(self.datapoints, noise=self.noise, random_state=self.seed)
        self.X = np.column_stack((self.X, np.ones(self.X.shape[0])))

    def recurrent(self, n_moons=2, degradation=1):
        self.X, self.y = make_moons(self.datapoints, noise=self.noise, random_state=self.seed)

        for i in range(n_moons-1):
            X, y = make_moons(self.datapoints, noise=self.noise, random_state=self.seed + 5*(i+1))
            X = X + 10*(i+1) + (np.random.randn(X.shape[0], X.shape[1])*degradation)
            self.X = np.concatenate((self.X, X))
            self.y = np.concatenate((self.y, y))

        # add a bias vector... for now
        self.X = np.column_stack((self.X, np.ones(self.X.shape[0])))

    def visualize(self):
        plt.figure()
        plt.scatter(self.X[:,0], self.X[:,1], s=40, c=self.y, cmap=plt.cm.Blues)
        plt.show()


moons = Moons(datapoints=40, noise=0.1)
moons.recurrent(n_moons=6, degradation=1.1)
moons.visualize()