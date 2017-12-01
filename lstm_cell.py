import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class LSTMCell(object):

    def __init__(self, xs, ys, recur, learning_rate):
        self.recur = recur
        self.learning_rate = learning_rate

        self.x = np.zeros(xs+ys)
        self.xs = xs+ys
        self.y = np.zeros(ys)

        self.cs = np.zeros(ys)

        # initialize placeholder gates
        self.forget_gate = np.random.randn(ys, self.xs)
        self.input_gate = np.random.randn(ys, self.xs)
        self.output_gate = np.random.randn(ys, self.xs)
        self.cell_state = np.random.randn(ys, self.xs)

        # initialize placeholder gradients
        self.forget_gradient = np.zeros_like(self.forget_gate)
        self.input_gradient = np.zeros_like(self.input_gate)
        self.output_gradient = np.zeros_like(self.output_gate)
        self.cell_state_gradient = np.zeros_like(self.cell_state)

        # initialize placeholder cell state
        self.cell_state = []


    def train(self):
        pass

    def test(self):
        pass

    def feedforward(self):
        f = self.sigmoid(np.dot(self.forget_gate, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.input_gate, self.x))
        c = self.tanh(np.dot(self.cell_state, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.output_gate, self.x))
        self.y = o * self.tanh(self.cs)
        return self.cs, self.y, f, i, c, o

    def backpropagate(self, e, pcs, f, i, c, o, dfcs, dfhs):
        e = np.clip(e + dfhs, -6, 6)
        do = self.tanh(self.cs) * e
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        dc = dcs * i
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        di = dcs * c
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        df = dcs * pcs
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        dpcs = dcs * f
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df,self.f)[:self.ys]
        # return update gradients for forget, input, cell, output, cell state, hidden state
        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        # update forget, input, cell, output, cell state, hidden state
        self.forget_gradient = 0.9 * self.forget_gradient + 0.1 * fu**2
        self.input_gradient = 0.9 * self.input_gradient + 0.1 * iu**2
        self.cell_state_gradient = 0.9 * self.cell_state_gradient + 0.1 * cu**2
        self.output_gradient = 0.9 * self.output_gradient + 0.1 * ou**2

        self.forget_gate -= self.learning_rate / np.sqrt(self.forget_gradient + 1e-8) * fu
        self.input_gate -= self.learning_rate / np.sqrt(self.input_gradient + 1e-8) * iu
        self.cell_state -= self.learning_rate / np.sqrt(self.cell_state_gradient + 1e-8) * cu
        self.output_gate -= self.learning_rate / np.sqrt(self.output_gradient + 1e-8) * ou

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x)**2

    def save(self):
        pass

    def load(self):
        pass

    def pp(self):
        print("Loaded")