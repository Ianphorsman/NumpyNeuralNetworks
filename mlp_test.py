import numpy as np
import _pickle as picklerick
import matplotlib.pyplot as plt

from vec_nlp import NeuralNetwork

def feed(neural_net, x):
    neural_net.feedforward(x)
    return neural_net.activation_output[0]

y = np.cos(np.arange(250)*(8*np.pi/250))[:,None]
X = np.arange(250)
X = np.column_stack((X, np.ones(X.shape[0])))

#print(X[20])

nn = NeuralNetwork(X, y, filename='mlp_cos_test', iterations=1000, input_nodes=2, hidden_nodes=6, output_nodes=1)
#print(nn.inspect_shape())
#print(X[20].shape)

nn.train()
nn.save()


X_test = np.arange(250,500)
mn = nn.load()

x_results = np.arange(250)
x_results_t = np.column_stack((x_results, np.ones(x_results.shape[0])))
y_results = np.apply_along_axis(lambda i: feed(mn, i), 1, np.atleast_2d(x_results_t))
print(y_results)


plt.figure()
plt.scatter(x_results, y_results)
plt.show()
