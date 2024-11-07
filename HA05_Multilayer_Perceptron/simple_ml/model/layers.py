import numpy as np

from . import Model
from ..data.types import Variable


class Linear(Model):
    def __init__(self, in_dim: int, out_dim: int):
        super(Linear, self).__init__()
        self.n = in_dim
        self.m = out_dim
        """
            * W: weights, n (in_dim) neurons -> m (out_dim) neurons
                W = [
                    [w_00, w_01, ..., w_0m],
                    [w_10, w_11, ..., w_1m],
                    ...,
                    [w_n0, w_n1, ..., w_nm]
                ]

            * b: biases, add by broadcasting
                b = [b_0, b_1, ..., b_m]

            * X: input data, N samples (batch size)
                X = [
                    [x_0^(0), x_1^(0), ..., x_n^(0)],
                    [x_0^(1), x_1^(1), ..., x_n^(1)],
                    ...,
                    [x_0^(N - 1), x_1^(N - 1), ..., x_n^(N - 1)]
                ]
        """
        # self.W = Variable(np.random.randn(self.n, self.m), derivable = True)
        self.W = Variable(np.random.uniform(-0.5, 0.5, (self.n, self.m)), derivable = True)
        # self.b = Variable(np.random.randn(self.m), derivable = True)
        self.b = Variable(np.ones(self.m) * 0.1, derivable = True)
            # TODO: initial value selection?
        self.params = [self.W, self.b]

    def forward(self, X):
        super(Linear, self).forward(X)
        """
            Y = X @ W + b
        """
        self.output = X @ self.W + self.b
        return self.output
    
    def backward(self):
        """
            * For multiple samples in X, take the average of gradients.
            * X = self.input.value
            * dE/dY = Y.gradient
            * dE/dW = dE/dY * dY/dW = X.value.T @ Y.gradient / N
                dE/dW[i, j] = 1/N * Sum_{k=0}^{N-1} {X[k, i] * dE/dY[k, j]}
                1/N is multiplied to take the average over N samples.
            * dE/db = dE/dY * dY/db = dE/dY * 1 = mean(Y.gradient, axis = 0)
                mean() is used to take the average over N samples.
            * dE/dX = Sum_{j=0}^{m-1} {dE/dY_j * dY_j/dX} = mean(Y.gradient @ W.value.T, axis = 0)
                Each line of dE/dY @ W.T is the gradient of the corresponding sample in X:
                    [Sum_{j=0}^{m-1} {dE/dY_j * W_0j}, Sum_{j=0}^{m-1} {dE/dY_j * W_1j}, ..., Sum_{j=0}^{m-1} {dE/dY_j * W_nj}]
                mean() is used to take the average over N samples.
        """
        X, Y = self.input, self.output
        N = X.value.shape[0]
        self.W.gradient = X.value.T @ Y.gradient / N
        self.b.gradient = np.mean(Y.gradient, axis = 0)
        X.gradient = np.mean(Y.gradient @ self.W.value.T, axis = 0)


class ReLU(Model):
    def forward(self, X):
        super(ReLU, self).forward(X)
        """
            Y = max(X, 0)
        """
        self.output = Variable(np.maximum(X.value, 0), derivable = True)
        return self.output
    
    def backward(self):
        """
            dE/dX = dE/dY * dY/dX = dE/dY * (X > 0)
        """
        self.input.gradient = self.output.gradient * (self.output.value > 0)


class Sigmoid(Model):
    def __init__(self, x_left_bound = -100, x_right_bound = 100):
        super(Sigmoid, self).__init__()
        self.x_left_bound = x_left_bound
        self.x_right_bound = x_right_bound
    
    def forward(self, X):
        super(Sigmoid, self).forward(X)
        """
            Y = 1 / (1 + exp(-X))
        """
        self.output = Variable(1 / (1 + np.exp(np.clip(-X.value, self.x_left_bound, self.x_right_bound))), derivable = True)
        return self.output
    
    def backward(self):
        """
            dE/dX = dE/dY * dY/dX = dE/dY * Y * (1 - Y)
        """
        self.input.gradient = self.output.gradient * self.output.value * (1 - self.output.value)


class Tanh(Model): # TODO: to be checked
    def __init__(self, x_left_bound = -100, x_right_bound = 100, epsilon = 1e-8):
        super(Tanh, self).__init__()
        self.x_left_bound = x_left_bound
        self.x_right_bound = x_right_bound
        self.epsilon = epsilon
    
    def forward(self, X):
        super(Tanh, self).forward(X)
        """
            Y = (exp(X) - exp(-X)) / (exp(X) + exp(-X))
        """
        exp_plus = np.exp(np.clip(X.value, self.x_left_bound, self.x_right_bound))
        exp_minus = np.exp(np.clip(-X.value, self.x_left_bound, self.x_right_bound))
        self.output = Variable((exp_plus - exp_minus) / (exp_plus + exp_minus + self.epsilon), derivable = True)
        # self.output = Variable((exp_plus - exp_minus) / np.maximum(exp_plus + exp_minus, self.epsilon), derivable = True)
        return self.output
    
    def backward(self):
        """
            dE/dX = dE/dY * dY/dX = dE/dY * (1 - Y ** 2)
        """
        self.input.gradient = self.output.gradient * (1 - self.output.value ** 2)


class Softmax(Model): # TODO: to be checked
    def __init__(self, x_left_bound = -100, x_right_bound = 100, epsilon = 1e-8):
        super(Softmax, self).__init__()
        self.x_left_bound = x_left_bound
        self.x_right_bound = x_right_bound
        self.epsilon = epsilon
    
    def forward(self, X):
        super(Softmax, self).forward(X)
        """
            Y = exp(X) / Sum_{i=0}^{n-1} {exp(X_i)}
        """
        exp = np.exp(np.clip(X.value, self.x_left_bound, self.x_right_bound))
        self.output = Variable(exp / np.sum(exp, axis = 1, keepdims = True), derivable = True)
        return self.output
    
    def backward(self):
        """
            dE/dX = dE/dY * dY/dX = dE/dY * (diag(Y) - Y @ Y.T)
        """
        Y = self.output.value
        self.input.gradient = self.output.gradient * (np.diag(Y) - Y[:, :, None] @ Y[:, None, :])   

