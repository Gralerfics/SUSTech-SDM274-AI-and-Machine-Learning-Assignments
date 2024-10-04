from enum import Enum

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


"""
    Linear Regression Model
"""
class OptimizerType(Enum):
    SGD = 1
    BGD = 2
    MBGD = 3

class NormalizationType(Enum):
    NONE = 0
    MINMAX = 1
    MEAN = 2

class LinearRegression:
    def __init__(self, dim = 1, learning_rate = 0.0001, max_iterations = 1000, tolerance = None, normalization_type = NormalizationType.NONE, optimizer_type = OptimizerType.SGD, batch_size = None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.normalization_type = normalization_type
        self.tolerance = tolerance

        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        if optimizer_type == OptimizerType.MBGD and batch_size is None:
            raise ValueError("batch_size should be specified for MBGD.")

        self.w = np.zeros(dim + 1)
        
        self.loss_history = None
        self.w_history = None
    
    @staticmethod
    def extend_X(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def min_max_normalization(X):
        X_min, X_max = np.min(X, axis = 0), np.max(X, axis = 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            X_normalized = (X - X_min) / (X_max - X_min)
        return np.where(np.isnan(X_normalized), X, X_normalized)

    @staticmethod
    def mean_normalization(X):
        pass
        # X_mean, X_std = np.mean(X, axis = 0), np.std(X, axis = 0)
        # with np.errstate(divide = 'ignore', invalid = 'ignore'):
        #     X_normalized = (X - X_mean) / X_std
        # return np.where(np.isnan(X_normalized), X, X_normalized)

    def predict(self, X):
        X_ext = self.extend_X(X)
        return X_ext @ self.w
    
    def loss(self, X, t):
        return np.mean((t - self.predict(X)) ** 2)

    def gradients(self, X, t):
        X_ext = self.extend_X(X)
        return -X_ext.T @ (t - self.predict(X)) / X_ext.shape[0]
    
    def optimize(self, X, t):
        if self.normalization_type == NormalizationType.MINMAX:
            X = self.min_max_normalization(X)
        elif self.normalization_type == NormalizationType.MEAN:
            X = self.mean_normalization(X)

        if self.tolerance is not None:
            last_loss = np.inf

        self.w_history = []
        self.loss_history = []

        for i in range(self.max_iterations):
            _X, _t = X, t
            if self.optimizer_type == OptimizerType.SGD:
                indices = np.random.randint(0, X.shape[0])
                _X, _t = X[indices:(indices + 1)], t[indices:(indices + 1)]
            elif self.optimizer_type == OptimizerType.MBGD:
                indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
                _X, _t = X[indices], t[indices]

            grads = self.gradients(_X, _t)
            self.w -= self.learning_rate * grads
            new_loss = self.loss(X, t)

            self.w_history.append(self.w.copy())
            self.loss_history.append(new_loss)

            if self.tolerance is not None:
                if np.abs(new_loss - last_loss) < self.tolerance:
                    print(f"Converged at iteration {i}.")
                    break
                last_loss = new_loss


"""
    Data Preprocessing (modified from HA-02.pdf)
"""
def random_split(X, y, test_size = 0.2, seed = None):
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int(len(X) * (1 - test_size))
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test


"""
    Test the code
"""
def plot_surface_path(X, t, w_history):
    def loss(w, X, t):
        return np.mean((t - LinearRegression.extend_X(X) @ w) ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    w0_vals = np.linspace(-10, 30, 100)
    w1_vals = np.linspace(0, 2, 100)
    W0, W1 = np.meshgrid(w0_vals, w1_vals)
    Z = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w = np.array([W0[i, j], W1[i, j]])
            Z[i, j] = loss(w, X, t)
    ax.plot_surface(W0, W1, Z, cmap=cm.coolwarm, alpha=0.6)

    # Descent path
    w_history = np.array(w_history)
    ax.plot(w_history[:, 0], w_history[:, 1], [loss(w, X, t) for w in w_history], 'r-o', markersize=5)

    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    ax.set_zlabel('Loss')
    plt.show()

if __name__ == "__main__":
    # Generate data (modified from HA-02.pdf)
    N = 100
    X_train = np.arange(N).reshape(N, 1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size = X_train.shape)
    y_train = y_train.reshape(-1)

    # Split the data
    X_train, y_train, X_test, y_test = random_split(X_train, y_train, test_size = 0.2, seed = 42)

    # Initialize the linear regression model
    model = LinearRegression(
        dim = 1,
        learning_rate = 0.0001,
        max_iterations = 100000,
        tolerance = 1e-5,
        normalization_type = NormalizationType.NONE,
        optimizer_type = OptimizerType.MBGD,
        batch_size = 10
    )
    model.optimize(X_train, y_train)

    # Optimization results
    print(model.w)

    # Closed-form solution
    X_ext = LinearRegression.extend_X(X_train)
    print(np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ y_train)

    # Optimized line
    # plt.scatter(X_train, y_train, color = 'blue')
    # plt.plot(X_train, model.predict(X_train), color = 'red')
    # plt.show()

    # Loss curve
    # plt.plot(loss_history)
    # plt.show()

    # Surface plot
    plot_surface_path(X_train, y_train, model.w_history)

