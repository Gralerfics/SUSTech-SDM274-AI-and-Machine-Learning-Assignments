from enum import Enum

import numpy as np

import matplotlib.pyplot as plt


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

        for i in range(self.max_iterations):
            _X, _t = X, t
            if self.optimizer_type == OptimizerType.SGD:
                indices = np.random.randint(0, X.shape[0])
                _X, _t = X[indices:(indices + 1)], t[indices:(indices + 1)]
            # elif self.optimizer_type == OptimizerType.BGD:
            #     pass
            elif self.optimizer_type == OptimizerType.MBGD:
                indices = np.random.choice(X.shape[0], self.batch_size, replace=False)
                _X, _t = X[indices], t[indices]

            grads = self.gradients(_X, _t)
            self.w -= self.learning_rate * grads
            new_loss = self.loss(X, t)

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
if __name__ == "__main__":
    # Generate data (modified from HA-02.pdf)
    N = 100
    X_train = np.arange(N).reshape(N, 1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 1, size = X_train.shape) # TODO 1, 5
    y_train = y_train.reshape(-1)

    # Split the data
    X_train, y_train, X_test, y_test = random_split(X_train, y_train, test_size = 0.2, seed = 42)

    # Initialize the linear regression model
    model = LinearRegression(
        dim = 1,
        learning_rate = 0.0005,
        max_iterations = 100000,
        tolerance = 1e-5,
        use_min_max_normalization = True,
        optimizer_type = OptimizerType.BGD
    )
    model.optimize(X_train, y_train)

    # Print the results
    print(model.w)

    X_ext = LinearRegression.extend_X(X_train)
    print(np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ y_train)

    # Plot the results
    plt.scatter(X_train, y_train, color = 'blue')
    plt.plot(X_train, model.predict(X_train), color = 'red')
    plt.show()

