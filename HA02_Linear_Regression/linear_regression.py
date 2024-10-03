import numpy as np


"""
    [Optimizer] Gradient Descent Base Class
"""
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.001, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.optim_vars: dict = None

    def update(self, model, aux_vars):
        pass

    def optimize(self, model, aux_vars):
        pass

"""
    [Optimizer] Stochastic Gradient Descent (SGD)
"""
pass # TODO

"""
    [Optimizer] Batch Gradient Descent (BGD)
"""
pass # TODO

"""
    [Optimizer] Mini-Batch Gradient Descent (MBGD)
"""
pass # TODO


"""
    [Model] Linear Regression
"""
class LinearRegression:
    def __init__(self):
        pass

    # TODO


"""
    Data Preprocessing
"""
def random_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
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
    # Random data (from HA-02.pdf)
    X_train = np.arange(100).reshape(100, 1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1)

    # Split the data
    X_train, y_train, X_test, y_test = random_split(X_train, y_train, test_size=0.2, random_state=42)

    # TODO

