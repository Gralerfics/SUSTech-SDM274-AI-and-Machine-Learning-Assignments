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

    def _min_max_normalize(self, X, record_params = False):
        if record_params:
            self.X_min, self.X_max = np.min(X, axis = 0), np.max(X, axis = 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            X_normalized = (X - self.X_min) / (self.X_max - self.X_min)
        return np.where(np.isnan(X_normalized), X, X_normalized)

    def _mean_normalize(self, X, record_params = False): # TODO
        if record_params:
            self.X_mean, self.X_std = np.mean(X, axis = 0), np.std(X, axis = 0)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            X_normalized = (X - self.X_mean) / self.X_std
        return np.where(np.isnan(X_normalized), X, X_normalized)
    
    def _normalize(self, X, record_params = False):
        if self.normalization_type == NormalizationType.MINMAX:
            return self._min_max_normalize(X, record_params)
        elif self.normalization_type == NormalizationType.MEAN:
            return self._mean_normalize(X, record_params)
        return X

    @staticmethod
    def _predict(X, w): # model 在 X 上 w 处的预测
        X_ext = LinearRegression.extend_X(X)
        return X_ext @ w
    
    def predict(self, X, w = None): # model 在 X 上 w 处的预测 (输入的 X 未归一化; 不传入 w 时使用当前的 w)
        w = self.w if w is None else w
        X_norm = self._normalize(X)
        return LinearRegression._predict(X_norm, w)
    
    @staticmethod
    def _loss(X, t, w): # (X, t) 的 loss 曲面上 w 处的值
        return np.mean((t - LinearRegression._predict(X, w)) ** 2)
    
    def loss(self, X, t, w = None): # (X, t) 的 loss 曲面上 w 处的值 (输入的 X 未归一化; 不传入 w 时使用当前的 w)
        w = self.w if w is None else w
        X_norm = self._normalize(X)
        return LinearRegression._loss(X_norm, t, w)

    @staticmethod
    def _gradients(X, t, w): # (X, t) 的 loss 曲面上 w 处的梯度
        X_ext = LinearRegression.extend_X(X)
        return -X_ext.T @ (t - LinearRegression._predict(X, w)) / X_ext.shape[0]
    
    def optimize(self, X, t):
        # Normalization for the first time
        X_norm = self._normalize(X, record_params = True)

        # Initialize history
        if self.tolerance is not None:
            last_loss = np.inf
        self.w_history = []
        self.loss_history = []

        for i in range(self.max_iterations):
            # Select samples
            _X, _t = X_norm, t
            if self.optimizer_type == OptimizerType.SGD:
                indices = np.random.randint(0, _X.shape[0])
                _X, _t = _X[indices:(indices + 1)], _t[indices:(indices + 1)]
            elif self.optimizer_type == OptimizerType.MBGD:
                indices = np.random.choice(_X.shape[0], self.batch_size, replace = False)
                _X, _t = _X[indices], _t[indices]

            # Calculate gradients on the selected samples
            grads = self._gradients(_X, _t, self.w)

            # Update weights
            self.w -= self.learning_rate * grads

            # Calculate overall loss and check convergence
            new_loss = self._loss(X_norm, t, self.w)
            if self.tolerance is not None:
                if np.abs(new_loss - last_loss) < self.tolerance:
                    print(f"Converged at iteration {i}.")
                    break
                last_loss = new_loss

            # Record history
            self.w_history.append(self.w.copy())
            self.loss_history.append(new_loss)


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
def plot_results(model, X, t):
    # Optimization results
    print(model.w)

    # Closed-form solution
    X_norm = model._normalize(X)
    X_ext = LinearRegression.extend_X(X_norm)
    optimal_w = np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ t
    print(optimal_w)

    # Plot
    fig = plt.figure(figsize = (30, 9))
    fig.suptitle(f'Linear Regression\n(learning_rate = {model.learning_rate}, max_iterations = {model.max_iterations}, tolerance = {model.tolerance}, normalization_type = {model.normalization_type}, optimizer_type = {model.optimizer_type}, batch_size = {model.batch_size})', fontsize = 16)

    # Surface, Descent path & Optimal point
    ax1 = fig.add_subplot(131, projection = '3d')
    margin_ratio = 0.12
    w_history = np.array(model.w_history)
    w0_min, w0_max = np.min(w_history[:, 0]), np.max(w_history[:, 0])
    w1_min, w1_max = np.min(w_history[:, 1]), np.max(w_history[:, 1])
    w0_min, w0_max = min(w0_min, optimal_w[0]), max(w0_max, optimal_w[0])
    w1_min, w1_max = min(w1_min, optimal_w[1]), max(w1_max, optimal_w[1])
    w0_min, w0_max = w0_min - margin_ratio * (w0_max - w0_min), w0_max + margin_ratio * (w0_max - w0_min)
    w1_min, w1_max = w1_min - margin_ratio * (w1_max - w1_min), w1_max + margin_ratio * (w1_max - w1_min)
    w0_mesh, w1_mesh = np.meshgrid(
        np.linspace(w0_min, w0_max, 100),
        np.linspace(w1_min, w1_max, 100)
    )
    loss_mesh = np.zeros_like(w0_mesh)
    for i in range(w0_mesh.shape[0]):
        for j in range(w0_mesh.shape[1]):
            w = np.array([w0_mesh[i, j], w1_mesh[i, j]])
            loss_mesh[i, j] = model.loss(X, t, w)
    ax1.plot_surface(w0_mesh, w1_mesh, loss_mesh, cmap = cm.coolwarm, alpha = 0.6)
    ax1.plot(w_history[:, 0], w_history[:, 1], [model.loss(X, t, w) for w in w_history], 'r-o', markersize = 1)
    ax1.scatter(optimal_w[0], optimal_w[1], model.loss(X, t, optimal_w), color = 'blue', s = 10)
    ax1.set_xlabel('w_0')
    ax1.set_ylabel('w_1')
    ax1.set_zlabel('Loss (MSE)')
    ax1.set_title('Loss Surface & Descent Path')

    # Optimized line
    ax2 = fig.add_subplot(132)
    X_ext = LinearRegression.extend_X(X_norm)
    ax2.scatter(X, t, color = 'blue')
    ax2.plot(X, X_ext @ model.w, color = 'red')
    ax2.set_xlabel('X')
    ax2.set_ylabel('t')
    ax2.set_title('Optimized Line')

    # Loss curve
    ax3 = fig.add_subplot(133)
    ax3.plot(model.loss_history)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Curve')

    # Show plot
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
        learning_rate = 0.01,
        max_iterations = 10000,
        tolerance = 1e-6,
        normalization_type = NormalizationType.MEAN,
        optimizer_type = OptimizerType.SGD,
        # batch_size = 16
    )
    model.optimize(X_train, y_train)

    # Plot the results
    plot_results(model, X_train, y_train)

