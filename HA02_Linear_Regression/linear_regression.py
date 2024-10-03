import numpy as np


"""
    [Loss] Mean Squared Error
"""
def MeanSquaredError(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2)


"""
    [Optimizer] Stochastic Gradient Descent (SGD)
"""
pass

"""
    [Optimizer] Batch Gradient Descent (BGD)
"""
pass

"""
    [Optimizer] Mini-Batch Gradient Descent (MBGD)
"""
pass


"""
    [Model] Linear Regression
"""
pass


"""
    Test the LinearRegression class
"""
if __name__ == "__main__":
    # Random data (from HA-02.pdf)
    X_train = np.arange(10).reshape(100, 1)
    a, b = 1, 10
    y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
    y_train = y_train.reshape(-1)

    # TODO

