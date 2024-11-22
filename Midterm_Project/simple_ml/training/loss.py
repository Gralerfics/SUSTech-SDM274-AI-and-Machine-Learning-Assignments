import numpy as np

from ..data.types import Variable


class Loss:
    """ @Override """
    def __init__(self):
        pass # hyperparameters

    """ @Override """
    def forward(self, X: Variable, T: np.ndarray):
        pass # return loss value

    """ @Override """
    def backward(self, X: Variable, T: np.ndarray):
        pass # save the gradient into X.gradient

    def calculate(self, X, T, calc_grad = True):
        """
            X is the input of the loss function (e.g. the output of the model).
            T is the target (label) of the model output.
            calc_grad is a flag to decide whether to calculate the gradient simultaneously (save into X.gradient).
        """
        loss = self.forward(X, T)
        if calc_grad:
            self.backward(X, T)
        return loss
    
    def __call__(self, X, T, calc_grad = True):
        return self.calculate(X, T, calc_grad)


class MSELoss(Loss):
    def forward(self, X, T): # TODO: 实现单维 T 自动转列向量，下同（在基类中实现一个函数？）。若只有一维，需从 X 形状判断其为多维单样本还是单维多样本。
        return np.mean((X.value - T) ** 2) / 2
    
    def backward(self, X, T):
        X.gradient = (X.value - T)


class CrossEntropyLoss(Loss):
    def __init__(self, epsilon = 1e-8, w_0 = 1, w_1 = 1):
        self.epsilon = epsilon
        self.w_0 = w_0
        self.w_1 = w_1

    def forward(self, X, T):
        """
            E = -[w_1 * t(n) * log(y(n)) + w_0 * (1 - t(n)) * log(1 - y(n))]
        """
        return -np.mean(self.w_1 * T * np.log(X.value + self.epsilon) + self.w_0 * (1 - T) * np.log(1 - X.value + self.epsilon))
    
    def backward(self, X, T):
        """
            dE/dy(n) = [y(n) - t(n)] / [y(n) * (1 - y(n))]
        """
        X.gradient = (X.value - T) / (X.value * (1 - X.value) + self.epsilon)


class BinaryPerceptronLoss(Loss):
    def _range_transform(self, T):
        """
            T is transformed to -1 and 1 for binary classification, where 0 -> -1 and 1 -> 1.
            TODO: customizing the target labels
        """
        return 2 * T - 1

    def forward(self, X, T):
        """
            L = max(0, -t * y)
        """
        T_transformed = self._range_transform(T)
        loss = np.maximum(0, -T_transformed * X.value)
        return np.mean(loss)
    
    def backward(self, X, T):
        """
            dL/dX = -T_transformed if -T_transformed * X < 0, otherwise 0
        """
        T_transformed = self._range_transform(T)
        gradient = np.where(-T_transformed * X.value > 0, -T_transformed, 0)
        X.gradient = gradient / X.value.shape[0]

