import numpy as np

from .. import Variable


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
    def forward(self, X, T):
        return np.mean((X.value - T) ** 2) / 2
    
    def backward(self, X, T):
        X.gradient = (X.value - T)


class CrossEntropyLoss(Loss):
    def __init__(self, epsilon = 1e-8):
        self.epsilon = epsilon

    def forward(self, X, T):
        """
            E = -[t(n) * log(y(n)) + (1 - t(n)) * log(1 - y(n))]
        """
        return -np.mean(T * np.log(X.value + self.epsilon) + (1 - T) * np.log(1 - X.value + self.epsilon))
    
    def backward(self, X, T):
        """
            dE/dy(n) = [y(n) - t(n)] / [y(n) * (1 - y(n))]
        """
        X.gradient = (X.value - T) / (X.value * (1 - X.value) + self.epsilon)

