import numpy as np


class Optimizer:
    """ @Override (super().__init__() should be called) """
    def __init__(self, params):
        self.params = params
    
    # def zeroize_gradients(self): # TODO: is needed? gradient calculation is currently not accumulative.
    #     for param in self.params:
    #         param.gradient = np.zeros_like(param.value)

    """ @Override """
    def step(self):
        pass # update the parameters in self.params


class GD(Optimizer):
    """
        Gradient Descent Optimizer
        P.S. batch is decided by the data provider - in any case, the Variable().value's 0-th dimension is the batch size.
    """
    def __init__(self, params, lr = 0.01):
        super(GD, self).__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.gradient


class MomentumGD(Optimizer): # TODO: to be checked
    """
        Gradient Descent Optimizer with Momentum
    """
    def __init__(self, params, lr = 0.01, momentum = 0.9):
        super(MomentumGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(param.value) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.gradient
            param.value += self.velocity[i]



class Adam(Optimizer):
    """
        Adam Optimizer (TODO: Regularization to be checked)
    """
    def __init__(self, params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, regularization = None, regularization_lambda = 0.0):
        super(Adam, self).__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [np.zeros_like(param.value) for param in self.params]
        self.v = [np.zeros_like(param.value) for param in self.params]
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.gradient
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.gradient ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # regularization term
            if self.regularization == "L1":
                regularization_term = self.regularization_lambda * np.sign(param.value)
            elif self.regularization == "L2":
                regularization_term = self.regularization_lambda * param.value
            else:
                regularization_term = 0

            param.value -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + regularization_term)

