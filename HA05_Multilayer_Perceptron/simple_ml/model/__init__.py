from typing import Union

import numpy as np

from ..data.types import Variable


class Model:
    """ @Override (super().__init__() should be called) """
    def __init__(self):
        """ Model initialization """
        self.input = None
        self.output = None
        self.params = []
        pass # model hyperparameters
        pass # parameters -> self.params
    
    """ @Override (super().forward(X) should be called) """
    def forward(self, X: Variable) -> Variable:
        """ Forward propagation and model structure recording """
        assert isinstance(X, Variable)
        self.input = X # TODO: is it necessary to set self.input only when the model is called for the first time? Hint: currently the input of the input layer should be updated every time.
        pass # forward propagation and results -> self.output
        pass # return self.output
    
    """ @Override """
    def backward(self):
        """ Manually gradient calculation (forward propagation should be conducted before backward propagation) """
        pass # update .gradient of each Variable in self.input and self.params

    def parameters(self):
        """ A view of model parameters """
        return self.params

    def __call__(self, *args: Variable) -> Variable:
        return self.forward(*args)


class Sequential(Model):
    def __init__(self, layers_list):
        super(Sequential, self).__init__()
        self.layers = layers_list
        for layer in self.layers:
            self.params.extend(layer.parameters())
    
    def forward(self, X):
        super(Sequential, self).forward(X)
        self.output = X
        for layer in self.layers:
            self.output = layer(self.output)
        return self.output
    
    def backward(self):
        for layer in self.layers[::-1]:
            layer.backward()

