from typing import Union

import numpy as np

from ..data.types import Variable


def eval_binary_accuracy(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
    """
        Y: the output of the model (output values)
        T: the target (label) of the model output (one of two integers)

        z_func: the function float -> int to convert the output values to labels
        if z_func is None, use the default method:
            decision_boundary: the threshold to decide the label
                > decision_boundary, -> label_pos: the positive label
                < decision_boundary, -> label_neg: the negative label
    """
    if isinstance(Y, Variable):
        Y = Y.value
    if z_func is not None:
        Y = z_func(Y)
    else:
        Y = np.where(Y >= decision_boundary, label_pos, label_neg)
    return np.mean(np.round(Y).astype(int) == T.astype(int))


def eval_classify_accuracy(Y: Union[Variable, np.ndarray], T: np.ndarray):
    """
        Y: the output of the model (probabilities of each class)
        T: the target (label) of the model output (one-hot encoding)
    """
    if isinstance(Y, Variable):
        Y = Y.value
    Y = np.argmax(Y, axis = 1)
    T = np.argmax(T, axis = 1)
    return np.mean(Y == T)

