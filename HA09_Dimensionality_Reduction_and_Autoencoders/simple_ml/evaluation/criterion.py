from typing import Union

import numpy as np

from ..data.types import Variable


def eval_multiclass_accuracy(Y: Union[Variable, np.ndarray], T: np.ndarray):
    """
        Y: the output of the model (probabilities of each class)
        T: the target (label) of the model output (one-hot encoding)
    """
    if isinstance(Y, Variable):
        Y = Y.value
    Y = np.argmax(Y, axis = 1)
    T = np.argmax(T, axis = 1)
    return np.mean(Y == T)


def eval_binary_confusion_matrix(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
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
    Y = Y.astype(int)
    T = T.astype(int)
    TP = np.sum((Y == T) & (T == label_pos))
    TN = np.sum((Y == T) & (T == label_neg))
    FP = np.sum((Y != T) & (T == label_neg))
    FN = np.sum((Y != T) & (T == label_pos))
    return np.array([[TP, FN], [FP, TN]])


def eval_binary_accuracy(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
    """
        Accuracy = T / (T + F) = (TP + TN) / (TP + TN + FP + FN)

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
    Y = np.round(Y).astype(int)
    T = T.astype(int)
    return np.mean(Y == T)


def eval_binary_recall(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
    """
        Recall = TP / (TP + FN)

        The same as eval_binary_accuracy, but return the recall rate.
    """
    if isinstance(Y, Variable):
        Y = Y.value
    if z_func is not None:
        Y = z_func(Y)
    else:
        Y = np.where(Y >= decision_boundary, label_pos, label_neg)
    Y = np.round(Y).astype(int)
    T = T.astype(int)
    TP_FN = np.sum(T == label_pos)
    return np.sum((Y == T) & (T == label_pos)) / TP_FN if TP_FN > 0 else 1


def eval_binary_precision(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
    """
        Precision = TP / P = TP / (TP + FP)

        The same as eval_binary_accuracy, but return the precision rate.
    """
    if isinstance(Y, Variable):
        Y = Y.value
    if z_func is not None:
        Y = z_func(Y)
    else:
        Y = np.where(Y >= decision_boundary, label_pos, label_neg)
    Y = np.round(Y).astype(int)
    T = T.astype(int)
    P = np.sum(Y == label_pos)
    return np.sum((Y == T) & (T == label_pos)) / P if P > 0 else 1


def eval_binary_f1_score(Y: Union[Variable, np.ndarray], T: np.ndarray, decision_boundary = 0.0, label_pos: int = 1, label_neg: int = -1, z_func = None):
    """
        F1 Score = 2 * Precision * Recall / (Precision + Recall)

        The same as eval_binary_accuracy, but return the F1 score.
    """
    precision = eval_binary_precision(Y, T, decision_boundary, label_pos, label_neg, z_func)
    recall = eval_binary_recall(Y, T, decision_boundary, label_pos, label_neg, z_func)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


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


def eval_regression_r2(Y: Union[Variable, np.ndarray], T: np.ndarray): # TODO: to be checked
    if isinstance(Y, Variable):
        Y = Y.value
    SS_res = np.sum((Y - T) ** 2)
    SS_tot = np.sum((T - np.mean(T)) ** 2)
    return 1 - SS_res / SS_tot if SS_tot > 0 else 0

