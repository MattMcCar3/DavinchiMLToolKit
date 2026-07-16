"""Evaluation metrics."""
import numpy as np


def classification_error(y_true, y_pred):
    """1 - accuracy. Fraction of predictions that are wrong."""
    return np.mean(y_true != y_pred)


def mean_squared_error(y_true, y_pred):
    """Mean squared error for regression."""
    return np.mean((y_true - y_pred) ** 2)
