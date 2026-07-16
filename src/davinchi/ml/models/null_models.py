"""Baseline models that ignore features entirely."""
import numpy as np


def null_classifier(y_train, X_test):
    """Predicts the plurality (most common) class label for every test point."""
    classes, counts = np.unique(y_train, return_counts=True)
    plurality_class = classes[np.argmax(counts)]
    return np.full(len(X_test), plurality_class, dtype=y_train.dtype)


def null_regressor(y_train, X_test):
    """Predicts the mean of training targets for every test point."""
    mean_val = np.mean(y_train)
    return np.full(len(X_test), mean_val, dtype=float)
