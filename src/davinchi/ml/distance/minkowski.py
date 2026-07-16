"""Minkowski distance for numeric features."""
import numpy as np


def minkowski_distance(x, y, p=2):
    """Minkowski distance between two 1D numeric feature vectors. p=2 is Euclidean."""
    differences = np.abs(x - y)
    powered = differences ** p
    total = np.sum(powered)
    return total ** (1.0 / p)
