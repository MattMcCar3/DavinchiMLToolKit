"""Z-score normalization, fit on training data only."""
import numpy as np


def compute_z_score_normalization(X_train):
    """
    Compute mean and standard deviation from training data only.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    mu : np.ndarray, shape (n_features,)
    sigma : np.ndarray, shape (n_features,)
    """
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma == 0] = 1.0  # avoid divide-by-zero for constant columns
    return mu, sigma


def normalize_fold(X_train, X_test):
    """
    Fit z-score params on training fold only, apply to both train and test.
    Test data plays no part in computing mu or sigma.
    """
    mu, sigma = compute_z_score_normalization(X_train)
    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma
    return X_train_norm, X_test_norm, mu, sigma
