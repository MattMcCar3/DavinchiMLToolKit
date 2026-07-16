"""Held-out tuning splits and 5x2 cross-validation fold generation."""
import numpy as np


def make_tuning_split(X, y, task, tuning_frac=0.20, random_state=42):
    """
    Separate a held-out tuning set from the main data.

    Classification: stratified so class proportions are preserved.
    Regression: uniform sampling across the sorted target range.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    n_tune = int(round(n * tuning_frac))

    if task == "classification":
        tune_idx = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_cls_tune = int(round(len(cls_idx) * tuning_frac))
            tune_idx.extend(cls_idx[:n_cls_tune])
        tune_idx = np.array(sorted(tune_idx))
    else:
        order = np.argsort(y)
        # take every k-th sorted index to spread tuning points across the target range
        step = max(1, n // n_tune)
        tune_idx = order[::step][:n_tune]

    tune_mask = np.zeros(n, dtype=bool)
    tune_mask[tune_idx] = True

    X_tune, y_tune = X[tune_mask], y[tune_mask]
    X_main, y_main = X[~tune_mask], y[~tune_mask]
    return X_main, y_main, X_tune, y_tune


def make_5x2_folds(X, y, task, random_state=42):
    """
    Generate 5 rounds of 2-fold CV splits from the main data.

    Each round: randomly split data in half.
      - Classification: stratified split
      - Regression: uniform split (sort by target, interleave)

    Returns a list of 5 rounds, each a list of 2 fold dicts with train_idx/test_idx.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    rounds = []

    for _ in range(5):
        if task == "classification":
            idx_a, idx_b = [], []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0].copy()
                rng.shuffle(cls_idx)
                half = len(cls_idx) // 2
                idx_a.extend(cls_idx[:half])
                idx_b.extend(cls_idx[half:])
            idx_a, idx_b = np.array(idx_a), np.array(idx_b)
        else:
            order = np.argsort(y)
            shuffled_pairs = order.copy()
            rng.shuffle(shuffled_pairs)
            half = n // 2
            idx_a, idx_b = shuffled_pairs[:half], shuffled_pairs[half:]

        rounds.append([
            {"train_idx": idx_a, "test_idx": idx_b},
            {"train_idx": idx_b, "test_idx": idx_a},
        ])

    return rounds
