"""
Condensed (edited) k-NN: compress a training set by keeping only points that
are necessary to preserve the decision boundary / target tolerance.

Historical note: an earlier version of this function never received a fitted
vdm_model, and combined_distance() silently zeroes out categorical distance
when vdm_model is None. On all-categorical datasets that made every pairwise
distance evaluate to 0, so "condensing" degenerated into an arbitrary
comparison against a single reference point. Fixed here by fitting VDM before
condensing and threading it through to combined_distance. See
tests/ml/test_condensed_knn.py for the regression test.
"""
import numpy as np
from ..distance.combined import combined_distance
from ..distance.vdm import fit_vdm


def condensed_knn(X_train, y_train, task, num_idx, cat_idx, epsilon=None, vdm_model=None, p=2):
    """
    Parameters
    ----------
    task : "classification" or "regression"
    epsilon : required for regression; a point is added if its nearest
        condensed neighbor's target differs by more than epsilon.
    vdm_model : fitted VDM model (required if cat_idx is non-empty, for
        distances to be meaningful — see module docstring).
    """
    if len(cat_idx) > 0 and vdm_model is None:
        raise ValueError(
            "condensed_knn received categorical features (cat_idx non-empty) "
            "but no vdm_model. Distances would silently degenerate to 0 for "
            "those features — fit_vdm() first and pass the result in."
        )

    condensed_indices = {0}
    condensed_X = [X_train[0]]
    condensed_y = [y_train[0]]
    changed = True

    while changed:
        changed = False
        c_X = np.array(condensed_X)
        c_y = np.array(condensed_y)

        for i in range(1, len(X_train)):
            if i in condensed_indices:
                continue

            x_i = X_train[i]
            y_i = y_train[i]

            if len(cat_idx) == 0:
                distances = np.sum(np.abs(c_X[:, num_idx] - x_i[num_idx]) ** p, axis=1)
            else:
                distances = np.array([
                    combined_distance(x_i, x_c, num_idx, cat_idx, vdm_model, p)
                    for x_c in c_X
                ])

            nn_idx = np.argsort(distances)[0]
            nearest_label = c_y[nn_idx]

            if task == "classification":
                if nearest_label != y_i:
                    condensed_X.append(x_i)
                    condensed_y.append(y_i)
                    condensed_indices.add(i)
                    changed = True
            elif task == "regression":
                if abs(nearest_label - y_i) > epsilon:
                    condensed_X.append(x_i)
                    condensed_y.append(y_i)
                    condensed_indices.add(i)
                    changed = True

    return np.array(condensed_X), np.array(condensed_y)


def fit_vdm_for_condensing(X_train, y_train, cat_idx, p=2):
    """Convenience wrapper: fit VDM on categorical columns before condensing."""
    if len(cat_idx) == 0:
        return None
    return fit_vdm(X_train[:, cat_idx], y_train, p=p)
