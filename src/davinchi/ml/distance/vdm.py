"""Value Difference Metric (VDM) for categorical features."""
import numpy as np


def fit_vdm(X_cat, y, p=2):
    """
    Fit a VDM model: for each categorical feature, map each observed value to
    its per-class proportion vector among training examples with that value.

    Parameters
    ----------
    X_cat : np.ndarray, shape (n_samples, n_cat_features)
    y : np.ndarray, shape (n_samples,)
    p : norm order, applied later in vdm_distance

    Returns
    -------
    dict with "tables" (list of {value: proportions} per feature), "classes", "p"
    """
    classes = np.unique(y)
    n_features = X_cat.shape[1]
    vdm_tables = []

    for f in range(n_features):
        feature_vals = np.unique(X_cat[:, f])
        table = {}
        for v in feature_vals:
            mask = X_cat[:, f] == v
            C_i = np.sum(mask)
            proportions = np.array([
                np.sum(mask & (y == c)) / C_i
                for c in classes
            ])
            table[v] = proportions
        vdm_tables.append(table)

    return {"tables": vdm_tables, "classes": classes, "p": p}


def vdm_distance(x, y_vec, vdm_model):
    """
    VDM distance between two categorical feature vectors, given a fitted vdm_model.
    Unseen values fall back to uniform class proportions (maximally uncertain).
    """
    tables = vdm_model["tables"]
    p = vdm_model["p"]
    total = 0.0

    for f, table in enumerate(tables):
        vi = x[f]
        vj = y_vec[f]
        if vi == vj:
            continue

        uniform = np.ones(len(vdm_model["classes"])) / len(vdm_model["classes"])
        pi = table.get(vi, uniform)
        pj = table.get(vj, uniform)
        total += np.sum(np.abs(pi - pj) ** p)

    return total
