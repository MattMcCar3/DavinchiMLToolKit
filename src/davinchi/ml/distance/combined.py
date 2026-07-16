"""Combined distance over mixed numeric + categorical feature vectors."""
import numpy as np
from .vdm import vdm_distance


def combined_distance(x, y_vec, num_idx, cat_idx, vdm_model=None, p=2):
    """
    Distance between two full data points, summing Minkowski distance (no root)
    over numeric features and VDM distance over categorical features.

    NOTE: if cat_idx is non-empty but vdm_model is None, the categorical
    contribution is skipped entirely (distance from those features is 0).
    This is intentional and matches the numeric-features-only case, but it
    is a real footgun: any caller with categorical features MUST fit and
    pass a vdm_model, or distances silently degenerate. See
    ml/models/condensed_knn.py for the bug this caused historically, and
    tests/ml/test_distance.py for the regression test that guards it now.
    """
    dist = 0.0

    if len(num_idx) > 0:
        x_num = x[num_idx]
        y_num = y_vec[num_idx]
        dist += np.sum(np.abs(x_num - y_num) ** p)

    if len(cat_idx) > 0 and vdm_model is not None:
        x_cat = x[cat_idx]
        y_cat = y_vec[cat_idx]
        dist += vdm_distance(x_cat, y_cat, vdm_model)

    return dist
