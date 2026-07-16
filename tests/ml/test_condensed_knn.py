import numpy as np
import pytest
from davinchi.ml.distance import fit_vdm
from davinchi.ml.models.condensed_knn import condensed_knn, fit_vdm_for_condensing


def test_condensed_knn_raises_without_vdm_for_categorical_features():
    """
    Regression test for the bug found in the original k-NN toolkit:
    condensing categorical data without a fitted VDM model silently
    degenerates (all distances become 0). This must now raise instead
    of silently producing garbage.
    """
    X_train = np.array([[0], [1], [0], [1]])
    y_train = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="vdm_model"):
        condensed_knn(
            X_train, y_train, task="classification",
            num_idx=[], cat_idx=[0], vdm_model=None,
        )


def test_condensed_knn_classification_with_vdm_compresses_reasonably():
    # two well-separated categorical clusters — condensing should keep far
    # fewer than all points, since most points agree with their neighbors
    rng = np.random.RandomState(0)
    n_per_class = 30
    X_train = np.array([[0]] * n_per_class + [[1]] * n_per_class)
    y_train = np.array([0] * n_per_class + [1] * n_per_class)

    vdm_model = fit_vdm_for_condensing(X_train, y_train, cat_idx=[0])
    c_X, c_y = condensed_knn(
        X_train, y_train, task="classification",
        num_idx=[], cat_idx=[0], vdm_model=vdm_model,
    )

    assert len(c_y) < len(y_train)
    # every retained point's label should still be recoverable from the set
    assert set(np.unique(c_y)) == {0, 1}


def test_condensed_knn_numeric_only_unaffected_by_vdm_wiring():
    # numeric-only case shouldn't require or use a vdm_model at all
    X_train = np.array([[0.0], [0.1], [5.0], [5.1]])
    y_train = np.array([0, 0, 1, 1])

    c_X, c_y = condensed_knn(
        X_train, y_train, task="classification",
        num_idx=[0], cat_idx=[], vdm_model=None,
    )
    assert len(c_y) <= len(y_train)
