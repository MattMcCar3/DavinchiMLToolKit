import numpy as np
from davinchi.ml.distance import minkowski_distance, fit_vdm, vdm_distance, combined_distance


def test_minkowski_euclidean():
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    assert minkowski_distance(x, y, p=2) == 5.0


def test_minkowski_identical_points_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert minkowski_distance(x, x, p=2) == 0.0


def test_vdm_distinguishes_categorical_values():
    # two categorical features, class label correlated with feature 0
    X_cat = np.array([[0], [0], [1], [1]])
    y = np.array([0, 0, 1, 1])
    vdm_model = fit_vdm(X_cat, y, p=2)

    same = vdm_distance(np.array([0]), np.array([0]), vdm_model)
    different = vdm_distance(np.array([0]), np.array([1]), vdm_model)

    assert same == 0.0
    assert different > 0.0


def test_combined_distance_skips_categorical_without_vdm_model():
    # documents the intentional (but dangerous) behavior: no vdm_model means
    # categorical features contribute 0 to the distance. Anything that calls
    # combined_distance with categorical features MUST supply a vdm_model.
    x = np.array([1.0])
    y = np.array([2.0])
    dist = combined_distance(x, y, num_idx=[], cat_idx=[0], vdm_model=None)
    assert dist == 0.0


def test_combined_distance_uses_vdm_when_provided():
    X_cat = np.array([[0], [0], [1], [1]])
    y_train = np.array([0, 0, 1, 1])
    vdm_model = fit_vdm(X_cat, y_train, p=2)

    x = np.array([0.0])
    y = np.array([1.0])
    dist = combined_distance(x, y, num_idx=[], cat_idx=[0], vdm_model=vdm_model)
    assert dist > 0.0
