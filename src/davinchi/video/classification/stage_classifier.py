"""Painting-stage classification from frame color features, using the same
k-NN implementation built and fixed for the JHU coursework project.

This is a genuine reuse of davinchi.ml, not a rewrite: label a handful of
reference frames per stage (e.g. "primed", "base_coat", "shading",
"highlights", "finished"), and classify new frames against them by
majority vote among nearest neighbors in color-feature space.
"""
import numpy as np
from ...ml.models.knn import knn_classify
from ...ml.preprocessing.normalize import normalize_fold
from ..features.color_features import color_feature_vector


class StageClassifier:
    def __init__(self, k=3):
        self.k = k
        self._X_train = None
        self._y_train = None
        self._mu = None
        self._sigma = None

    def fit(self, labeled_frames):
        """
        Parameters
        ----------
        labeled_frames : list of (frame_bgr, stage_label) tuples — your
            reference/example frames for each painting stage.
        """
        X = np.array([color_feature_vector(f) for f, _ in labeled_frames])
        y = np.array([label for _, label in labeled_frames])

        # normalize using training stats only, same discipline as the ml pipeline
        self._X_train, _, self._mu, self._sigma = normalize_fold(X, X)
        self._y_train = y
        return self

    def predict(self, frame_bgr):
        """Classify a single frame's painting stage."""
        if self._X_train is None:
            raise RuntimeError("StageClassifier.fit() must be called before predict()")

        x = color_feature_vector(frame_bgr)
        x_norm = (x - self._mu) / self._sigma
        x_norm = x_norm.reshape(1, -1)

        num_idx = list(range(x_norm.shape[1]))
        pred = knn_classify(
            self._X_train, self._y_train, x_norm, self.k,
            num_idx=num_idx, cat_idx=[], vdm_model=None, p=2,
        )
        return pred[0]
