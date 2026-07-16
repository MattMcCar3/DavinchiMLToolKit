"""Frame-level color feature vector for stage classification (feeds into
davinchi.ml.models.knn — same k-NN you already built and fixed for the
coursework project, now reused for a real task)."""
import cv2
import numpy as np


def color_feature_vector(frame_bgr, bins=8):
    """
    Compact color descriptor for a frame: mean + std of each HSV channel,
    plus a coarse hue histogram. Deliberately simple — enough to
    distinguish visually distinct painting stages (e.g. bare primer vs.
    base coat vs. highlights) without anything resembling deep learning.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    means = hsv.reshape(-1, 3).mean(axis=0)
    stds = hsv.reshape(-1, 3).std(axis=0)

    hue_hist = cv2.calcHist([hsv.astype(np.uint8)], [0], None, [bins], [0, 180])
    hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()

    return np.concatenate([means, stds, hue_hist])
