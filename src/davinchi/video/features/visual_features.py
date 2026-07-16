"""Frame-level visual features for cut detection: color histograms and
histogram-distance between consecutive frames. Simple and fast — no deep
learning needed to notice a hard cut between two very different frames."""
import cv2
import numpy as np


def frame_histogram(frame_bgr, bins=32):
    """
    Compute a normalized color histogram for a frame, concatenated across
    the 3 BGR channels. Cheap and reasonably robust to small camera shake —
    good enough for detecting hard cuts, not meant for anything subtler.
    """
    hist = []
    for channel in range(3):
        h = cv2.calcHist([frame_bgr], [channel], None, [bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.append(h)
    return np.concatenate(hist)


def histogram_distance(hist_a, hist_b):
    """Chi-square-like distance between two histograms; 0 = identical."""
    eps = 1e-10
    return np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b + eps))


def compute_visual_diff_series(frames_with_timestamps, bins=32):
    """
    Given an iterable of (timestamp, frame_bgr), return parallel arrays of
    timestamps and frame-to-frame histogram distances. The distance at
    index i is between frame i-1 and frame i (index 0 is always 0.0).
    """
    timestamps = []
    diffs = []
    prev_hist = None

    for ts, frame in frames_with_timestamps:
        hist = frame_histogram(frame, bins=bins)
        timestamps.append(ts)

        if prev_hist is None:
            diffs.append(0.0)
        else:
            diffs.append(histogram_distance(prev_hist, hist))

        prev_hist = hist

    return np.array(timestamps), np.array(diffs)
