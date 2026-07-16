import numpy as np
from davinchi.video.features.visual_features import (
    frame_histogram, histogram_distance, compute_visual_diff_series
)


def make_solid_frame(color_bgr, size=(20, 20)):
    frame = np.zeros((*size, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


def test_histogram_distance_zero_for_identical_frames():
    frame = make_solid_frame((0, 0, 255))
    h1 = frame_histogram(frame)
    h2 = frame_histogram(frame)
    assert histogram_distance(h1, h2) == 0.0


def test_histogram_distance_positive_for_different_frames():
    red = frame_histogram(make_solid_frame((0, 0, 255)))
    blue = frame_histogram(make_solid_frame((255, 0, 0)))
    assert histogram_distance(red, blue) > 0.0


def test_visual_diff_series_flags_the_cut():
    frames = [
        (0.0, make_solid_frame((0, 0, 255))),
        (0.1, make_solid_frame((0, 0, 255))),
        (0.2, make_solid_frame((255, 0, 0))),  # the cut
        (0.3, make_solid_frame((255, 0, 0))),
    ]
    timestamps, diffs = compute_visual_diff_series(frames)
    assert diffs[0] == 0.0          # first frame has no predecessor
    assert diffs[1] == 0.0          # identical to previous
    assert diffs[2] > diffs[1]      # the cut shows up as a spike
