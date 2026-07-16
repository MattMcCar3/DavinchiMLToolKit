"""Hard-cut detection: flag timestamps where frame-to-frame histogram
distance spikes above a threshold."""
from ..ingest.video_reader import read_frames
from ..features.visual_features import compute_visual_diff_series


def detect_cuts(video_path, threshold=0.5, sample_every_n=1, min_gap_sec=0.5):
    """
    Detect hard cuts in a video.

    Parameters
    ----------
    threshold : histogram-distance threshold above which a frame transition
        is considered a cut. Needs tuning per footage — start here and adjust.
    min_gap_sec : merge detections closer together than this, so one visual
        cut doesn't get reported as several nearly-identical timestamps.

    Returns
    -------
    list of float — timestamps (sec) of detected cuts
    """
    frames = read_frames(video_path, sample_every_n=sample_every_n)
    timestamps, diffs = compute_visual_diff_series(frames)

    cuts = []
    last_cut_time = -min_gap_sec

    for t, d in zip(timestamps, diffs):
        if d > threshold and (t - last_cut_time) >= min_gap_sec:
            cuts.append(float(t))
            last_cut_time = t

    return cuts
