"""Unifies cut points, silence spans, and stage labels into one timeline
representation, independent of any editing software. This is the seam
between detection (video/) and Resolve integration (resolve/) — Resolve
code only ever needs to understand this structure, not three separate
detector outputs.
"""


def build_timeline(duration_sec, cuts=None, silence_spans=None, stage_labels=None):
    """
    Parameters
    ----------
    duration_sec : float — total video duration
    cuts : list of float timestamps (sec) — points to consider splitting at
    silence_spans : list of (start, end) tuples — dead air to consider trimming
    stage_labels : list of (timestamp, label) tuples — painting stage at that point

    Returns
    -------
    dict — a plain-data timeline representation:
        {
            "duration_sec": float,
            "cuts": [float, ...],
            "silence_spans": [{"start": float, "end": float}, ...],
            "stage_labels": [{"timestamp": float, "label": str}, ...],
        }
    """
    cuts = sorted(cuts or [])
    silence_spans = silence_spans or []
    stage_labels = stage_labels or []

    return {
        "duration_sec": duration_sec,
        "cuts": cuts,
        "silence_spans": [{"start": s, "end": e} for s, e in silence_spans],
        "stage_labels": [{"timestamp": t, "label": lbl} for t, lbl in stage_labels],
    }
