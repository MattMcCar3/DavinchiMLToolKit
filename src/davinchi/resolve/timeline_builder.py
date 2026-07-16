"""
Push a davinchi timeline dict (see video/timeline/segments.py) into an
already-open Resolve timeline as markers.

V1 scope, deliberately: this ADDS MARKERS onto the existing timeline rather
than attempting automatic cuts/blade edits. Actually splitting clips via the
API is significantly more involved (and more dangerous to get wrong on real
footage) — markers give you a reviewable, non-destructive first pass: open
the timeline in Resolve, see exactly where the detectors think the cuts/
silence/stage changes are, and cut manually or extend this module once the
detectors are proven trustworthy on your footage.

UNTESTED against real Resolve — verify marker color names and frame-rate
handling against your Resolve version before trusting it on real footage.
"""

MARKER_COLORS = {
    "cut": "Red",
    "silence": "Blue",
    "stage": "Green",
}


def _seconds_to_frame_id(seconds, frame_rate):
    return int(round(seconds * frame_rate))


def push_timeline_markers(timeline, resolve_timeline, frame_rate):
    """
    Parameters
    ----------
    timeline : dict — output of davinchi.video.timeline.build_timeline
    resolve_timeline : Resolve Timeline object (from get_current_timeline())
    frame_rate : float — the Resolve project/timeline frame rate, needed to
        convert our second-based timestamps into Resolve's frame IDs.

    Returns
    -------
    int — number of markers successfully added
    """
    added = 0

    for cut_ts in timeline["cuts"]:
        frame_id = _seconds_to_frame_id(cut_ts, frame_rate)
        ok = resolve_timeline.AddMarker(
            frame_id, MARKER_COLORS["cut"], "Cut", f"Detected cut at {cut_ts:.2f}s", 1, ""
        )
        added += int(bool(ok))

    for span in timeline["silence_spans"]:
        frame_id = _seconds_to_frame_id(span["start"], frame_rate)
        duration_frames = max(1, _seconds_to_frame_id(span["end"] - span["start"], frame_rate))
        ok = resolve_timeline.AddMarker(
            frame_id, MARKER_COLORS["silence"], "Silence",
            f"Silent {span['start']:.2f}s - {span['end']:.2f}s", duration_frames, ""
        )
        added += int(bool(ok))

    for label in timeline["stage_labels"]:
        frame_id = _seconds_to_frame_id(label["timestamp"], frame_rate)
        ok = resolve_timeline.AddMarker(
            frame_id, MARKER_COLORS["stage"], "Stage",
            f"Stage: {label['label']}", 1, ""
        )
        added += int(bool(ok))

    return added
