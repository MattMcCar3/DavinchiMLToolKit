"""Silence/dead-air detection: flag spans where RMS energy stays below a
threshold for at least a minimum duration."""
from ..ingest.audio_reader import extract_audio_waveform
from ..features.audio_features import compute_rms_envelope


def detect_silence(video_path, rms_threshold=0.02, min_duration_sec=1.0,
                    window_sec=0.5, hop_sec=0.25):
    """
    Detect silent spans in a video's audio track.

    Parameters
    ----------
    rms_threshold : RMS energy below this counts as "silent". Depends on
        mic gain and room noise floor — needs tuning per setup.
    min_duration_sec : only report spans at least this long, so brief
        pauses between words don't get flagged as dead air.

    Returns
    -------
    list of (start_sec, end_sec) tuples
    """
    waveform, sr = extract_audio_waveform(video_path)
    timestamps, rms = compute_rms_envelope(waveform, sr, window_sec=window_sec, hop_sec=hop_sec)

    is_silent = rms < rms_threshold
    spans = []
    span_start = None

    for i, silent in enumerate(is_silent):
        t = timestamps[i]
        if silent and span_start is None:
            span_start = t
        elif not silent and span_start is not None:
            if t - span_start >= min_duration_sec:
                spans.append((float(span_start), float(t)))
            span_start = None

    if span_start is not None:
        end_t = timestamps[-1]
        if end_t - span_start >= min_duration_sec:
            spans.append((float(span_start), float(end_t)))

    return spans
