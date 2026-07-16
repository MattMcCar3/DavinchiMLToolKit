"""Audio-level features for silence detection: simple RMS energy envelope.
Starting simple on purpose — a threshold on RMS is enough to find dead air;
no need for spectral features or a heavier audio library yet."""
import numpy as np


def compute_rms_envelope(waveform, sample_rate, window_sec=0.5, hop_sec=0.25):
    """
    Compute RMS energy over sliding windows.

    Returns
    -------
    timestamps : np.ndarray — center time (sec) of each window
    rms : np.ndarray — RMS energy of each window
    """
    window_size = int(window_sec * sample_rate)
    hop_size = int(hop_sec * sample_rate)

    timestamps = []
    rms_values = []

    for start in range(0, len(waveform) - window_size + 1, hop_size):
        window = waveform[start:start + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        center_time = (start + window_size / 2) / sample_rate

        timestamps.append(center_time)
        rms_values.append(rms)

    return np.array(timestamps), np.array(rms_values)
