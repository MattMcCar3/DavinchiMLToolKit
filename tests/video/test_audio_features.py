import numpy as np
from davinchi.video.features.audio_features import compute_rms_envelope


def test_rms_envelope_zero_for_silence():
    sr = 1000
    waveform = np.zeros(sr * 2, dtype=np.float32)
    _, rms = compute_rms_envelope(waveform, sr, window_sec=0.5, hop_sec=0.25)
    assert np.allclose(rms, 0.0)


def test_rms_envelope_positive_for_tone():
    sr = 1000
    t = np.arange(sr * 2) / sr
    waveform = 0.5 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
    _, rms = compute_rms_envelope(waveform, sr, window_sec=0.5, hop_sec=0.25)
    assert np.all(rms > 0.1)


def test_rms_envelope_distinguishes_silence_from_tone():
    sr = 1000
    silence = np.zeros(sr, dtype=np.float32)
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    waveform = np.concatenate([tone, silence])

    timestamps, rms = compute_rms_envelope(waveform, sr, window_sec=0.2, hop_sec=0.1)
    tone_region_rms = rms[timestamps < 0.8]
    silence_region_rms = rms[timestamps > 1.2]

    assert tone_region_rms.mean() > silence_region_rms.mean()
