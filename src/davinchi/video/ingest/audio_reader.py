"""Audio extraction via ffmpeg (shelling out — no extra audio libraries needed).
Keeps the dependency footprint small since this is meant to be simple to run."""
import subprocess
import numpy as np


def extract_audio_waveform(video_path, sample_rate=16000):
    """
    Extract mono PCM audio from a video file using ffmpeg, return as a
    float32 numpy array in [-1.0, 1.0] plus the sample rate used.

    Requires ffmpeg to be installed and on PATH.
    """
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "s16le",       # raw signed 16-bit PCM
        "-acodec", "pcm_s16le",
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-loglevel", "error",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed for {video_path}: "
            f"{result.stderr.decode(errors='replace')}"
        )

    raw = np.frombuffer(result.stdout, dtype=np.int16)
    waveform = raw.astype(np.float32) / 32768.0
    return waveform, sample_rate
