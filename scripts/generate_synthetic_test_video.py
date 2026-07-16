"""
Generates a short synthetic test video with known ground truth, so the
detection pipeline can be developed and sanity-checked before real footage
exists. Structure (30s total, 640x480, 30fps):

  Video: three solid-color 10s segments (red / green / blue) standing in for
         three distinct "painting stages", with hard cuts at 10s and 20s.
  Audio: alternating tone / silence, so silence detection has known spans:
         0-4s tone, 4-8s SILENCE, 8-16s tone, 16-19s SILENCE, 19-30s tone.

Requires ffmpeg on PATH.
"""
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr.decode(errors="replace"), file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def generate(output_path=None, size="640x480", fps=30):
    output_path = output_path or (OUTPUT_DIR / "synthetic_test.mp4")

    video_only = OUTPUT_DIR / "_video_only.mp4"
    audio_only = OUTPUT_DIR / "_audio_only.wav"

    # --- video: red(10s) + green(10s) + blue(10s), hard cuts between ---
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=red:size={size}:rate={fps}:duration=10",
        "-f", "lavfi", "-i", f"color=c=green:size={size}:rate={fps}:duration=10",
        "-f", "lavfi", "-i", f"color=c=blue:size={size}:rate={fps}:duration=10",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]",
        "-map", "[outv]",
        str(video_only),
    ])

    # --- audio: tone/silence/tone/silence/tone, matching the timeline in the docstring ---
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=11",
        "-filter_complex", "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[outa]",
        "-map", "[outa]",
        str(audio_only),
    ])

    # --- mux video + audio together ---
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_only), "-i", str(audio_only),
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest",
        str(output_path),
    ])

    video_only.unlink(missing_ok=True)
    audio_only.unlink(missing_ok=True)

    print(f"Synthetic test video written to: {output_path}")
    print("Ground truth: cuts at ~10.0s, ~20.0s | silence at ~4-8s, ~15-18s")
    return output_path


if __name__ == "__main__":
    generate()
