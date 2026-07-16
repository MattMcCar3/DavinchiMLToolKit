"""
End-to-end pipeline: run cut detection + silence detection on a video,
build a unified timeline, and export it as JSON. Stage classification is
included but requires labeled reference frames, so it's demoed separately
(see scripts/demo_stage_classifier.py) rather than run blind here.

Usage:
    python -m scripts.process_footage path/to/clip.mp4 [--out timeline.json]
"""
import argparse
from pathlib import Path

from davinchi.video.ingest.video_reader import get_video_info
from davinchi.video.detectors.cut_detector import detect_cuts
from davinchi.video.detectors.silence_detector import detect_silence
from davinchi.video.timeline.segments import build_timeline
from davinchi.video.timeline.export import export_timeline_json


def process(video_path, out_path=None, cut_threshold=0.5, silence_rms_threshold=0.02):
    video_path = Path(video_path)
    info = get_video_info(video_path)

    print(f"Processing {video_path.name} ({info['duration_sec']:.1f}s, {info['fps']:.1f}fps)")

    print("Detecting cuts...")
    cuts = detect_cuts(video_path, threshold=cut_threshold)
    print(f"  Found {len(cuts)} cut(s): {[round(c, 2) for c in cuts]}")

    print("Detecting silence...")
    silence_spans = detect_silence(video_path, rms_threshold=silence_rms_threshold)
    print(f"  Found {len(silence_spans)} silent span(s): "
          f"{[(round(s, 2), round(e, 2)) for s, e in silence_spans]}")

    timeline = build_timeline(
        duration_sec=info["duration_sec"],
        cuts=cuts,
        silence_spans=silence_spans,
        stage_labels=None,  # requires a fitted StageClassifier — see demo script
    )

    out_path = out_path or video_path.with_suffix(".timeline.json")
    export_timeline_json(timeline, out_path)
    print(f"Timeline written to: {out_path}")
    return timeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cut-threshold", type=float, default=0.5)
    parser.add_argument("--silence-threshold", type=float, default=0.02)
    args = parser.parse_args()

    process(args.video_path, args.out, args.cut_threshold, args.silence_threshold)
