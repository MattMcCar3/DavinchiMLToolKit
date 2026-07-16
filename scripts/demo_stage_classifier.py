"""
Demonstrates the StageClassifier end-to-end using the synthetic test video's
three color segments as stand-ins for real painting stages. This is a
plumbing demo, not an accuracy claim — real stage classification needs
real labeled reference frames from actual painting footage.
"""
from pathlib import Path
from davinchi.video.ingest.video_reader import read_frames
from davinchi.video.classification.stage_classifier import StageClassifier

VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "synthetic_test.mp4"


def grab_frame_at(video_path, target_ts):
    for ts, frame in read_frames(video_path):
        if ts >= target_ts:
            return frame
    raise RuntimeError(f"Could not find frame at {target_ts}s")


def main():
    # one reference frame per "stage" (red/green/blue segments)
    labeled_frames = [
        (grab_frame_at(VIDEO_PATH, 2.0), "stage_a_red"),
        (grab_frame_at(VIDEO_PATH, 12.0), "stage_b_green"),
        (grab_frame_at(VIDEO_PATH, 22.0), "stage_c_blue"),
    ]

    clf = StageClassifier(k=1).fit(labeled_frames)

    # classify a few unseen frames from within each segment
    test_points = [5.0, 15.0, 25.0]
    for ts in test_points:
        frame = grab_frame_at(VIDEO_PATH, ts)
        pred = clf.predict(frame)
        print(f"  t={ts:>5.1f}s -> predicted stage: {pred}")


if __name__ == "__main__":
    main()
