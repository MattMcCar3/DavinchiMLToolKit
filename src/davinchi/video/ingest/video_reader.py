"""Frame reading via OpenCV. Kept deliberately simple: sample frames at a
fixed stride rather than decoding every single frame, since cut/stage
detection doesn't need every frame to work."""
import cv2


def get_video_info(video_path):
    """Return basic info about a video file: fps, frame_count, width, height, duration_sec."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps else 0.0,
    }


def read_frames(video_path, sample_every_n=1):
    """
    Generator yielding (timestamp_sec, frame_bgr) for sampled frames.

    Parameters
    ----------
    sample_every_n : int
        1 = every frame. Higher values skip frames for speed — fine for
        cut/stage detection since we're looking for changes over time,
        not single-frame precision.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            timestamp = frame_idx / fps
            yield timestamp, frame

        frame_idx += 1

    cap.release()
