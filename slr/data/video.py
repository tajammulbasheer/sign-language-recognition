"""Video decoding helpers (OpenCV)."""

import cv2
import numpy as np


def count_frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frames(path, frame_start: int = 1, frame_end: int = -1, max_frames: int = None):
    """Decode a video to a list of RGB uint8 frames.

    frame_start / frame_end follow WLASL conventions (1-indexed, -1 = last frame). If the video is
    shorter than the annotated range it is assumed to be pre-trimmed, and all frames are used.
    `max_frames` uniformly subsamples long videos to bound memory and extraction time.
    Returns (frames, info) where info records whether the annotation range was applied.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(frames)
    if n == 0:
        raise IOError(f"No frames decoded from {path}")

    start = max(frame_start, 1) - 1
    end = n if frame_end in (-1, None) else frame_end
    trimmed = (start < n) and (end <= n) and (end > start) and (start > 0 or end < n)
    if trimmed:
        frames = frames[start:end]
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames, {"decoded_frames": n, "used_frames": len(frames), "trimmed": bool(trimmed)}


def sample_indices(n: int, k: int) -> np.ndarray:
    """k indices spread evenly over n frames (repeats frames if n < k)."""
    return np.linspace(0, n - 1, k).round().astype(int)


def resize_center_crop(frame: np.ndarray, size: int = 224, short_side: int = 256) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = short_side / min(h, w)
    frame = cv2.resize(frame, (max(size, round(w * scale)), max(size, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return frame[top:top + size, left:left + size]
