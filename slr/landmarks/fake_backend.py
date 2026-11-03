"""Deterministic stand-ins for MediaPipe, used only by tests and the smoke test.

They derive "landmarks" from image pixels, so images that look different get different
landmarks and a model can learn from them, without downloading any MediaPipe model. This lets
the full pipeline (extraction -> training -> export -> report) be tested on synthetic data.
"""

import cv2
import numpy as np


def _pixel_points(rgb, n_points, channels=3):
    small = cv2.resize(rgb, (7, 3), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vals = small.reshape(-1, 3)[:n_points]
    return vals[:, :channels]


class FakeHands:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, rgb):
        if rgb.mean() < 8:  # an all-black image has "no hand", like MediaPipe would report
            return []
        pts = _pixel_points(rgb, 21)
        pts[:, 2] -= 0.5
        return [(pts, "Right", 0.99)]

    def close(self):
        pass


class FakePose:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, rgb):
        pose = np.zeros((33, 4), np.float32)
        pose[:21, :3] = _pixel_points(rgb, 21)
        pose[11], pose[12] = [0.35, 0.5, 0, 1], [0.65, 0.5, 0, 1]   # shoulders, for normalization
        pose[:, 3] = 1.0
        return pose

    def close(self):
        pass


class FakeYoloHandDetector:
    """Returns one box over the central region of any non-black frame."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, rgb):
        from slr.landmarks.hand_detector import HandBox

        if rgb.mean() < 8:
            return []
        h, w = rgb.shape[:2]
        kps = np.zeros((21, 3), np.float32)
        kps[:, 0] = np.linspace(w * 0.3, w * 0.7, 21)
        kps[:, 1] = np.linspace(h * 0.3, h * 0.7, 21)
        kps[:, 2] = 1.0
        return [HandBox(np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75], np.float32), 0.9, kps)]
