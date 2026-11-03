"""
Per-frame landmark extraction with three interchangeable backends:

  mediapipe        MediaPipe HandLandmarker on the full frame (the usual approach)
  yolo+mediapipe   YOLO finds hand boxes -> MediaPipe runs on each enlarged crop -> landmarks
                   are mapped back to full-frame coordinates. Falls back to the full frame when
                   YOLO finds no hand. This is the project's main pipeline: small, distant hands
                   (common in WLASL's full-body videos) become large in the crop, so MediaPipe
                   finds them more often.
  yolo-pose        YOLO-pose keypoints only (2D, no MediaPipe), as an ablation

Coordinates are returned in units of image *height* for both axes (x = px / H, y = py / H), so
distances are isotropic regardless of aspect ratio.
"""

from dataclasses import dataclass, field

import numpy as np

from slr.landmarks.hand_detector import square_crop_box

BACKENDS = ("mediapipe", "yolo+mediapipe", "yolo-pose")
SLOT_OF = {"Right": 0, "Left": 1}   # MediaPipe handedness label -> slot


@dataclass
class FrameLandmarks:
    hands: np.ndarray          # (2, 21, 3)
    hand_present: np.ndarray   # (2,) float 0/1
    hand_score: np.ndarray     # (2,)
    pose: np.ndarray           # (33, 4): x, y, z, visibility
    pose_present: float


@dataclass
class ExtractionStats:
    frames: int = 0
    frames_with_hand: int = 0
    hands_found: int = 0
    yolo_boxes: int = 0
    crops_without_landmarks: int = 0
    full_frame_fallbacks: int = 0
    frames_with_pose: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["hand_detection_rate"] = self.frames_with_hand / self.frames if self.frames else 0.0
        return d


def _assign_slots(hands):
    """hands: [(lm (21,3), label, score)] -> up to 2 hands placed in Right/Left slots.

    If two hands claim the same label, the more confident one keeps it.
    """
    hands = sorted(hands, key=lambda h: -h[2])[:2]
    slots = [None, None]
    for lm, label, score in hands:
        s = SLOT_OF.get(label, 0)
        if slots[s] is not None:
            s = 1 - s
        if slots[s] is None:
            slots[s] = (lm, score)
    return slots


def _dedupe(hands, min_dist=0.03):
    """Drop a hand whose wrist nearly coincides with a more confident one (overlapping crops)."""
    kept = []
    for h in sorted(hands, key=lambda h: -h[2]):
        if all(np.linalg.norm(h[0][0, :2] - k[0][0, :2]) > min_dist for k in kept):
            kept.append(h)
    return kept


class LandmarkExtractor:
    def __init__(self, backend="yolo+mediapipe", hand_model=None, pose_model=None,
                 yolo_weights=None, use_pose=True, yolo_conf=0.3, crop_scale=1.6,
                 fake=False, device=None):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend, self.use_pose, self.crop_scale = backend, use_pose, crop_scale
        self.stats = ExtractionStats()

        if fake:
            from slr.landmarks.fake_backend import FakeHands, FakePose, FakeYoloHandDetector
            hands_cls, pose_cls, yolo_cls = FakeHands, FakePose, FakeYoloHandDetector
        else:
            from slr.landmarks.hand_detector import YoloHandDetector
            from slr.landmarks.mediapipe_backend import MediaPipeHands, MediaPipePose
            hands_cls, pose_cls, yolo_cls = MediaPipeHands, MediaPipePose, YoloHandDetector

        self.yolo = None
        if backend in ("yolo+mediapipe", "yolo-pose"):
            if not fake and not yolo_weights:
                raise ValueError(f"backend '{backend}' needs --yolo-weights")
            self.yolo = yolo_cls(yolo_weights, conf=yolo_conf, device=device)
        self.hands_full = hands_cls(hand_model, num_hands=2) if backend != "yolo-pose" else None
        self.hands_crop = hands_cls(hand_model, num_hands=1) if backend == "yolo+mediapipe" else None
        self.pose = pose_cls(pose_model) if use_pose else None

    # -- backends ---------------------------------------------------------------------------

    def _full_frame(self, rgb, w, h):
        return [(lm * np.array([w / h, 1.0, w / h], np.float32), label, score)
                for lm, label, score in self.hands_full(rgb)]

    def _yolo_mediapipe(self, rgb, w, h):
        boxes = self.yolo(rgb)
        self.stats.yolo_boxes += len(boxes)
        if not boxes:
            self.stats.full_frame_fallbacks += 1
            return self._full_frame(rgb, w, h)
        found = []
        for b in boxes:
            x1, y1, x2, y2 = square_crop_box(b.xyxy, w, h, self.crop_scale)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            res = self.hands_crop(rgb[y1:y2, x1:x2])
            if not res:
                self.stats.crops_without_landmarks += 1
                continue
            lm, label, score = res[0]
            cw, ch = x2 - x1, y2 - y1
            full = np.empty_like(lm)
            full[:, 0] = (x1 + lm[:, 0] * cw) / h
            full[:, 1] = (y1 + lm[:, 1] * ch) / h
            full[:, 2] = lm[:, 2] * cw / h
            found.append((full, label, score))
        return _dedupe(found)

    def _yolo_pose(self, rgb, w, h):
        boxes = self.yolo(rgb)
        self.stats.yolo_boxes += len(boxes)
        found = []
        for b in boxes:
            kp = b.keypoints
            lm = np.stack([kp[:, 0] / h, kp[:, 1] / h, np.zeros(21, np.float32)], 1).astype(np.float32)
            found.append((lm, None, b.conf))
        # YOLO gives no handedness: the hand further right in the image (signer's right hand
        # in a non-mirrored video) goes to the "Right" slot.
        found.sort(key=lambda f: -f[0][0, 0])
        return [(lm, ("Right", "Left")[i], s) for i, (lm, _, s) in enumerate(found[:2])]

    # -- public -----------------------------------------------------------------------------

    def __call__(self, rgb: np.ndarray) -> FrameLandmarks:
        h, w = rgb.shape[:2]
        if self.backend == "mediapipe":
            hands = self._full_frame(rgb, w, h)
        elif self.backend == "yolo+mediapipe":
            hands = self._yolo_mediapipe(rgb, w, h)
        else:
            hands = self._yolo_pose(rgb, w, h)

        out_h = np.zeros((2, 21, 3), np.float32)
        present = np.zeros(2, np.float32)
        score = np.zeros(2, np.float32)
        for s, item in enumerate(_assign_slots(hands)):
            if item is not None:
                out_h[s], present[s], score[s] = item[0], 1.0, item[1]

        pose = np.zeros((33, 4), np.float32)
        pose_present = 0.0
        if self.pose is not None:
            p = self.pose(rgb)
            if p is not None:
                pose = p.copy()
                pose[:, 0] *= w / h
                pose[:, 2] *= w / h
                pose_present = 1.0
                self.stats.frames_with_pose += 1

        self.stats.frames += 1
        self.stats.hands_found += int(present.sum())
        self.stats.frames_with_hand += int(present.any())
        return FrameLandmarks(out_h, present, score, pose, pose_present)

    def close(self):
        for m in (self.hands_full, self.hands_crop, self.pose):
            if m is not None and hasattr(m, "close"):
                m.close()


def stack_frames(frames):
    """List[FrameLandmarks] -> dict of stacked arrays (what gets cached to .npz)."""
    return {
        "hands": np.stack([f.hands for f in frames]).astype(np.float32),
        "hand_present": np.stack([f.hand_present for f in frames]).astype(np.float32),
        "hand_score": np.stack([f.hand_score for f in frames]).astype(np.float32),
        "pose": np.stack([f.pose for f in frames]).astype(np.float32),
        "pose_present": np.array([f.pose_present for f in frames], np.float32),
    }
