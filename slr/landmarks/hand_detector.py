"""YOLO hand detector: finds hand boxes (and YOLO's own 21 keypoints) in an RGB frame."""

from dataclasses import dataclass

import numpy as np


@dataclass
class HandBox:
    xyxy: np.ndarray        # (4,) pixel box
    conf: float
    keypoints: np.ndarray   # (21, 3) pixel x, y, visibility from YOLO-pose (zeros if unavailable)


class YoloHandDetector:
    """Wraps an Ultralytics YOLO model trained on hands (see slr.train_hand_detector)."""

    def __init__(self, weights, conf: float = 0.3, max_hands: int = 2, imgsz: int = 640, device=None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.max_hands, self.imgsz, self.device = conf, max_hands, imgsz, device

    def __call__(self, rgb: np.ndarray):
        # Ultralytics treats numpy input as BGR, so flip channels.
        res = self.model.predict(rgb[..., ::-1], conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 max_det=self.max_hands, iou=0.5, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        if getattr(res, "keypoints", None) is not None and res.keypoints.data.shape[1] == 21:
            kps = res.keypoints.data.cpu().numpy()
        else:
            kps = np.zeros((len(boxes), 21, 3), np.float32)
        order = np.argsort(-confs)[: self.max_hands]
        return [HandBox(boxes[i], float(confs[i]), kps[i]) for i in order]


def square_crop_box(xyxy, img_w, img_h, scale: float = 1.6):
    """Square box around a detection, enlarged by `scale` and clipped to the image.

    MediaPipe's landmark model expects some context around the hand, so a tight YOLO box is
    enlarged before cropping.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    nx1, ny1 = max(0, int(round(cx - side / 2))), max(0, int(round(cy - side / 2)))
    nx2, ny2 = min(img_w, int(round(cx + side / 2))), min(img_h, int(round(cy + side / 2)))
    return nx1, ny1, nx2, ny2
