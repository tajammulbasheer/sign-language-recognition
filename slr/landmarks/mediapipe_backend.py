"""MediaPipe Tasks wrappers (HandLandmarker, PoseLandmarker) in IMAGE mode.

MediaPipe 0.10.30+ removed the legacy `mp.solutions` API, so only the Tasks API is used.
IMAGE mode is used even for video: frames are processed independently, which keeps results
identical whether a frame comes from a file, a crop or a webcam.
"""

import numpy as np


class MediaPipeHands:
    def __init__(self, model_path, num_hands: int = 2, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray):
        """Returns [(landmarks (21,3) normalized to this image, handedness 'Left'|'Right', score)]."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect(img)
        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            out.append((arr, hd[0].category_name, float(hd[0].score)))
        return out

    def close(self):
        self.landmarker.close()


class MediaPipePose:
    def __init__(self, model_path, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray):
        """Returns (33, 4) array of x, y, z, visibility, or None if no person is found."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect(img)
        if not res.pose_landmarks:
            return None
        return np.array([[p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0]
                         for p in res.pose_landmarks[0]], np.float32)

    def close(self):
        self.landmarker.close()
