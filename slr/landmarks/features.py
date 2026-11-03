"""
Turn raw landmark arrays into model features.

Every model in the project consumes the same per-frame layout, a (J, 6) array of joints:

    channels: [local_x, local_y, local_z, global_x, global_y, presence]
    joints:   0-20  right-hand slot (MediaPipe's 21 landmarks)
              21-41 left-hand slot
              42-48 upper body (nose, shoulders, elbows, wrists), only if pose is used

local_*   hand shape: relative to that hand's wrist, divided by the wrist->middle-knuckle
          length, so it doesn't depend on where the hand is or how big it appears.
global_*  where the joint is relative to the body: relative to the shoulder midpoint, divided by
          shoulder width (falls back to the image centre if no pose was found). Zero for static
          images, where position in the photo is meaningless.
presence  1 if the hand / body joint was detected in that frame, else 0 (all channels zero).

RNNs flatten this to J * 6 values per frame; the TimeSformer-style model treats each joint as
a token.
"""

import numpy as np

UPPER_BODY = [0, 11, 12, 13, 14, 15, 16]       # MediaPipe pose indices
POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5]            # left/right swap within UPPER_BODY
NUM_HAND_JOINTS = 42
C = 6


def num_joints(use_pose: bool) -> int:
    return NUM_HAND_JOINTS + (len(UPPER_BODY) if use_pose else 0)


def interpolate_gaps(hands, present, max_gap: int = 5):
    """Linearly fill short runs of missing hand frames (hand briefly lost by the detector).

    hands (T, 2, 21, 3), present (T, 2) -> filled copies. Gaps longer than `max_gap` frames, and
    gaps at the start or end of the clip, are left empty.
    """
    hands, present = hands.copy(), present.copy()
    T = len(present)
    for s in range(2):
        t = 0
        while t < T:
            if present[t, s]:
                t += 1
                continue
            start = t
            while t < T and not present[t, s]:
                t += 1
            end = t  # first present frame after the gap (or T)
            if start == 0 or end == T or end - start > max_gap:
                continue
            a, b = hands[start - 1, s], hands[end, s]
            for i, k in enumerate(range(start, end), 1):
                w = i / (end - start + 1)
                hands[k, s] = (1 - w) * a + w * b
                present[k, s] = 1.0
    return hands, present


def body_frame(pose, pose_present, aspect_center=(0.5, 0.5)):
    """Per-frame origin (T, 2) and scale (T,) from the shoulders, with a fallback."""
    T = len(pose_present)
    origin = np.tile(np.array(aspect_center, np.float32), (T, 1))
    scale = np.full(T, 0.35, np.float32)
    ok = pose_present > 0
    if ok.any():
        ls, rs = pose[:, 11, :2], pose[:, 12, :2]
        width = np.linalg.norm(ls - rs, axis=1)
        good = ok & (width > 1e-3)
        origin[good] = ((ls + rs) / 2)[good]
        scale[good] = width[good]
    return origin, scale


def build_features(hands, hand_present, pose=None, pose_present=None,
                   use_pose=True, use_global=True, image_center=(0.5, 0.5)):
    """Raw arrays for T frames -> (T, J, 6) float32 features. See module docstring."""
    hands = np.asarray(hands, np.float32)
    hand_present = np.asarray(hand_present, np.float32)
    T = hands.shape[0]
    J = num_joints(use_pose)
    out = np.zeros((T, J, C), np.float32)

    if pose is None:
        pose = np.zeros((T, 33, 4), np.float32)
        pose_present = np.zeros(T, np.float32)
    origin, scale = body_frame(pose, pose_present, image_center)

    for s in range(2):
        lm = hands[:, s]                                   # (T, 21, 3)
        wrist = lm[:, :1]
        size = np.linalg.norm(lm[:, 9, :2] - lm[:, 0, :2], axis=1)
        size = np.where(size > 1e-4, size, 1.0)[:, None, None]
        local = (lm - wrist) / size
        mask = hand_present[:, s][:, None, None]
        sl = slice(s * 21, (s + 1) * 21)
        out[:, sl, 0:3] = local * mask
        if use_global:
            out[:, sl, 3:5] = (lm[..., :2] - origin[:, None]) / scale[:, None, None] * mask
        out[:, sl, 5] = hand_present[:, s][:, None]

    if use_pose:
        vis = pose[:, UPPER_BODY, 3] * pose_present[:, None]
        g = (pose[:, UPPER_BODY, :2] - origin[:, None]) / scale[:, None, None]
        out[:, NUM_HAND_JOINTS:, 3:5] = g * (vis[..., None] > 0)
        out[:, NUM_HAND_JOINTS:, 5] = vis
    return out


def mirror(features):
    """Horizontal flip: swap hand slots and left/right body joints, negate x channels."""
    f = features.copy()
    f[..., :21, :], f[..., 21:42, :] = features[..., 21:42, :], features[..., :21, :]
    if f.shape[-2] > NUM_HAND_JOINTS:
        f[..., NUM_HAND_JOINTS:, :] = features[..., [NUM_HAND_JOINTS + i for i in POSE_MIRROR], :]
    f[..., 0] *= -1
    f[..., 3] *= -1
    return f
