"""
Augmentation for landmark sequences.

Temporal (applied when choosing which frames to use):
  speed        play the sign 0.8x-1.25x as fast (people sign at different speeds)
  crop         start/end a little early or late (annotations are imprecise)
  frame drop   randomly drop frames before resampling (simulates detector hiccups / low fps)

Spatial (applied to the (T, J, 6) features):
  mirror       left-handed signer (swap hands, flip x), probability `p_mirror`
  rotate       small in-plane rotation of hand shapes and body positions
  scale/shift  signer closer to / further from the camera
  noise        landmark jitter
  hand dropout randomly blank one hand for a few frames (missed detection)
"""

import numpy as np

from slr.landmarks.features import mirror


def sample_frame_indices(n: int, T: int, train: bool, rng: np.random.Generator,
                         speed=(0.8, 1.25), crop=0.1, drop=0.1) -> np.ndarray:
    """Pick T frame indices from a clip of n frames.

    Eval: T evenly spaced indices over the whole clip (deterministic).
    Train: random crop, random speed and random frame drop, then evenly spaced resampling.
    """
    if not train or n <= 1:
        return np.linspace(0, max(n - 1, 0), T).round().astype(int)
    idx = np.arange(n)
    if drop > 0 and n > 4:
        keep = rng.random(n) > drop
        keep[[0, -1]] = True
        idx = idx[keep]
    m = len(idx)
    # Crop: keep a random window covering (1 - crop) .. 100% of the clip.
    win = max(2, int(round(m * rng.uniform(1 - crop, 1.0))))
    start = rng.integers(0, m - win + 1)
    idx = idx[start:start + win]
    # Speed: stretch/compress the window, then sample T points across it.
    factor = rng.uniform(*speed)
    span = min(len(idx), max(2, int(round(len(idx) / factor))))
    offset = rng.integers(0, len(idx) - span + 1)
    pos = np.linspace(offset, offset + span - 1, T)
    return idx[np.clip(pos.round().astype(int), 0, len(idx) - 1)]


def spatial_augment(f: np.ndarray, rng: np.random.Generator, p_mirror=0.5, max_rot_deg=15,
                    scale=(0.85, 1.15), shift=0.1, noise=0.01, p_hand_drop=0.1) -> np.ndarray:
    f = f.copy()
    if rng.random() < p_mirror:
        f = mirror(f)
    theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], np.float32)
    present = f[..., 5:6] > 0
    for a, b in ((0, 2), (3, 5)):                          # local xy, global xy
        f[..., a:b] = f[..., a:b] @ rot.T
    f[..., 3:5] = f[..., 3:5] * rng.uniform(*scale) + rng.uniform(-shift, shift, 2)
    f[..., :5] += rng.normal(0, noise, f[..., :5].shape).astype(np.float32)
    f[..., :5] *= present                                   # missing joints stay exactly zero
    if rng.random() < p_hand_drop and f.shape[0] > 4:
        slot = rng.integers(0, 2)
        T = f.shape[0]
        length = rng.integers(1, max(2, T // 4))
        t0 = rng.integers(0, T - length + 1)
        f[t0:t0 + length, slot * 21:(slot + 1) * 21] = 0
    return f
