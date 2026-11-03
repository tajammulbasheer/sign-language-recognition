"""PyTorch datasets: landmark sequences (video), single-frame landmarks and raw images (static)."""

import numpy as np
import torch
from torch.utils.data import Dataset

from slr.augment import sample_frame_indices, spatial_augment
from slr.landmarks.features import build_features


def load_clip_features(cache_path, use_pose=True):
    """Cached per-video landmarks -> (L, J, 6) features for every frame of the clip."""
    with np.load(cache_path) as d:
        aspect = float(d["aspect"]) if "aspect" in d else 1.0
        return build_features(d["hands"], d["hand_present"], d["pose"], d["pose_present"],
                              use_pose=use_pose, use_global=True, image_center=(0.5 * aspect, 0.5))


class SequenceDataset(Dataset):
    """WLASL clips as fixed-length landmark sequences, (T, J, 6) per item.

    All clips are loaded into memory once (a few tens of MB for WLASL100).
    """

    def __init__(self, entries, num_frames=32, use_pose=True, train=False, augment=True, seed=0):
        self.items = [(load_clip_features(e["cache"], use_pose), int(e["label"])) for e in entries]
        self.T, self.train, self.augment = num_frames, train, augment and train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        feats, label = self.items[i]
        if self.augment:
            idx = sample_frame_indices(len(feats), self.T, True, self.rng)
            x = spatial_augment(feats[idx], self.rng)
        else:
            x = feats[sample_frame_indices(len(feats), self.T, False, self.rng)]
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), label


def worker_init(worker_id):
    """Give each DataLoader worker its own augmentation RNG stream."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if hasattr(ds, "rng"):
        ds.rng = np.random.default_rng(info.seed % 2**32)


class StaticLandmarkDataset(Dataset):
    """Single-image landmark features (42, 6) from slr.extract_images."""

    def __init__(self, features, labels, train=False, seed=0):
        self.x, self.y = features.astype(np.float32), labels.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.train:
            x = spatial_augment(x[None], self.rng, p_mirror=0.5, max_rot_deg=15, shift=0.0,
                                noise=0.01, p_hand_drop=0.0)[0]
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])


class ImageDataset(Dataset):
    """Raw images for the CNN baseline."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), int(label)
