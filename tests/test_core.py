"""Fast unit tests (no GPU, no downloads). Run: pytest -q"""

import json

import cv2
import numpy as np
import pytest
import torch

from slr.augment import sample_frame_indices, spatial_augment
from slr.landmarks.extractor import LandmarkExtractor, _assign_slots, stack_frames
from slr.landmarks.features import (NUM_HAND_JOINTS, build_features, interpolate_gaps, mirror,
                                    num_joints)
from slr.models.sequence import MODELS, build_sequence_model, count_params
from slr.models.static import LandmarkMLP


def random_hands(T=10, present=(1, 1)):
    rng = np.random.default_rng(0)
    hands = rng.uniform(0.2, 0.8, (T, 2, 21, 3)).astype(np.float32)
    pres = np.tile(np.array(present, np.float32), (T, 1))
    return hands, pres


# ---------------------------------------------------------------- features

def test_feature_shape_and_wrist_origin():
    hands, pres = random_hands()
    f = build_features(hands, pres, use_pose=False)
    assert f.shape == (10, NUM_HAND_JOINTS, 6)
    assert np.allclose(f[:, 0, :3], 0) and np.allclose(f[:, 21, :3], 0)   # wrists at origin
    assert np.allclose(f[..., 5], 1)


def test_missing_hand_is_all_zero():
    hands, pres = random_hands(present=(1, 0))
    f = build_features(hands, pres, use_pose=False)
    assert np.all(f[:, 21:42] == 0)


def test_pose_joints_added():
    hands, pres = random_hands()
    pose = np.zeros((10, 33, 4), np.float32)
    pose[:, 11, :2], pose[:, 12, :2], pose[..., 3] = (0.4, 0.5), (0.6, 0.5), 1
    f = build_features(hands, pres, pose, np.ones(10, np.float32), use_pose=True)
    assert f.shape[1] == num_joints(True) == 49
    assert np.allclose(f[:, 42 + 1, 3:5], (-0.5, 0))   # left shoulder: half a shoulder-width left


def test_mirror_is_an_involution():
    hands, pres = random_hands()
    f = build_features(hands, pres, use_pose=True)
    assert np.allclose(mirror(mirror(f)), f)


def test_interpolate_short_gap_only():
    hands, pres = random_hands(T=12)
    pres[3:5, 0] = 0           # 2-frame gap -> filled
    pres[7:12, 1] = 0          # gap until the end -> left alone
    h2, p2 = interpolate_gaps(hands, pres, max_gap=5)
    assert p2[3:5, 0].all() and not p2[7:, 1].any()
    assert np.allclose(h2[4, 0], hands[2, 0] + (hands[5, 0] - hands[2, 0]) * 2 / 3, atol=1e-5)


# ---------------------------------------------------------------- augmentation

@pytest.mark.parametrize("n", [1, 5, 40, 300])
def test_frame_indices_in_range(n):
    rng = np.random.default_rng(0)
    for train in (True, False):
        idx = sample_frame_indices(n, 32, train, rng)
        assert len(idx) == 32 and idx.min() >= 0 and idx.max() < n
        assert np.all(np.diff(idx) >= 0)       # time never runs backwards


def test_spatial_augment_keeps_missing_joints_zero():
    hands, pres = random_hands(present=(1, 0))
    f = build_features(hands, pres, use_pose=False)
    g = spatial_augment(f, np.random.default_rng(1), p_mirror=0.0)
    assert np.all(g[:, 21:42] == 0)


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", MODELS)
def test_sequence_models_forward(name):
    m = build_sequence_model(name, 49, 6, 100, num_frames=32)
    out = m(torch.randn(3, 32, 49, 6))
    assert out.shape == (3, 100)
    assert count_params(m) < 3_000_000      # small enough for real-time CPU inference


def test_static_mlp_forward():
    assert LandmarkMLP(42, 6, 29)(torch.randn(4, 42, 6)).shape == (4, 29)


# ---------------------------------------------------------------- extraction

def test_slot_assignment_resolves_duplicate_labels():
    a, b = np.zeros((21, 3)), np.ones((21, 3))
    slots = _assign_slots([(a, "Right", 0.6), (b, "Right", 0.9)])
    assert slots[0][0] is b and slots[1][0] is a      # more confident keeps "Right"


def test_fake_extractor_crop_mapping_stays_in_image():
    img = np.full((120, 200, 3), 128, np.uint8)
    ex = LandmarkExtractor("yolo+mediapipe", fake=True, use_pose=True)
    fl = ex(img)
    assert fl.hand_present.sum() == 1
    xs, ys = fl.hands[0, :, 0] * 120, fl.hands[0, :, 1] * 120   # back to pixels
    assert xs.min() >= 0 and xs.max() <= 200 and ys.min() >= 0 and ys.max() <= 120
    black = ex(np.zeros_like(img))
    assert black.hand_present.sum() == 0
    arrays = stack_frames([fl, black])
    assert arrays["hands"].shape == (2, 2, 21, 3)
    assert ex.stats.as_dict()["hand_detection_rate"] == 0.5


# ---------------------------------------------------------------- data

def _write_video(path, n_frames, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, size)
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), i * 3 % 255, np.uint8))
    w.release()


def test_read_frames_trim_and_pretrimmed(tmp_path):
    from slr.data.video import read_frames

    v = tmp_path / "a.avi"
    _write_video(v, 40)
    frames, info = read_frames(v, 11, 30)
    assert info["trimmed"] and len(frames) == 20
    frames, info = read_frames(v, 50, 90)          # range beyond the video: assume pre-trimmed
    assert not info["trimmed"] and len(frames) == 40


def test_wlasl_manifest(tmp_path):
    from slr.data.wlasl import build_manifest

    (tmp_path / "videos").mkdir()
    ann = [{"gloss": g, "instances": [
        {"video_id": f"{g}{i}", "split": ["train", "val", "test"][i % 3], "frame_start": 1,
         "frame_end": -1, "signer_id": i} for i in range(6)]} for g in ("book", "drink", "extra")]
    (tmp_path / "WLASL_v0.3.json").write_text(json.dumps(ann))
    for g in ("book", "drink"):
        for i in range(5):                         # one video per gloss missing
            _write_video(tmp_path / "videos" / f"{g}{i}.avi", 5)
    m = build_manifest(tmp_path, num_classes=2)
    assert m["classes"] == ["book", "drink"]
    assert sum(m["stats"]["available"].values()) == 10
    assert sum(m["stats"]["missing_videos"].values()) == 2


def test_asl_find_class_root_nested(tmp_path):
    from slr.data.asl_alphabet import ASL_CLASSES, build_manifest

    root = tmp_path / "asl_alphabet_train" / "asl_alphabet_train"
    for c in ASL_CLASSES:
        (root / c).mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(root / c / f"{c}{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
    m = build_manifest(tmp_path, max_per_class=None)
    n = {k: len(v) for k, v in m["splits"].items()}
    assert n == {"train": 29 * 7, "val": 29 * 2, "test": 29 * 1}


# ---------------------------------------------------------------- training / report

def test_trainer_resume_continues(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    from slr.training import Trainer

    ds = TensorDataset(torch.randn(16, 42, 6), torch.randint(0, 3, (16,)))
    dl = DataLoader(ds, batch_size=8)
    t = Trainer(LandmarkMLP(42, 6, 3), tmp_path, torch.device("cpu"), epochs=2, lr=1e-3,
                warmup_epochs=0, steps_per_epoch=len(dl))
    t.fit(dl, dl, 3)
    t2 = Trainer(LandmarkMLP(42, 6, 3), tmp_path, torch.device("cpu"), epochs=4, lr=1e-3,
                 warmup_epochs=0, steps_per_epoch=len(dl))
    assert t2.try_resume() and t2.start_epoch == 2
    t2.fit(dl, dl, 3)
    assert [h["epoch"] for h in t2.history] == [1, 2, 3, 4]


def test_readme_update_is_idempotent(tmp_path):
    from slr.report import END, START, update_readme

    r = tmp_path / "README.md"
    r.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
    update_readme(r, "new \\ table")
    update_readme(r, "new \\ table")
    text = r.read_text()
    assert text.count(START) == 1 and "new \\ table" in text and "old" not in text and "outro" in text
