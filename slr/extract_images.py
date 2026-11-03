"""
Extract hand landmarks from the ASL alphabet images (static signs) and cache them.

    python -m slr.extract_images --asl-dir ~/data/slr/asl_alphabet \
        --ood-dir ~/data/slr/asl_alphabet_test --models-dir ~/data/slr/models \
        --yolo-weights ~/runs/slr/hand_detector/weights/best.pt \
        --out ~/runs/slr/landmarks/asl_alphabet

Writes, per backend, <out>/<backend>.npz with arrays for every split (train/val/test/ood):
  <split>_features (N, 42, 6), <split>_labels (N,), <split>_paths (N,)
and <out>/manifest.json (classes + split lists) and <out>/<backend>_stats.json
(hand-detection rate per split and per class). Re-running skips backends already done.

By default it extracts with both `mediapipe` (full image) and `yolo+mediapipe` (YOLO crop),
which gives the detection-rate comparison in the README.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from slr.common import read_json, write_json
from slr.data.asl_alphabet import build_manifest
from slr.landmarks.extractor import LandmarkExtractor
from slr.landmarks.features import build_features


def extract_split(extractor, items, desc):
    feats, labels, paths, found = [], [], [], []
    for path, label in tqdm(items, desc=desc, leave=False):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        fl = extractor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        f = build_features(fl.hands[None], fl.hand_present[None], use_pose=False, use_global=False)[0]
        feats.append(f)
        labels.append(label)
        paths.append(path)
        found.append(bool(fl.hand_present.any()))
    return (np.stack(feats).astype(np.float32) if feats else np.zeros((0, 42, 6), np.float32),
            np.array(labels, np.int64), np.array(paths), np.array(found))


def per_class_rate(labels, found, classes):
    return {classes[c]: float(found[labels == c].mean()) for c in range(len(classes))
            if (labels == c).any()}


def run(args):
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest is None:
        manifest = build_manifest(args.asl_dir, args.ood_dir, args.max_per_class, args.seed)
        write_json(manifest_path, manifest)
    classes = manifest["classes"]
    print({k: len(v) for k, v in manifest["splits"].items()}, "images per split")

    models = Path(args.models_dir).expanduser() if args.models_dir else None
    for backend in args.backends:
        npz_path = out / f"{backend.replace('+', '_')}.npz"
        if npz_path.exists():
            print(f"{backend}: already extracted ({npz_path}), skipping")
            continue
        extractor = LandmarkExtractor(
            backend=backend,
            hand_model=models / "hand_landmarker.task" if models else None,
            yolo_weights=args.yolo_weights, use_pose=False, fake=args.fake)
        arrays, stats = {}, {"backend": backend, "splits": {}}
        for split, items in manifest["splits"].items():
            f, y, p, found = extract_split(extractor, items, f"{backend}/{split}")
            arrays.update({f"{split}_features": f, f"{split}_labels": y,
                           f"{split}_paths": p, f"{split}_found": found})
            has_hand = y != classes.index("nothing")      # "nothing" images contain no hand by design
            stats["splits"][split] = {
                "images": int(len(y)),
                "hand_detection_rate": float(found[has_hand].mean()) if has_hand.any() else 0.0,
                "hand_detection_rate_all_images": float(found.mean()) if len(found) else 0.0,
                "per_class": per_class_rate(y, found, classes),
            }
            print(f"  {backend}/{split}: hand found in {stats['splits'][split]['hand_detection_rate']:.1%} "
                  f"of {int(has_hand.sum())} images that contain a hand")
        stats["extractor"] = extractor.stats.as_dict()
        extractor.close()
        np.savez_compressed(npz_path, classes=np.array(classes), **arrays)
        write_json(out / f"{backend.replace('+', '_')}_stats.json", stats)
        print(f"{backend}: saved {npz_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asl-dir", required=True)
    p.add_argument("--ood-dir", default=None, help="ASL Alphabet Test folder (out-of-distribution)")
    p.add_argument("--models-dir", default=None, help="Folder with hand_landmarker.task")
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--backends", nargs="+", default=["mediapipe", "yolo+mediapipe"])
    p.add_argument("--max-per-class", type=int, default=600,
                   help="Images per class sampled from the 3,000 available (keeps extraction fast)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake", action="store_true", help="Use fake landmark models (tests only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
