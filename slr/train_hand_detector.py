"""
Train the YOLO hand detector on Ultralytics' Hand Keypoints dataset.

    python -m slr.train_hand_detector --data-dir ~/data/slr --out ~/runs/slr/hand_detector

Dataset: 26,768 images (18,776 train / 7,992 val) with one "hand" box and 21 keypoints per
hand, downloaded automatically from Ultralytics' GitHub (~300 MB) into <data-dir>/hand-keypoints.
Images are CC BY-NC-SA 4.0, so the trained weights are for non-commercial use.

A YOLO *pose* model is trained, so each detection gives both a hand box (used to crop for
MediaPipe) and YOLO's own 21 keypoints (used by the `yolo-pose` ablation backend).

Outputs (in --out):
  weights/best.pt, weights/last.pt   Ultralytics checkpoints
  results.csv                        per-epoch metrics written by Ultralytics
  detector_results.json             final box/keypoint mAP on the val split (read by the report)

Resumable: if weights/last.pt exists and training didn't finish, it resumes from there;
if it finished, only the final metrics are (re)computed.
"""

import argparse
import shutil
from pathlib import Path

import yaml

from slr.common import read_json, warn_if_low_gpu_memory, write_json


def ensure_hand_dataset(data_dir: Path, url: str) -> Path:
    """Download + unzip the hand dataset into <data-dir>/hand-keypoints if it isn't there.

    Done here rather than by Ultralytics, which always unpacks into its own global
    datasets folder and ignores the `path` in the YAML.
    """
    import zipfile

    from slr.data.download import download_url

    root = data_dir / "hand-keypoints"
    if (root / "images" / "train").exists() and (root / "images" / "val").exists():
        return root
    zip_path = download_url(url, data_dir / "hand-keypoints.zip")
    print(f"  unzipping {zip_path.name}")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    nested = root / "hand-keypoints"  # some archive versions wrap everything in a folder
    if not (root / "images").exists() and (nested / "images").exists():
        for child in nested.iterdir():
            shutil.move(str(child), root / child.name)
        nested.rmdir()
    if not (root / "images" / "train").exists():
        raise FileNotFoundError(f"Unexpected archive layout: {root / 'images' / 'train'} not found")
    zip_path.unlink()
    return root


def dataset_yaml(data_dir: Path) -> Path:
    """Copy Ultralytics' hand-keypoints.yaml, pointing `path` at our data directory."""
    import ultralytics

    src = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "hand-keypoints.yaml"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; upgrade ultralytics (pip install -U ultralytics)")
    cfg = yaml.safe_load(src.read_text())
    cfg["path"] = str(ensure_hand_dataset(data_dir, cfg["download"]).resolve())
    dest = data_dir / "hand-keypoints.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


def final_metrics(out: Path, weights: Path, data_yaml: Path, args) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch,
                               device=args.device, project=str(out), name="val",
                               exist_ok=True, plots=False, verbose=False)
    res = {
        "weights": str(weights),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "box_map50": float(m.box.map50),
        "box_map50_95": float(m.box.map),
        "pose_map50": float(m.pose.map50),
        "pose_map50_95": float(m.pose.map),
    }
    write_json(out / "detector_results.json", res)
    return res


def run(args):
    from ultralytics import YOLO

    data_dir = Path(args.data_dir).expanduser()
    out = Path(args.out).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_yaml(data_dir)
    best, last = out / "weights" / "best.pt", out / "weights" / "last.pt"

    done = read_json(out / "detector_results.json")
    if done and best.exists():
        print(f"Already trained: {best} (box mAP50 {done['box_map50']:.3f}). Delete {out} to retrain.")
        return
    warn_if_low_gpu_memory()

    common = dict(project=str(out.parent), name=out.name, exist_ok=True, device=args.device,
                  workers=args.workers, seed=args.seed, deterministic=True, plots=True)
    if last.exists() and not training_finished(last):
        print(f"Resuming interrupted training from {last}")
        YOLO(str(last)).train(resume=True)
    elif last.exists():
        print(f"Training already finished ({last}); computing final metrics only")
    else:
        YOLO(args.model).train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                               batch=args.batch, patience=args.patience, fraction=args.fraction,
                               val=not args.no_val, **common)
    if not best.exists() and last.exists():
        shutil.copy(last, best)  # val=False runs only write last.pt
    if args.no_val:
        print(f"Trained without validation; weights at {best}")
        return
    res = final_metrics(out, best, data_yaml, args)
    print(f"Hand detector: box mAP50 {res['box_map50']:.3f}, keypoint mAP50 {res['pose_map50']:.3f}")


def training_finished(ckpt_path: Path) -> bool:
    """Ultralytics marks a finished checkpoint with epoch == -1 (optimizer stripped)."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) == -1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="yolo11n-pose.pt", help="Pretrained starting weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0, help="Train on a fraction of the images")
    p.add_argument("--device", default=None, help="e.g. 0 or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-val", action="store_true", help="Skip validation (smoke tests)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
