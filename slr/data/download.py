"""
Download every external asset the pipeline needs, into one data directory.

    python -m slr.data.download --data-dir ~/data/slr

Downloads (each is skipped if already present):
  asl_alphabet/        Kaggle  grassknoted/asl-alphabet        (~1 GB, 29 classes, 87k images)
  asl_alphabet_test/   Kaggle  danrasband/asl-alphabet-test    (870 images, different backgrounds)
  wlasl/               Kaggle  risangbaskoro/wlasl-processed   (~5 GB, videos + WLASL_v0.3.json)
  models/              MediaPipe hand and pose landmarker .task files (Google storage)

The YOLO hand dataset is downloaded by `slr.train_hand_detector` (via Ultralytics, from GitHub).

Kaggle needs an API token: kaggle.com -> Settings -> API -> "Create New Token", then put
kaggle.json in ~/.kaggle/ (chmod 600). If you already downloaded a dataset by hand, point the
scripts at it with the --*-dir options instead; nothing here is mandatory.

Licences: WLASL is for academic, non-commercial use (C-UDA). Datasets are never committed to git.
"""

import argparse
import os
import shutil
import time
import urllib.request
from pathlib import Path

KAGGLE_DATASETS = {
    "asl_alphabet": "grassknoted/asl-alphabet",
    "asl_alphabet_test": "danrasband/asl-alphabet-test",
    "wlasl": "risangbaskoro/wlasl-processed",
}

# Pinned model versions (not "latest") so results are reproducible.
MEDIAPIPE_MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def download_url(url: str, dest: Path, retries: int = 3):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # network errors vary by platform
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}. Download it in a browser and save it as {dest}.")


def ensure_mediapipe_models(models_dir) -> dict:
    models_dir = Path(models_dir)
    return {name: download_url(url, models_dir / name) for name, url in MEDIAPIPE_MODELS.items()}


def kaggle_download(handle: str, dest: Path):
    """Download a Kaggle dataset with kagglehub and link it at `dest`."""
    if dest.exists() and any(dest.iterdir()):
        print(f"  have {dest}")
        return dest
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("kagglehub is not installed: pip install kagglehub") from e
    has_token = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not has_token:
        print("  WARNING: no Kaggle token found at ~/.kaggle/kaggle.json; the download may fail.")
    print(f"  downloading Kaggle dataset {handle} (this can take a while)")
    src = Path(kagglehub.dataset_download(handle))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():          # stale or broken link from an earlier run
        dest.unlink()
    elif dest.exists():            # only an empty directory can reach this point
        dest.rmdir()
    dest.symlink_to(src, target_is_directory=True)  # no second copy of multi-GB data
    print(f"  linked {dest} -> {src}")
    return dest


def run(args):
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = args.only or ["mediapipe", *KAGGLE_DATASETS]
    if "mediapipe" in wanted:
        print("MediaPipe models:")
        ensure_mediapipe_models(data_dir / "models")
    for key, handle in KAGGLE_DATASETS.items():
        if key in wanted:
            print(f"{key}:")
            kaggle_download(handle, data_dir / key)
    print(f"Done. Data directory: {data_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--only", nargs="*", choices=["mediapipe", *KAGGLE_DATASETS],
                   help="Download only these items")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
