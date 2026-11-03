"""
WLASL (word-level ASL video): parse the official annotation file and build a manifest.

Uses WLASL_v0.3.json from the dataset itself, so the official train/val/test split and signer
IDs are preserved. WLASL100 is the 100 most frequent glosses, i.e. the first 100 entries of the
file (as defined by the dataset authors).

Annotation conventions (from the WLASL README): frames are 1-indexed at 25 fps, and
frame_end == -1 means "until the end of the video". Some Kaggle copies ship videos that are
already trimmed to the sign, so trimming is only applied when the video is long enough to
contain the annotated range (see `slr.data.video.read_frames`).
"""

import json
from collections import Counter
from pathlib import Path

from slr.common import VIDEO_EXTS


def find_annotation_file(root) -> Path:
    root = Path(root).expanduser()
    hits = sorted(root.rglob("WLASL_v*.json"))
    if not hits:
        raise FileNotFoundError(f"No WLASL_v*.json found under {root}")
    return hits[-1]  # highest version


def index_videos(root) -> dict:
    """{video_id: path} for every video file under `root`."""
    return {p.stem: p for p in Path(root).expanduser().rglob("*")
            if p.suffix.lower() in VIDEO_EXTS}


def build_manifest(root, num_classes: int = 100):
    """Manifest for WLASL-<num_classes>.

    Returns {"classes": [gloss, ...], "splits": {split: [entry]}, "stats": {...}}, where each
    entry is {"video_id", "path", "label", "signer_id", "frame_start", "frame_end"}.
    Instances whose video file is missing are skipped and counted (WLASL is built from web
    links, and every public copy is missing some videos).
    """
    ann_path = find_annotation_file(root)
    glosses = json.loads(ann_path.read_text())[:num_classes]
    videos = index_videos(root)
    classes = [g["gloss"] for g in glosses]
    splits = {"train": [], "val": [], "test": []}
    missing = Counter()
    for label, g in enumerate(glosses):
        for inst in g["instances"]:
            split = inst.get("split")
            if split not in splits:
                continue
            path = videos.get(str(inst["video_id"]))
            if path is None:
                missing[split] += 1
                continue
            splits[split].append({
                "video_id": str(inst["video_id"]),
                "path": str(path),
                "label": label,
                "signer_id": inst.get("signer_id"),
                "frame_start": int(inst.get("frame_start", 1)),
                "frame_end": int(inst.get("frame_end", -1)),
            })
    stats = {
        "annotation_file": str(ann_path),
        "num_classes": len(classes),
        "available": {k: len(v) for k, v in splits.items()},
        "missing_videos": dict(missing),
        "classes_without_train_videos": [classes[i] for i in range(len(classes))
                                         if not any(e["label"] == i for e in splits["train"])],
    }
    return {"classes": classes, "splits": splits, "stats": stats}
