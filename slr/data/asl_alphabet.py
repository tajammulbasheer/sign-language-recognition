"""
ASL alphabet (static images): find the class folders and build train/val/test lists.

Kaggle's grassknoted/asl-alphabet is used for train/val/test (a seeded stratified split), and
danrasband/asl-alphabet-test is a separate out-of-distribution (OOD) test: different people,
backgrounds and lighting.

Why the OOD set matters: the main dataset is many near-identical frames of the same signer in
the same room, so a random split puts almost the same photo in train and test. In-distribution
accuracy is therefore optimistic; the OOD accuracy is the honest number.
"""

import random
from pathlib import Path

from slr.common import IMAGE_EXTS

ASL_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]


def find_class_root(root, min_classes: int = 20) -> Path:
    """Return the directory under `root` whose subfolders are the class folders.

    Kaggle archives nest folders differently between versions
    (e.g. asl_alphabet_train/asl_alphabet_train/A/...), so search rather than hard-code paths.
    """
    root = Path(root).expanduser()
    wanted = {c.lower() for c in ASL_CLASSES}
    best, best_hits = None, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        try:
            names = {p.name.lower() for p in d.iterdir() if p.is_dir()}
        except PermissionError:
            continue
        hits = len(names & wanted)
        if hits > best_hits:
            best, best_hits = d, hits
    if best is None or best_hits < min_classes:
        raise FileNotFoundError(
            f"Could not find ASL class folders (A, B, ..., space) under {root}. "
            f"Check the dataset path.")
    return best


def list_images(class_root, classes=ASL_CLASSES, max_per_class=None, seed=0):
    """[(path, label_index)] for every image whose folder name matches a class (case-insensitive)."""
    class_root = Path(class_root)
    lookup = {p.name.lower(): p for p in class_root.iterdir() if p.is_dir()}
    rng = random.Random(seed)
    items, missing = [], []
    for idx, name in enumerate(classes):
        folder = lookup.get(name.lower())
        if folder is None:
            missing.append(name)
            continue
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if max_per_class and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))
        items.extend((str(f), idx) for f in files)
    if missing:
        print(f"WARNING: no folder for classes {missing} under {class_root}")
    return items


def stratified_split(items, fractions=(0.7, 0.15, 0.15), seed=0):
    """Split [(path, label)] into train/val/test, keeping class proportions."""
    rng = random.Random(seed)
    by_label = {}
    for path, label in items:
        by_label.setdefault(label, []).append(path)
    out = {"train": [], "val": [], "test": []}
    for label, paths in sorted(by_label.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        out["train"] += [(p, label) for p in paths[:n_train]]
        out["val"] += [(p, label) for p in paths[n_train:n_train + n_val]]
        out["test"] += [(p, label) for p in paths[n_train + n_val:]]
    return out


def build_manifest(train_root, ood_root=None, max_per_class=600, seed=0):
    """Manifest: {"classes": [...], "splits": {"train"|"val"|"test"|"ood": [(path, label)]}}."""
    items = list_images(find_class_root(train_root), max_per_class=max_per_class, seed=seed)
    splits = stratified_split(items, seed=seed)
    if ood_root:
        splits["ood"] = list_images(find_class_root(ood_root), seed=seed)
    return {"classes": ASL_CLASSES, "splits": splits}
