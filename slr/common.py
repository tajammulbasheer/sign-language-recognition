"""Shared helpers: seeding, devices, JSON I/O, small path utilities."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 0):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_memory_report() -> str:
    """One-line summary of free/total GPU memory, or a note that no GPU is visible."""
    if not torch.cuda.is_available():
        return "GPU: not available (running on CPU)"
    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    return f"GPU: {name}, {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB"


def warn_if_low_gpu_memory(min_free_gb: float = 6.0):
    print(gpu_memory_report())
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2**30 < min_free_gb:
            print(f"WARNING: less than {min_free_gb} GB of GPU memory is free. Another process may "
                  f"be using the GPU; lower --batch-size if you hit out-of-memory errors.")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)  # atomic: an interrupted write never leaves a half-written file


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_files(folder, exts):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
