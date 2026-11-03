"""
Pixel-based TimeSformer baseline on WLASL: fine-tunes facebook/timesformer-base-finetuned-k400
(TimeSformer, Bertasius et al. 2021, pretrained on Kinetics-400) on 8 RGB frames per clip.

    python -m slr.train_rgb --index ~/runs/slr/landmarks/wlasl100/index.json \
        --out ~/runs/slr/wlasl100/rgb_timesformer --amp

Uses the 16 frames per clip cached by `slr.extract_videos --rgb-frames 16`; training samples
8 of them (stride 2, random offset) with a random crop and horizontal flip.

This is the "heavy" reference point: ~121M parameters vs. well under 1M for the landmark models,
so the README can show the accuracy / speed trade-off. It is not exported to TFLite.

Same outputs and resume behaviour as slr.train_sequence. --tiny builds a small randomly
initialised TimeSformer without downloading anything (for tests).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from slr.common import get_device, read_json, set_seed, warn_if_low_gpu_memory, write_json
from slr.models.sequence import count_params
from slr.training import Trainer, classification_metrics, predict, run_is_complete, write_results

PRETRAINED = "facebook/timesformer-base-finetuned-k400"


class RGBClipDataset(Dataset):
    def __init__(self, entries, num_frames=8, train=False, mean=(0.45,) * 3, std=(0.225,) * 3,
                 image_size=224, seed=0):
        self.entries = [e for e in entries if Path(e["rgb"]).exists()]
        self.k, self.train, self.size = num_frames, train, image_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        import torch.nn.functional as F

        e = self.entries[i]
        frames = np.load(e["rgb"])                               # (16, H, W, 3) uint8
        n = len(frames)
        stride = max(1, n // self.k)
        offset = int(self.rng.integers(0, stride)) if self.train else 0
        idx = np.clip(offset + stride * np.arange(self.k), 0, n - 1)
        x = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2).float() / 255.0   # (T, 3, H, W)
        if self.train:
            h, w = x.shape[-2:]
            s = self.rng.uniform(0.8, 1.0)
            ch, cw = int(h * s), int(w * s)
            top, left = int(self.rng.integers(0, h - ch + 1)), int(self.rng.integers(0, w - cw + 1))
            x = x[..., top:top + ch, left:left + cw]
            if self.rng.random() < 0.5:
                x = x.flip(-1)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std, int(e["label"])


class HFVideoClassifier(torch.nn.Module):
    """Wraps a Hugging Face video model so forward(x) returns logits (Trainer expects that)."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, x):
        return self.hf(pixel_values=x).logits


def build_model(num_classes, num_frames, tiny=False, grad_ckpt=False):
    from transformers import TimesformerConfig, TimesformerForVideoClassification

    if tiny:
        cfg = TimesformerConfig(image_size=64, patch_size=16, num_frames=num_frames, hidden_size=32,
                                num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                                num_labels=num_classes)
        return HFVideoClassifier(TimesformerForVideoClassification(cfg)), (0.45,) * 3, (0.225,) * 3, 64
    m = TimesformerForVideoClassification.from_pretrained(
        PRETRAINED, num_labels=num_classes, ignore_mismatched_sizes=True, num_frames=num_frames)
    if grad_ckpt:
        m.gradient_checkpointing_enable()
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(PRETRAINED)
        mean, std = tuple(proc.image_mean), tuple(proc.image_std)
    except Exception:
        mean, std = (0.45,) * 3, (0.225,) * 3
    return HFVideoClassifier(m), mean, std, m.config.image_size


def run(args):
    out = Path(args.out).expanduser()
    if run_is_complete(out) and not args.force:
        print(f"{out} already finished; use --force to retrain.")
        return read_json(out / "results.json")
    set_seed(args.seed)
    device = get_device(args.device)
    warn_if_low_gpu_memory(8.0)
    index = read_json(Path(args.index).expanduser())
    classes = index["classes"]

    model, mean, std, size = build_model(len(classes), args.num_frames, args.tiny, args.grad_checkpointing)
    mk = lambda s, train: RGBClipDataset(index["splits"][s], args.num_frames, train, mean, std, size, args.seed)
    ds = {s: mk(s, s == "train") for s in ("train", "val", "test")}
    print("RGB TimeSformer: " + ", ".join(f"{s} {len(v)}" for s, v in ds.items()) +
          f" clips, {count_params(model) / 1e6:.1f}M params")
    if len(ds["train"]) == 0:
        raise SystemExit("No cached RGB frames found; run slr.extract_videos with --rgb-frames 16.")
    from slr.datasets import worker_init
    dl = {s: DataLoader(v, batch_size=args.batch_size, shuffle=(s == "train"), num_workers=args.num_workers,
                        pin_memory=True, worker_init_fn=worker_init) for s, v in ds.items()}

    trainer = Trainer(model, out, device, args.epochs, args.lr, weight_decay=0.05, label_smoothing=0.1,
                      warmup_epochs=1, steps_per_epoch=len(dl["train"]), amp=args.amp,
                      accum_steps=args.accum_steps)
    if args.resume:
        trainer.try_resume()
    write_json(out / "config.json", {**vars(args), "classes": classes, "params": count_params(model),
                                     "pretrained": None if args.tiny else PRETRAINED})
    t0 = time.time()
    trainer.fit(dl["train"], dl["val"], len(classes), log_every=1)
    trainer.load_best()
    logits, labels = predict(trainer.model, dl["test"], device, trainer.amp)
    test = classification_metrics(logits, labels, len(classes))
    results = {"model": "rgb_timesformer", "params": count_params(model), "num_frames": args.num_frames,
               "pretrained": None if args.tiny else PRETRAINED, "best_val_top1": trainer.best,
               "train_minutes": (time.time() - t0) / 60, "test": test, "classes": classes}
    write_results(out, results)
    print(f"TEST rgb_timesformer: top1 {test['top1']:.4f} top5 {test['top5']:.4f}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation (effective batch = batch x accum)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true", help="Less GPU memory, slower")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny", action="store_true", help="Tiny random model, no download (tests)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
