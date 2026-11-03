"""
Shared training loop pieces: resumable checkpoints, LR schedule, AMP, evaluation metrics.

Checkpointing writes two files in the run directory after every epoch:
  last.pt   model + optimizer + scheduler + AMP scaler + epoch + history (to resume exactly)
  best.pt   model weights of the best epoch by the validation metric
plus history.json (human-readable per-epoch metrics). `--resume` continues from last.pt with
the optimizer and LR schedule in the same state, so an interrupted run ends up where an
uninterrupted one would.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from slr.common import read_json, write_json


def cosine_warmup(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(1).mean())


def classification_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Per-instance top-1/top-5 (the WLASL convention), mean per-class accuracy, macro-F1."""
    from sklearn.metrics import confusion_matrix, f1_score

    preds = logits.argmax(1) if len(labels) else np.zeros(0, int)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes))) if len(labels) else \
        np.zeros((num_classes, num_classes), int)
    support = cm.sum(1)
    per_class = np.divide(np.diag(cm), support, out=np.zeros(num_classes), where=support > 0)
    return {
        "n": int(len(labels)),
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, 5),
        "mean_class_acc": float(per_class[support > 0].mean()) if (support > 0).any() else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=np.unique(labels),
                                   zero_division=0)) if len(labels) else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class_acc": per_class.tolist(),
    }


@torch.no_grad()
def predict(model, loader, device, amp=False):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        all_logits.append(out.float().cpu().numpy())
        all_labels.append(y.numpy())
    if not all_logits:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.int64)
    return np.concatenate(all_logits), np.concatenate(all_labels)


class Trainer:
    """Minimal resumable supervised trainer (classification)."""

    def __init__(self, model, run_dir, device, epochs, lr, weight_decay=0.05, label_smoothing=0.1,
                 warmup_epochs=5, steps_per_epoch=1, amp=False, grad_clip=1.0, accum_steps=1,
                 select_metric="top1"):
        self.model = model.to(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.epochs, self.amp = device, epochs, amp and device.type == "cuda"
        self.grad_clip, self.accum_steps, self.select_metric = grad_clip, accum_steps, select_metric
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        updates = max(1, steps_per_epoch // accum_steps)
        self.scheduler = cosine_warmup(self.optimizer, epochs * updates, warmup_epochs * updates)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch, self.best, self.history = 0, -1.0, []

    # -- checkpoints ------------------------------------------------------------------------

    def save(self, epoch):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": epoch, "best": self.best, "history": self.history}
        tmp = self.run_dir / "last.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.run_dir / "last.pt")
        write_json(self.run_dir / "history.json", self.history)

    def try_resume(self) -> bool:
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch, self.best, self.history = state["epoch"] + 1, state["best"], state["history"]
        print(f"Resumed from {path}: {self.start_epoch} epoch(s) done, best val "
              f"{self.select_metric} {self.best:.4f}")
        return True

    def load_best(self):
        self.model.load_state_dict(torch.load(self.run_dir / "best.pt", map_location=self.device,
                                              weights_only=True))

    # -- loop -------------------------------------------------------------------------------

    def train_epoch(self, loader, epoch):
        self.model.train()
        total, n = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}/{self.epochs}", leave=False)):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss = self.criterion(self.model(x), y) / self.accum_steps
            self.scaler.scale(loss).backward()
            if (i + 1) % self.accum_steps == 0 or i + 1 == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            total += loss.item() * self.accum_steps * len(y)
            n += len(y)
        return total / max(n, 1)

    def fit(self, train_loader, val_loader, num_classes, log_every=1):
        for epoch in range(self.start_epoch, self.epochs):
            loss = self.train_epoch(train_loader, epoch)
            logits, labels = predict(self.model, val_loader, self.device, self.amp)
            m = classification_metrics(logits, labels, num_classes)
            row = {"epoch": epoch + 1, "train_loss": loss, "val_top1": m["top1"],
                   "val_top5": m["top5"], "val_mean_class_acc": m["mean_class_acc"],
                   "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(row)
            score = m[self.select_metric]
            if score > self.best:
                self.best = score
                torch.save(self.model.state_dict(), self.run_dir / "best.pt")
            self.save(epoch)
            if (epoch + 1) % log_every == 0 or epoch + 1 == self.epochs:
                print(f"epoch {epoch + 1}/{self.epochs} loss {loss:.4f} val top1 {m['top1']:.4f} "
                      f"top5 {m['top5']:.4f} (best {self.best:.4f})")


def run_is_complete(run_dir) -> bool:
    return (Path(run_dir) / "results.json").exists()


def write_results(run_dir, results):
    write_json(Path(run_dir) / "results.json", results)


def load_config_if_resuming(run_dir, args, keys):
    """On resume, refuse to silently change settings that would make the run inconsistent."""
    old = read_json(Path(run_dir) / "config.json")
    if not old:
        return
    changed = {k: (old.get(k), getattr(args, k)) for k in keys if old.get(k) != getattr(args, k)}
    if changed:
        raise SystemExit(f"--resume with different settings than the original run: {changed}. "
                         f"Use a new --out directory, or the original settings.")
