"""
Train a sequence model on cached WLASL landmark sequences.

    python -m slr.train_sequence --index ~/runs/slr/landmarks/wlasl100/index.json \
        --model timesformer --out ~/runs/slr/wlasl100/timesformer_pose_aug_s0 --seed 0

--model        bilstm | gru | timesformer
--no-pose      hands only (drop the 7 upper-body joints)
--no-augment   turn off temporal + spatial augmentation (ablation)

The official WLASL split is used: train for fitting, val for choosing the best epoch, and test
is evaluated exactly once, with the best-val checkpoint, at the end.

Outputs in --out: best.pt, last.pt, history.json, config.json and results.json
(test top-1/top-5, mean per-class accuracy, macro-F1, confusion matrix, parameter count).
Re-running skips a finished run; --resume continues an interrupted one.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from slr.common import get_device, read_json, set_seed, warn_if_low_gpu_memory, write_json
from slr.datasets import SequenceDataset, worker_init
from slr.landmarks.features import C, num_joints
from slr.models.sequence import MODELS, build_sequence_model, count_params
from slr.training import (Trainer, classification_metrics, load_config_if_resuming, predict,
                          run_is_complete, write_results)

CONFIG_KEYS = ["index", "model", "num_frames", "use_pose", "augment", "epochs", "lr", "seed",
               "batch_size"]


def run(args):
    out = Path(args.out).expanduser()
    if run_is_complete(out) and not args.force:
        print(f"{out} already finished (results.json exists); use --force to retrain.")
        return read_json(out / "results.json")
    set_seed(args.seed)
    device = get_device(args.device)
    warn_if_low_gpu_memory(2.0)

    index = read_json(Path(args.index).expanduser())
    classes = index["classes"]
    J = num_joints(args.use_pose)
    mk = lambda split, train: SequenceDataset(index["splits"][split], args.num_frames,
                                              args.use_pose, train=train, augment=args.augment,
                                              seed=args.seed)
    train_ds, val_ds, test_ds = mk("train", True), mk("val", False), mk("test", False)
    print(f"{args.model}: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)} clips, "
          f"{len(classes)} classes, {J} joints, pose={args.use_pose}, augment={args.augment}")
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                            num_workers=args.num_workers, pin_memory=True,
                                            worker_init_fn=worker_init,
                                            persistent_workers=args.num_workers > 0,
                                            generator=torch.Generator().manual_seed(args.seed))
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    model = build_sequence_model(args.model, J, C, len(classes), args.num_frames)
    trainer = Trainer(model, out, device, args.epochs, args.lr, weight_decay=args.weight_decay,
                      label_smoothing=args.label_smoothing, warmup_epochs=args.warmup_epochs,
                      steps_per_epoch=len(train_dl), amp=args.amp)
    if args.resume and trainer.try_resume():
        load_config_if_resuming(out, args, CONFIG_KEYS)
    write_json(out / "config.json", {**vars(args), "num_joints": J, "channels": C,
                                     "classes": classes, "params": count_params(model)})

    t0 = time.time()
    trainer.fit(train_dl, val_dl, len(classes), log_every=args.log_every)
    trainer.load_best()
    logits, labels = predict(trainer.model, test_dl, device, trainer.amp)
    test = classification_metrics(logits, labels, len(classes))
    best_epoch = int(np.argmax([h["val_top1"] for h in trainer.history])) + 1
    results = {
        "model": args.model, "use_pose": args.use_pose, "augment": args.augment, "seed": args.seed,
        "params": count_params(model), "num_frames": args.num_frames, "num_joints": J,
        "best_epoch": best_epoch, "best_val_top1": trainer.best,
        "train_minutes": (time.time() - t0) / 60,
        "test": test, "classes": classes,
    }
    np.save(out / "test_logits.npy", logits)
    np.save(out / "test_labels.npy", labels)
    write_results(out, results)
    print(f"TEST {args.model}: top1 {test['top1']:.4f} top5 {test['top5']:.4f} "
          f"mean-class {test['mean_class_acc']:.4f} (best val epoch {best_epoch})")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="index.json from slr.extract_videos")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--no-pose", dest="use_pose", action="store_false")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--amp", action="store_true", help="Mixed precision on GPU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="Retrain even if results.json exists")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
