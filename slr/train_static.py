"""
Static signs (ASL alphabet images): landmark MLP vs. CNN on raw pixels.

    # landmark MLP on features from slr.extract_images
    python -m slr.train_static --kind mlp --landmarks ~/runs/slr/landmarks/asl_alphabet/yolo_mediapipe.npz \
        --out ~/runs/slr/static/mlp_yolo_mediapipe

    # CNN baseline on the same images
    python -m slr.train_static --kind cnn --manifest ~/runs/slr/landmarks/asl_alphabet/manifest.json \
        --out ~/runs/slr/static/cnn

Both are evaluated on the in-distribution test split AND on the out-of-distribution ASL
Alphabet Test set (different signers, backgrounds and lighting). The comparison shows whether a
model learned hand shape or memorised the training photos' background.

Outputs in --out: best.pt, last.pt, history.json, config.json, results.json
(results.json: test and ood metrics, including accuracy and confusion matrices).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from slr.common import get_device, read_json, set_seed, warn_if_low_gpu_memory, write_json
from slr.datasets import ImageDataset, StaticLandmarkDataset, worker_init
from slr.landmarks.features import C, NUM_HAND_JOINTS
from slr.models.sequence import count_params
from slr.models.static import LandmarkMLP, build_cnn
from slr.training import Trainer, classification_metrics, predict, run_is_complete, write_results


def mlp_data(args):
    d = np.load(Path(args.landmarks).expanduser(), allow_pickle=False)
    classes = [str(c) for c in d["classes"]]
    ds = {s: StaticLandmarkDataset(d[f"{s}_features"], d[f"{s}_labels"], train=(s == "train"), seed=args.seed)
          for s in ("train", "val", "test", "ood") if f"{s}_features" in d}
    found = {s: float(d[f"{s}_found"].mean()) for s in ds if f"{s}_found" in d and len(d[f"{s}_found"])}
    return ds, classes, LandmarkMLP(NUM_HAND_JOINTS, C, len(classes)), found


def cnn_data(args):
    import torchvision.transforms as T

    manifest = read_json(Path(args.manifest).expanduser())
    classes = manifest["classes"]
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.3, 0.05),
                          T.RandomRotation(15), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor(), norm])
    ds = {s: ImageDataset(items, train_tf if s == "train" else eval_tf)
          for s, items in manifest["splits"].items()}
    return ds, classes, build_cnn(len(classes), args.arch, pretrained=not args.no_pretrained), {}


def run(args):
    out = Path(args.out).expanduser()
    if run_is_complete(out) and not args.force:
        print(f"{out} already finished; use --force to retrain.")
        return read_json(out / "results.json")
    set_seed(args.seed)
    device = get_device(args.device)
    warn_if_low_gpu_memory(2.0)
    ds, classes, model, found = (mlp_data if args.kind == "mlp" else cnn_data)(args)
    print(f"{args.kind}: " + ", ".join(f"{s} {len(v)}" for s, v in ds.items()))

    dl = {s: DataLoader(v, batch_size=args.batch_size, shuffle=(s == "train"), num_workers=args.num_workers,
                        pin_memory=True, worker_init_fn=worker_init) for s, v in ds.items()}
    trainer = Trainer(model, out, device, args.epochs, args.lr, weight_decay=args.weight_decay,
                      label_smoothing=0.1, warmup_epochs=min(3, args.epochs // 5),
                      steps_per_epoch=len(dl["train"]), amp=args.amp)
    if args.resume:
        trainer.try_resume()
    write_json(out / "config.json", {**vars(args), "classes": classes, "params": count_params(model)})

    t0 = time.time()
    trainer.fit(dl["train"], dl["val"], len(classes), log_every=args.log_every)
    trainer.load_best()
    results = {"kind": args.kind, "params": count_params(model), "seed": args.seed,
               "train_minutes": (time.time() - t0) / 60, "classes": classes,
               "hand_detection_rate": found,
               "source": str(args.landmarks if args.kind == "mlp" else args.manifest)}
    for split in ("test", "ood"):
        if split in dl:
            logits, labels = predict(trainer.model, dl[split], device, trainer.amp)
            results[split] = classification_metrics(logits, labels, len(classes))
            print(f"{split.upper()} {args.kind}: top1 {results[split]['top1']:.4f} (n={len(labels)})")
    write_results(out, results)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["mlp", "cnn"], required=True)
    p.add_argument("--landmarks", help="<backend>.npz from slr.extract_images (for --kind mlp)")
    p.add_argument("--manifest", help="manifest.json from slr.extract_images (for --kind cnn)")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="mobilenet_v3_large")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=None, help="default: 60 for mlp, 8 for cnn")
    p.add_argument("--batch-size", type=int, default=None, help="default: 256 for mlp, 64 for cnn")
    p.add_argument("--lr", type=float, default=None, help="default: 2e-3 for mlp, 5e-4 for cnn")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    defaults = {"mlp": (60, 256, 2e-3), "cnn": (8, 64, 5e-4)}[a.kind]
    a.epochs = a.epochs or defaults[0]
    a.batch_size = a.batch_size or defaults[1]
    a.lr = a.lr or defaults[2]
    if a.kind == "mlp" and not a.landmarks:
        p.error("--kind mlp needs --landmarks")
    if a.kind == "cnn" and not a.manifest:
        p.error("--kind cnn needs --manifest")
    return a


if __name__ == "__main__":
    run(parse_args())
