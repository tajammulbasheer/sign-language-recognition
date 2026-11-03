"""
Extract per-frame hand + upper-body landmarks from WLASL videos and cache them, one file per video.

    python -m slr.extract_videos --wlasl-dir ~/data/slr/wlasl --num-classes 100 \
        --models-dir ~/data/slr/models --yolo-weights ~/runs/slr/hand_detector/weights/best.pt \
        --out ~/runs/slr/landmarks/wlasl100 --workers 4 --rgb-frames 16

Output in --out:
  cache/<backend>/<video_id>.npz   hands (L,2,21,3), hand_present (L,2), raw_hand_present (L,2),
                                   pose (L,33,4), pose_present (L,), aspect (W/H)
  rgb/<video_id>.npy               (optional) 16 RGB frames at 224x224 for the pixel TimeSformer
  index.json                       manifest + per-video frame counts and detection rates
  stats.json                       dataset availability and detection statistics

Resumable: videos whose cache file already exists are skipped, so an interrupted run continues
where it stopped. Videos that fail to decode are logged in stats.json and left out.

--compare-backend mediapipe --compare-n 150 also extracts plain full-frame MediaPipe on the
first 150 val+test videos, to measure how much the YOLO crops improve hand detection.
"""

import argparse
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from slr.common import read_json, write_json


def _worker(job):
    """Process a chunk of videos in its own process (MediaPipe and CUDA are not fork-safe)."""
    (entries, backend, cache_dir, rgb_dir, rgb_frames, hand_model, pose_model,
     yolo_weights, use_pose, fake, max_frames, device) = job
    from slr.data.video import read_frames, resize_center_crop, sample_indices
    from slr.landmarks.extractor import LandmarkExtractor, stack_frames
    from slr.landmarks.features import interpolate_gaps

    ex = LandmarkExtractor(backend=backend, hand_model=hand_model, pose_model=pose_model,
                           yolo_weights=yolo_weights, use_pose=use_pose, fake=fake, device=device)
    results = []
    for e in entries:
        dest = Path(cache_dir) / f"{e['video_id']}.npz"
        try:
            frames, info = read_frames(e["path"], e["frame_start"], e["frame_end"], max_frames)
            arrays = stack_frames([ex(f) for f in frames])
            raw = arrays["hand_present"].copy()
            arrays["hands"], arrays["hand_present"] = interpolate_gaps(arrays["hands"], raw)
            h, w = frames[0].shape[:2]
            np.savez_compressed(dest, raw_hand_present=raw, aspect=np.float32(w / h), **arrays)
            if rgb_dir is not None:
                idx = sample_indices(len(frames), rgb_frames)
                np.save(Path(rgb_dir) / f"{e['video_id']}.npy",
                        np.stack([resize_center_crop(frames[i]) for i in idx]).astype(np.uint8))
            results.append({"video_id": e["video_id"], "ok": True, **info,
                            "hand_rate": float(raw.any(1).mean()),
                            "pose_rate": float(arrays["pose_present"].mean())})
        except Exception as err:  # one bad video must not kill the whole extraction
            results.append({"video_id": e["video_id"], "ok": False, "error": repr(err),
                            "trace": traceback.format_exc(limit=2)})
    ex.close()
    return results


def run_backend(entries, backend, out, args, rgb=False, use_pose=True):
    cache_dir = out / "cache" / backend.replace("+", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = None
    if rgb:
        rgb_dir = out / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries
            if not (cache_dir / f"{e['video_id']}.npz").exists()
            or (rgb_dir is not None and not (rgb_dir / f"{e['video_id']}.npy").exists())]
    print(f"{backend}: {len(entries) - len(todo)} cached, {len(todo)} to extract")
    models = Path(args.models_dir).expanduser() if args.models_dir else Path(".")
    chunk = max(1, args.chunk_size)
    jobs = [(todo[i:i + chunk], backend, str(cache_dir), str(rgb_dir) if rgb_dir else None,
             args.rgb_frames, str(models / "hand_landmarker.task"),
             str(models / "pose_landmarker_lite.task"), args.yolo_weights, use_pose, args.fake,
             args.max_frames, args.device) for i in range(0, len(todo), chunk)]
    results = []
    if jobs:
        if args.workers <= 1:
            for j in tqdm(jobs, desc=backend):
                results += _worker(j)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool:
                for r in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc=backend):
                    results += r
    failed = [r for r in results if not r["ok"]]
    for r in failed[:5]:
        print(f"  failed {r['video_id']}: {r['error']}")
    return cache_dir, results


def summarize(cache_dir, entries):
    rates, frames = [], []
    for e in entries:
        p = cache_dir / f"{e['video_id']}.npz"
        if p.exists():
            with np.load(p) as d:
                rates.append(float(d["raw_hand_present"].any(1).mean()))
                frames.append(int(len(d["raw_hand_present"])))
    return {"videos": len(rates), "mean_frames": float(np.mean(frames)) if frames else 0.0,
            "hand_detection_rate": float(np.mean(rates)) if rates else 0.0}


def run(args):
    from slr.data.wlasl import build_manifest

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(out / "manifest.json")
    if manifest is None:
        manifest = build_manifest(args.wlasl_dir, args.num_classes)
        write_json(out / "manifest.json", manifest)
    print("WLASL availability:", manifest["stats"]["available"],
          "missing:", manifest["stats"]["missing_videos"])
    all_entries = [e for s in ("train", "val", "test") for e in manifest["splits"][s]]
    if args.limit:
        all_entries = all_entries[: args.limit]

    cache_dir, results = run_backend(all_entries, args.backend, out, args,
                                     rgb=args.rgb_frames > 0, use_pose=True)
    stats = read_json(out / "stats.json", {})
    stats["dataset"] = manifest["stats"]
    stats["main_backend"] = args.backend
    stats.setdefault("failed_videos", {})
    for r in results:
        if not r["ok"]:
            stats["failed_videos"][r["video_id"]] = r["error"]
        else:
            stats["failed_videos"].pop(r["video_id"], None)
    stats["detection"] = {args.backend: summarize(cache_dir, all_entries)}

    if args.compare_backend:
        sub = [e for s in ("val", "test") for e in manifest["splits"][s]][: args.compare_n]
        sub = [e for e in sub if (cache_dir / f"{e['video_id']}.npz").exists()]
        cmp_dir, _ = run_backend(sub, args.compare_backend, out, args, use_pose=False)
        stats["backend_comparison"] = {
            "videos": len(sub),
            args.backend: summarize(cache_dir, sub),
            args.compare_backend: summarize(cmp_dir, sub),
        }
        print("Hand detection rate on the same videos:",
              {k: f"{v['hand_detection_rate']:.1%}" for k, v in stats["backend_comparison"].items()
               if isinstance(v, dict)})

    # Index of usable videos, per split.
    index = {"classes": manifest["classes"], "backend": args.backend, "splits": {}}
    for s in ("train", "val", "test"):
        index["splits"][s] = [
            {**e, "cache": str(cache_dir / f"{e['video_id']}.npz"),
             "rgb": str(out / "rgb" / f"{e['video_id']}.npy")}
            for e in manifest["splits"][s] if (cache_dir / f"{e['video_id']}.npz").exists()]
    stats["usable_videos"] = {s: len(v) for s, v in index["splits"].items()}
    write_json(out / "index.json", index)
    write_json(out / "stats.json", stats)
    print("Usable videos per split:", stats["usable_videos"])
    print(f"Hand detection rate ({args.backend}): {stats['detection'][args.backend]['hand_detection_rate']:.1%}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wlasl-dir", required=True)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="yolo+mediapipe", choices=["mediapipe", "yolo+mediapipe", "yolo-pose"])
    p.add_argument("--compare-backend", default=None, choices=["mediapipe", "yolo+mediapipe", "yolo-pose"])
    p.add_argument("--compare-n", type=int, default=150)
    p.add_argument("--rgb-frames", type=int, default=16, help="Also cache N RGB frames per video (0 = off)")
    p.add_argument("--max-frames", type=int, default=150, help="Subsample longer clips to this many frames")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu")
    p.add_argument("--limit", type=int, default=None, help="Only the first N videos (debugging)")
    p.add_argument("--fake", action="store_true", help="Use fake landmark models (tests only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
