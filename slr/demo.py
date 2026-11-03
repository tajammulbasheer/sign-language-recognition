"""
Real-time sign recognition from a webcam or a video file.

    # word-level signs (WLASL model), webcam 0 -- run this on Windows, see note below
    python -m slr.demo --model runs/wlasl100/timesformer_pose_aug_s0/export/model.tflite \
        --models-dir data/models --yolo-weights runs/hand_detector/weights/best.pt --source 0

    # alphabet letters, on a recorded video, saving an annotated copy (works inside WSL)
    python -m slr.demo --model runs/static/mlp_yolo_mediapipe/export/model.tflite \
        --models-dir data/models --yolo-weights runs/hand_detector/weights/best.pt \
        --source my_clip.mp4 --output annotated.mp4 --no-display

--model accepts an exported model.tflite (uses model_meta.json next to it) or a training run
directory (uses best.pt with PyTorch). The pipeline per frame is: YOLO hand boxes -> MediaPipe
landmarks on each crop (+ upper-body pose for word signs) -> features -> model. Word signs are
classified from a sliding window of the last --window frames, resampled to the model's length.

--benchmark N  processes N frames and writes per-stage latency (detector + landmarks, model,
               end-to-end FPS) to --bench-out, which the README report picks up.

WSL note: WSL can't see USB webcams by default. Use a video file inside WSL, or run the demo from
Windows Python (pip install mediapipe ultralytics opencv-python ai-edge-litert), where
--source 0 opens the webcam.
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from slr.common import read_json, write_json
from slr.landmarks.extractor import LandmarkExtractor
from slr.landmarks.features import build_features


class Classifier:
    """Runs either a TFLite model or a PyTorch run directory, with one interface."""

    def __init__(self, path):
        path = Path(path).expanduser()
        if path.suffix == ".tflite":
            from slr.export_tflite import make_interpreter, tflite_runner

            meta = read_json(path.parent / "model_meta.json")
            if meta is None:
                raise FileNotFoundError(f"{path.parent}/model_meta.json missing; re-run slr.export_tflite")
            self.kind, self.classes = meta["kind"], meta["classes"]
            self.num_frames, self.use_pose = meta.get("num_frames"), meta.get("use_pose", False)
            run = tflite_runner(make_interpreter(path, 4))
            self.predict = lambda x: run(x[None].astype(np.float32))[0]
            self.backend = "tflite"
        else:
            import torch

            from slr.export_tflite import load_model

            model, self.kind, _, cfg = load_model(path)
            self.classes = cfg["classes"]
            self.num_frames, self.use_pose = cfg.get("num_frames"), cfg.get("use_pose", False)

            def predict(x):
                with torch.no_grad():
                    return model(torch.from_numpy(x[None].astype(np.float32)))[0].numpy()
            self.predict = predict
            self.backend = "pytorch"
        if self.kind not in ("sequence", "static"):
            raise ValueError("The demo runs landmark models only (sequence or static).")


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def draw(frame, fl, top, fps, kind):
    h = frame.shape[0]
    for s in range(2):
        if fl.hand_present[s]:
            for x, y, _ in fl.hands[s]:
                cv2.circle(frame, (int(x * h), int(y * h)), 3, (0, 255, 0) if s == 0 else (255, 128, 0), -1)
    y = 30
    for name, p in top:
        cv2.putText(frame, f"{name}: {p:.0%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 30
    cv2.putText(frame, f"{fps:.1f} FPS ({kind})", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def run(args):
    clf = Classifier(args.model)
    models = Path(args.models_dir).expanduser() if args.models_dir else Path(".")
    ex = LandmarkExtractor(backend=args.backend, hand_model=models / "hand_landmarker.task",
                           pose_model=models / "pose_landmarker_lite.task", yolo_weights=args.yolo_weights,
                           use_pose=clf.use_pose, fake=args.fake, device=args.device)
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source!r}. Inside WSL, webcams aren't visible: "
                         f"use a video file, or run the demo from Windows.")
    writer = None
    buf = collections.deque(maxlen=args.window)
    votes = collections.deque(maxlen=args.smooth)
    t_extract, t_model, t_total, n = [], [], [], 0
    top = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fl = ex(rgb)
        t1 = time.perf_counter()
        t_extract.append((t1 - t0) * 1000)
        buf.append(fl)
        n += 1

        logits = None
        if clf.kind == "static" and fl.hand_present.any():
            x = build_features(fl.hands[None], fl.hand_present[None], use_pose=False, use_global=False)[0]
            logits = clf.predict(x)
        elif clf.kind == "sequence" and len(buf) >= min(args.window, 8) and n % args.stride == 0:
            frames = list(buf)
            idx = np.linspace(0, len(frames) - 1, clf.num_frames).round().astype(int)
            hands = np.stack([frames[i].hands for i in idx])
            present = np.stack([frames[i].hand_present for i in idx])
            if present.any(1).mean() >= args.min_hand_ratio:
                pose = np.stack([frames[i].pose for i in idx])
                pp = np.array([frames[i].pose_present for i in idx], np.float32)
                aspect = rgb.shape[1] / rgb.shape[0]
                x = build_features(hands, present, pose, pp, use_pose=clf.use_pose, use_global=True,
                                   image_center=(0.5 * aspect, 0.5))
                logits = clf.predict(x)
        t2 = time.perf_counter()
        if logits is not None:
            t_model.append((t2 - t1) * 1000)
            votes.append(softmax(logits))
            p = np.mean(votes, 0)
            top = [(clf.classes[i], float(p[i])) for i in np.argsort(-p)[:3]]
        t_total.append((t2 - t0) * 1000)

        if not args.no_display or args.output:
            draw(bgr, fl, top, 1000 / max(np.mean(t_total[-30:]), 1e-6), clf.backend)
        if args.output:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
        if not args.no_display:
            cv2.imshow("Sign language recognition (q to quit)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.benchmark and n >= args.benchmark:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    ex.close()
    # Throughput uses the MEAN per-frame cost: the model runs only every --stride frames, so a
    # median would hide its cost entirely. Prediction latency = one frame's landmarks + one
    # model call (how stale a prediction is when it appears).
    lm_med = float(np.median(t_extract)) if t_extract else None
    model_med = float(np.median(t_model)) if t_model else None
    stats = {
        "frames": n, "backend": args.backend, "model": str(args.model), "model_runtime": clf.backend,
        "kind": clf.kind, "stride": args.stride if clf.kind == "sequence" else 1,
        "predictions": len(t_model),
        "landmarks_ms_median": lm_med,
        "model_ms_median": model_med,
        "prediction_latency_ms": (lm_med or 0) + (model_med or 0) if t_model else None,
        "frame_ms_mean": float(np.mean(t_total)) if t_total else None,
        "throughput_fps": float(1000 / np.mean(t_total)) if t_total else None,
        "extraction": ex.stats.as_dict(),
    }
    print({k: v for k, v in stats.items() if k != "extraction"})
    if args.bench_out:
        write_json(args.bench_out, stats)
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model.tflite or a training run directory")
    p.add_argument("--source", default="0", help="Webcam index or video path")
    p.add_argument("--models-dir", default=None, help="Folder with the MediaPipe .task files")
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--backend", default="yolo+mediapipe", choices=["mediapipe", "yolo+mediapipe", "yolo-pose"])
    p.add_argument("--window", type=int, default=48, help="Frames in the sliding window (word signs)")
    p.add_argument("--stride", type=int, default=4, help="Classify every N frames (word signs)")
    p.add_argument("--smooth", type=int, default=5, help="Average the last N predictions")
    p.add_argument("--min-hand-ratio", type=float, default=0.5,
                   help="Only classify a window if hands are visible in this fraction of frames")
    p.add_argument("--output", default=None, help="Write an annotated video here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--benchmark", type=int, default=0, help="Stop after N frames and report timing")
    p.add_argument("--bench-out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--fake", action="store_true", help="Fake landmark models (tests only)")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
