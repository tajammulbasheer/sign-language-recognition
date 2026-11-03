"""
End-to-end smoke test on synthetic data: every stage of the pipeline, in minutes, on CPU.

    python scripts/smoke_test.py [--workdir /tmp/slr_smoke] [--keep]

Builds a tiny fake ASL alphabet (29 classes) and a tiny fake WLASL (videos + WLASL_v0.3.json in the
real layout), then runs: landmark extraction (both backends) -> static MLP + CNN -> video
extraction (multiprocessing, backend comparison, RGB cache) -> BiLSTM / GRU / TimeSformer
(+ ablations, 2 seeds) -> RGB TimeSformer (tiny) -> TFLite export + parity -> real-time demo
benchmark -> report into a copy of README.md.

MediaPipe and YOLO are replaced by deterministic fakes (--fake), so nothing is downloaded. The
TFLite step needs requirements-export.txt and is skipped if litert-torch isn't installed.
Numbers produced here are meaningless; the point is that every stage runs and connects.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slr.data.asl_alphabet import ASL_CLASSES  # noqa: E402


def make_asl(root, per_class, seed, jitter):
    rng = np.random.default_rng(seed)
    for ci, c in enumerate(ASL_CLASSES):
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = np.zeros((96, 96, 3), np.uint8)
            if c != "nothing":     # "nothing" = no hand, like the real class
                base = np.array([(ci * 37) % 255, (ci * 91) % 255, (ci * 53) % 255], np.float32)
                img[:] = np.clip(base + rng.normal(0, jitter, 3), 30, 255)
                img[20 + ci:60, 20:40 + ci] = 255 - img[0, 0]
            cv2.imwrite(str(d / f"{c}_{i}.jpg"), img)


def make_wlasl(root, glosses=6, per_split=(6, 2, 2), seed=0):
    rng = np.random.default_rng(seed)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    ann = []
    for g in range(glosses):
        inst = []
        for split, n in zip(("train", "val", "test"), per_split):
            for i in range(n):
                vid = f"{g:02d}{split[0]}{i}"
                nf = int(rng.integers(20, 40))
                w = cv2.VideoWriter(str(root / "videos" / f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 72))
                for t in range(nf):
                    frame = np.full((72, 96, 3), 40, np.uint8)
                    phase = t / nf * 2 * np.pi
                    x = int(48 + 30 * np.cos(phase * (1 + g % 3)))
                    y = int(36 + 20 * np.sin(phase + g))
                    color = ((g * 50) % 255, (g * 90 + t * 3) % 255, 200)
                    cv2.circle(frame, (x, y), 14, color, -1)
                    w.write(frame)
                w.release()
                inst.append({"video_id": vid, "split": split, "frame_start": 1, "frame_end": -1,
                             "signer_id": i, "fps": 25})
        ann.append({"gloss": f"word{g}", "instances": inst})
    (root / "WLASL_v0.3.json").write_text(json.dumps(ann))


def sh(cmd, log, name):
    t = time.time()
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True)
    (log / f"{name}.log").write_text(r.stdout + "\n" + r.stderr)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"  [{status:6}] {name} ({time.time() - t:.0f}s)")
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-3000:])
        raise SystemExit(f"Smoke test failed at step '{name}' (log: {log / (name + '.log')})")
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true", help="Keep the work directory afterwards")
    args = ap.parse_args()
    work = Path(args.workdir or tempfile.mkdtemp(prefix="slr_smoke_"))
    if work.exists():
        shutil.rmtree(work)
    data, runs, log = work / "data", work / "runs", work / "logs"
    for p in (data, runs, log):
        p.mkdir(parents=True)
    print(f"Smoke test in {work}")

    make_asl(data / "asl_alphabet" / "asl_alphabet_train" / "asl_alphabet_train", 20, 0, 6)
    make_asl(data / "asl_alphabet_test" / "asl-alphabet-test", 4, 1, 25)
    make_wlasl(data / "wlasl")
    lm_img, lm_vid = runs / "landmarks" / "asl_alphabet", runs / "landmarks" / "wlasl100"
    fast = ["--device", "cpu", "--num-workers", "0"]

    sh(["-m", "slr.extract_images", "--asl-dir", str(data / "asl_alphabet"), "--ood-dir",
        str(data / "asl_alphabet_test"), "--out", str(lm_img), "--fake", "--max-per-class", "20"], log, "extract_images")
    sh(["-m", "slr.extract_images", "--asl-dir", str(data / "asl_alphabet"), "--out", str(lm_img), "--fake"],
       log, "extract_images_rerun_skips")
    sh(["-m", "slr.train_static", "--kind", "mlp", "--landmarks", str(lm_img / "yolo_mediapipe.npz"),
        "--out", str(runs / "static" / "mlp_yolo_mediapipe"), "--epochs", "4", *fast], log, "train_static_mlp")
    sh(["-m", "slr.train_static", "--kind", "mlp", "--landmarks", str(lm_img / "mediapipe.npz"),
        "--out", str(runs / "static" / "mlp_mediapipe"), "--epochs", "2", *fast], log, "train_static_mlp_mp")
    sh(["-m", "slr.train_static", "--kind", "cnn", "--manifest", str(lm_img / "manifest.json"), "--arch", "resnet18",
        "--no-pretrained", "--image-size", "64", "--out", str(runs / "static" / "cnn"), "--epochs", "1", *fast],
       log, "train_static_cnn")

    sh(["-m", "slr.extract_videos", "--wlasl-dir", str(data / "wlasl"), "--num-classes", "5", "--out", str(lm_vid),
        "--fake", "--workers", "2", "--chunk-size", "4", "--compare-backend", "mediapipe", "--compare-n", "6"],
       log, "extract_videos")
    out = sh(["-m", "slr.extract_videos", "--wlasl-dir", str(data / "wlasl"), "--num-classes", "5", "--out",
              str(lm_vid), "--fake", "--workers", "1"], log, "extract_videos_resume")
    assert "0 to extract" in out, "re-running extraction should find everything cached"
    index = str(lm_vid / "index.json")

    seq = runs / "wlasl100"
    configs = [("bilstm", [], 0), ("gru", [], 0), ("timesformer", [], 0), ("timesformer", [], 1),
               ("timesformer", ["--no-augment"], 0), ("bilstm", ["--no-pose"], 0)]
    for model, extra, seed in configs:
        tag = f"{model}_{'hands' if '--no-pose' in extra else 'pose'}_{'noaug' if '--no-augment' in extra else 'aug'}_s{seed}"
        sh(["-m", "slr.train_sequence", "--index", index, "--model", model, "--out", str(seq / tag),
            "--epochs", "3", "--batch-size", "8", "--seed", str(seed), "--log-every", "1", *extra, *fast],
           log, f"train_{tag}")
    # Resume: pretend the run was interrupted after its last checkpoint.
    (seq / "bilstm_pose_aug_s0" / "results.json").unlink()
    out = sh(["-m", "slr.train_sequence", "--index", index, "--model", "bilstm", "--out", str(seq / "bilstm_pose_aug_s0"),
              "--epochs", "3", "--batch-size", "8", "--resume", *fast], log, "train_resume")
    assert "Resumed" in out
    sh(["-m", "slr.train_rgb", "--index", index, "--out", str(seq / "rgb_timesformer"), "--tiny", "--epochs", "1",
        "--batch-size", "2", "--accum-steps", "2", *fast], log, "train_rgb_tiny")

    import importlib.util

    have_litert = importlib.util.find_spec("litert_torch") is not None
    if not have_litert:
        print("  [skip  ] TFLite export (pip install -r requirements-export.txt to include it)")
    if have_litert:
        sh(["-m", "slr.export_tflite", "--runs", str(seq / "timesformer_pose_aug_s0"), str(seq / "bilstm_pose_aug_s0"),
            str(seq / "gru_pose_aug_s0"), str(runs / "static" / "mlp_yolo_mediapipe"), str(seq / "rgb_timesformer"),
            "--index", index, "--static-landmarks", str(lm_img / "yolo_mediapipe.npz"), "--bench-iters", "20", "--onnx"],
           log, "export_tflite")
        demo_model = seq / "timesformer_pose_aug_s0" / "export" / "model.tflite"
        static_model = runs / "static" / "mlp_yolo_mediapipe" / "export" / "model.tflite"
    else:
        demo_model, static_model = seq / "timesformer_pose_aug_s0", runs / "static" / "mlp_yolo_mediapipe"
    clip = next((data / "wlasl" / "videos").glob("*.mp4"))
    sh(["-m", "slr.demo", "--model", str(demo_model), "--source", str(clip), "--fake", "--no-display",
        "--window", "16", "--stride", "2", "--output", str(work / "annotated.mp4"), "--benchmark", "30",
        "--bench-out", str(runs / "demo_benchmark.json")], log, "demo_sequence")
    sh(["-m", "slr.demo", "--model", str(static_model), "--source", str(clip), "--fake", "--no-display",
        "--benchmark", "10"], log, "demo_static")

    readme = work / "README.md"
    shutil.copy(REPO / "README.md", readme)
    sh(["-m", "slr.report", "--runs-dir", str(runs), "--assets-dir", str(work / "assets"), "--readme", str(readme)],
       log, "report")
    block = readme.read_text().split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    figures = sorted(p.name for p in (work / "assets").glob("*.png"))
    print(f"  README results block: {len(block.splitlines())} lines, figures: {figures}")
    print(f"PASSED. Logs in {log}" if args.keep else "PASSED.")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
