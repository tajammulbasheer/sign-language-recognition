"""
Export trained landmark models to TensorFlow Lite (LiteRT), check they match PyTorch, and
benchmark CPU latency.

    python -m slr.export_tflite --runs ~/runs/slr/wlasl100/timesformer_pose_aug_s0 \
        ~/runs/slr/wlasl100/bilstm_pose_aug_s0 ~/runs/slr/static/mlp_yolo_mediapipe \
        --index ~/runs/slr/landmarks/wlasl100/index.json \
        --static-landmarks ~/runs/slr/landmarks/asl_alphabet/yolo_mediapipe.npz

For each run directory (from slr.train_sequence or slr.train_static --kind mlp):
  <run>/export/model.tflite      float32 TFLite model, batch size 1
  <run>/export/model.onnx        (with --onnx) ONNX copy for other runtimes
  <run>/export/export_results.json
      parity:  top-1 agreement and max |logit difference| between PyTorch and TFLite on the
               whole test set
      latency: median / p95 milliseconds per clip on CPU, for TFLite (1 and 4 threads) and
               PyTorch (1 thread), plus model size in MB
Run directories from slr.train_rgb get a PyTorch latency benchmark only (GPU and CPU): the
121M-parameter pixel model is the heavy reference point and isn't meant for TFLite.

Conversion uses Google's litert-torch (pip install -r requirements-export.txt, Linux).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from slr.common import read_json, write_json


def load_model(run_dir: Path):
    cfg = read_json(run_dir / "config.json")
    if cfg is None:
        raise FileNotFoundError(f"{run_dir}/config.json missing; is this a training run directory?")
    if "kind" in cfg:                                   # static
        from slr.landmarks.features import C, NUM_HAND_JOINTS
        from slr.models.static import LandmarkMLP

        if cfg["kind"] != "mlp":
            raise ValueError("Only the landmark MLP is exported; the CNN baseline is not.")
        model, kind = LandmarkMLP(NUM_HAND_JOINTS, C, len(cfg["classes"])), "static"
        shape = (1, NUM_HAND_JOINTS, C)
    elif cfg.get("model") in ("bilstm", "gru", "timesformer"):
        from slr.models.sequence import build_sequence_model

        model = build_sequence_model(cfg["model"], cfg["num_joints"], cfg["channels"],
                                     len(cfg["classes"]), cfg["num_frames"])
        kind, shape = "sequence", (1, cfg["num_frames"], cfg["num_joints"], cfg["channels"])
    elif "pretrained" in cfg:                           # RGB TimeSformer
        from slr.train_rgb import build_model

        model, _, _, size = build_model(len(cfg["classes"]), cfg["num_frames"], tiny=cfg.get("tiny", False))
        kind, shape = "rgb", (1, cfg["num_frames"], 3, size, size)
    else:
        raise ValueError(f"Unrecognised run directory {run_dir}")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    return model.eval(), kind, shape, cfg


def time_fn(fn, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def make_interpreter(path, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


def tflite_runner(it):
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def run(x):
        it.set_tensor(inp, x)
        it.invoke()
        return it.get_tensor(out)
    return run


def test_inputs(kind, cfg, args):
    """Whole test split as (N, ...) float32 arrays, in the same form the model was trained on."""
    if kind == "sequence":
        from slr.datasets import SequenceDataset

        index = read_json(Path(args.index).expanduser())
        ds = SequenceDataset(index["splits"]["test"], cfg["num_frames"], cfg["use_pose"], train=False)
        xs, ys = zip(*[ds[i] for i in range(len(ds))]) if len(ds) else ([], [])
    else:
        d = np.load(Path(args.static_landmarks).expanduser())
        xs, ys = [torch.from_numpy(x) for x in d["test_features"]], list(d["test_labels"])
    return np.stack([x.numpy() for x in xs]).astype(np.float32), np.array(ys)


def export_one(run_dir: Path, args):
    model, kind, shape, cfg = load_model(run_dir)
    out_dir = run_dir / "export"
    out_dir.mkdir(exist_ok=True)
    res = {"run": str(run_dir), "kind": kind, "input_shape": list(shape),
           "params": sum(p.numel() for p in model.parameters())}
    torch.set_num_threads(1)
    sample = torch.randn(*shape)

    if kind == "rgb":
        with torch.no_grad():
            res["latency_torch_cpu_1thread"] = time_fn(lambda: model(sample), warmup=2, runs=10)
            if torch.cuda.is_available():
                m, s = model.cuda(), sample.cuda()

                def gpu():
                    with torch.autocast("cuda"):
                        m(s)
                    torch.cuda.synchronize()
                res["latency_torch_gpu_fp16"] = time_fn(gpu, warmup=5, runs=30)
        write_json(out_dir / "export_results.json", res)
        print(f"{run_dir.name}: RGB model benchmarked (no TFLite export)")
        return res

    try:
        import litert_torch
    except ImportError as e:
        raise SystemExit("litert-torch is not installed: pip install -r requirements-export.txt") from e

    tfl_path = out_dir / "model.tflite"
    write_json(out_dir / "model_meta.json", {   # everything the demo needs to use model.tflite
        "kind": kind, "classes": cfg["classes"], "input_shape": list(shape),
        "num_frames": cfg.get("num_frames"), "use_pose": cfg.get("use_pose", False)})
    with torch.no_grad():
        litert_torch.convert(model, (sample,)).export(str(tfl_path))
    res["tflite_mb"] = tfl_path.stat().st_size / 2**20
    if args.onnx:
        onnx_path = out_dir / "model.onnx"
        try:
            torch.onnx.export(model, (sample,), str(onnx_path), input_names=["landmarks"],
                              output_names=["logits"], dynamo=False)
            res["onnx_mb"] = onnx_path.stat().st_size / 2**20
        except Exception as e:  # ONNX is a bonus format; never lose the TFLite result over it
            res["onnx_error"] = repr(e)
            print(f"  ONNX export failed ({e}); continuing with TFLite only")

    # Parity on the real test set.
    xs, ys = test_inputs(kind, cfg, args)
    run1 = tflite_runner(make_interpreter(tfl_path, 1))
    with torch.no_grad():
        torch_logits = np.concatenate([model(torch.from_numpy(xs[i:i + 64])).numpy()
                                       for i in range(0, len(xs), 64)]) if len(xs) else np.zeros((0, 1))
    tfl_logits = np.stack([run1(x[None])[0] for x in xs]) if len(xs) else np.zeros((0, 1))
    res["parity"] = {
        "n": int(len(xs)),
        "top1_agreement": float((torch_logits.argmax(1) == tfl_logits.argmax(1)).mean()) if len(xs) else None,
        "max_abs_logit_diff": float(np.abs(torch_logits - tfl_logits).max()) if len(xs) else None,
        "tflite_top1": float((tfl_logits.argmax(1) == ys).mean()) if len(xs) else None,
        "torch_top1": float((torch_logits.argmax(1) == ys).mean()) if len(xs) else None,
    }

    x0 = sample.numpy()
    for threads in (1, 4):
        r = tflite_runner(make_interpreter(tfl_path, threads))
        res[f"latency_tflite_{threads}thread"] = time_fn(lambda: r(x0), runs=args.bench_iters)
    with torch.no_grad():
        res["latency_torch_cpu_1thread"] = time_fn(lambda: model(sample), runs=args.bench_iters)
    write_json(out_dir / "export_results.json", res)
    p = res["parity"]
    print(f"{run_dir.name}: {res['tflite_mb']:.2f} MB, TFLite {res['latency_tflite_1thread']['median_ms']:.2f} ms "
          f"(1 thread), agreement {p['top1_agreement']}, max diff {p['max_abs_logit_diff']}")
    return res


def run(args):
    results = []
    for r in args.runs:
        run_dir = Path(r).expanduser()
        done = read_json(run_dir / "export" / "export_results.json")
        if done and not args.force:
            print(f"{run_dir.name}: already exported, skipping")
            results.append(done)
            continue
        results.append(export_one(run_dir, args))
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--index", help="WLASL index.json (for sequence-model parity check)")
    p.add_argument("--static-landmarks", help="ASL alphabet <backend>.npz (for MLP parity check)")
    p.add_argument("--onnx", action="store_true", help="Also export ONNX")
    p.add_argument("--bench-iters", type=int, default=200, help="Timed runs per latency benchmark")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
