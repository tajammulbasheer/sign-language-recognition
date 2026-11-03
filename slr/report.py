"""
Build the results section of the README from experiment outputs. Nothing is typed by hand.

    python -m slr.report --runs-dir ~/runs/slr --readme README.md --assets-dir assets

Reads (anything missing is shown as "not run yet"):
  hand_detector/detector_results.json
  landmarks/asl_alphabet/*_stats.json         static-image hand detection rates
  static/*/results.json                        landmark MLPs and CNN baseline
  landmarks/wlasl100/stats.json                WLASL availability + detection comparison
  wlasl100/<model>_<pose|hands>_<aug|noaug>_s<seed>/results.json   sequence models
  wlasl100/rgb_timesformer/results.json        pixel TimeSformer baseline
  */export/export_results.json                 TFLite size, parity, latency
  demo_benchmark.json                          end-to-end pipeline timing (slr.demo --benchmark)

Writes <assets-dir>/*.png, <runs-dir>/report/RESULTS.md, and replaces the block between
<!-- RESULTS:START --> and <!-- RESULTS:END --> in --readme.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slr.common import read_json

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
TEXT, TEXT_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
MODEL_COLORS = {"bilstm": BLUE, "gru": ORANGE, "timesformer": AQUA, "rgb_timesformer": VIOLET, "mlp": YELLOW}
MODEL_NAMES = {"bilstm": "BiLSTM", "gru": "BiGRU", "timesformer": "Landmark TimeSformer",
               "rgb_timesformer": "RGB TimeSformer (pixels)"}
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NA = "not run yet"

# Published WLASL100 per-instance top-1 accuracy, for context only (different pipelines).
# Pose-TGCN and I3D: Li et al., WACV 2020; ST-GCN: as reported in SignBERT+ (Hu et al., 2023).
LITERATURE = [("Pose-TGCN (pose)", 55.43, "Li et al., WACV 2020"),
              ("ST-GCN (pose)", 50.78, "reported in Hu et al., SignBERT+, 2023"),
              ("I3D (RGB)", 65.89, "Li et al., WACV 2020")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%" if isinstance(x, (int, float)) else "—"


def fmt_params(n):
    return f"{n / 1e6:.0f}M" if n >= 10e6 else f"{n / 1e6:.2f}M"


def mean_std(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{100 * vals[0]:.1f}%"
    return f"{100 * np.mean(vals):.1f} ± {100 * np.std(vals):.1f}%"


def save_bars(labels, groups, title, path, ylabel="Accuracy", errors=None):
    """Grouped bars. groups = [(name, values, color)]."""
    fig, ax = plt.subplots(figsize=(7, 3.9))
    n = len(groups)
    width = 0.75 / n
    for gi, (name, vals, color) in enumerate(groups):
        keep = [i for i, v in enumerate(vals) if v is not None]   # no bar for a run that doesn't exist
        xs = [i + (gi - (n - 1) / 2) * width for i in keep]
        vs = [vals[i] for i in keep]
        err = [errors[gi][i] for i in keep] if errors else None
        bars = ax.bar(xs, vs, width * 0.92, color=color, label=name, yerr=err,
                      error_kw={"ecolor": TEXT_2, "elinewidth": 1, "capsize": 3})
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{100 * v:.0f}%", ha="center",
                    va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left")
    if n > 1:
        ax.legend(frameon=False, loc="upper right", ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------------------------------------------------- sections

def detector_section(runs):
    r = read_json(runs / "hand_detector" / "detector_results.json")
    if not r:
        return f"| Box mAP50 | Box mAP50-95 | Keypoint mAP50 |\n|---|---|---|\n| {NA} | — | — |"
    return ("| Model | Box mAP50 | Box mAP50-95 | Keypoint mAP50 |\n|---|---|---|---|\n"
            f"| {r['model'].replace('.pt', '')}, {r['epochs']} epochs | {r['box_map50']:.3f} | "
            f"{r['box_map50_95']:.3f} | {r['pose_map50']:.3f} |")


def static_section(runs, assets, figs):
    lm = runs / "landmarks" / "asl_alphabet"
    det_rows = []
    for backend, label in (("mediapipe", "MediaPipe, full image"), ("yolo_mediapipe", "YOLO crop + MediaPipe")):
        s = read_json(lm / f"{backend}_stats.json")
        if s:
            sp = s["splits"]
            det_rows.append(f"| {label} | {pct(sp.get('test', {}).get('hand_detection_rate'))} | "
                            f"{pct(sp.get('ood', {}).get('hand_detection_rate'))} |")
    det = ("| Hand found in (images that contain a hand) | ASL Alphabet test | ASL Alphabet Test (OOD) |\n|---|---|---|\n" +
           ("\n".join(det_rows) if det_rows else f"| {NA} | — | — |"))

    rows, labels, test_acc, ood_acc = [], [], [], []
    runs_static = sorted((runs / "static").glob("*/results.json")) if (runs / "static").exists() else []
    for p in runs_static:
        r = read_json(p)
        name = ("CNN on pixels (" + read_json(p.parent / "config.json", {}).get("arch", "cnn") + ")"
                if r["kind"] == "cnn" else
                "Landmark MLP (" + ("YOLO crop + MediaPipe" if Path(r.get("source", "")).stem.startswith("yolo")
                                    else "MediaPipe") + ")")
        t, o = r.get("test", {}).get("top1"), r.get("ood", {}).get("top1")
        rows.append(f"| {name} | {fmt_params(r['params'])} | {pct(t)} | {pct(o)} |")
        if t is not None and o is not None:
            labels.append(name.replace(" (", "\n("))
            test_acc.append(t)
            ood_acc.append(o)
    acc = ("| Model | Params | Test (same signer) | ASL Alphabet Test (OOD) |\n|---|---|---|---|\n" +
           ("\n".join(rows) if rows else f"| {NA} | — | — | — |"))
    if labels:
        save_bars(labels, [("In-distribution test", test_acc, BLUE), ("Out-of-distribution", ood_acc, ORANGE)],
                  "Static signs: same-signer test vs. new signers/backgrounds", assets / "static_ood.png")
        figs.append(("static_ood.png", "Static alphabet accuracy, in-distribution vs out-of-distribution"))
    return det, acc


def collect_sequence_runs(runs):
    groups = defaultdict(list)
    root = runs / "wlasl100"
    for p in sorted(root.glob("*/results.json")) if root.exists() else []:
        r = read_json(p)
        if r.get("model") == "rgb_timesformer":
            continue
        key = (r["model"], r["use_pose"], r["augment"])
        groups[key].append((r, p.parent))
    return groups


def wlasl_section(runs, assets, figs):
    stats = read_json(runs / "landmarks" / "wlasl100" / "stats.json")
    if stats:
        u = stats.get("usable_videos", {})
        miss = stats.get("dataset", {}).get("missing_videos", {})
        data = (f"WLASL100 videos available: **{u.get('train', 0)} train / {u.get('val', 0)} val / "
                f"{u.get('test', 0)} test** (official split; {sum(miss.values())} annotated videos are "
                f"missing from the public copy).")
        cmp_ = stats.get("backend_comparison")
        if cmp_:
            names = [k for k in cmp_ if isinstance(cmp_[k], dict)]
            data += ("\n\n| Hand found in (same " + str(cmp_["videos"]) + " val/test videos) | Frames with a hand |\n|---|---|\n" +
                     "\n".join(f"| {'YOLO crop + MediaPipe' if 'yolo' in k else 'MediaPipe, full frame'} | "
                               f"{pct(cmp_[k]['hand_detection_rate'])} |" for k in names))
    else:
        data = f"WLASL100 data: {NA}."

    groups = collect_sequence_runs(runs)
    rows = []
    for (model, pose, aug), items in sorted(groups.items(), key=lambda kv: (list(MODEL_NAMES).index(kv[0][0]), not kv[0][1], not kv[0][2])):
        t1 = [r["test"]["top1"] for r, _ in items]
        t5 = [r["test"]["top5"] for r, _ in items]
        mc = [r["test"]["mean_class_acc"] for r, _ in items]
        rows.append(f"| {MODEL_NAMES[model]} | {'hands + pose' if pose else 'hands'} | {'yes' if aug else 'no'} | "
                    f"{mean_std(t1)} | {mean_std(t5)} | {mean_std(mc)} | {fmt_params(items[0][0]['params'])} | {len(items)} |")
    rgb = read_json(runs / "wlasl100" / "rgb_timesformer" / "results.json")
    if rgb:
        rows.append(f"| {MODEL_NAMES['rgb_timesformer']} | RGB frames | flip + crop | {pct(rgb['test']['top1'])} | "
                    f"{pct(rgb['test']['top5'])} | {pct(rgb['test']['mean_class_acc'])} | {fmt_params(rgb['params'])} | 1 |")
    table = ("| Model | Input | Augmentation | Top-1 | Top-5 | Mean class acc. | Params | Seeds |\n"
             "|---|---|---|---|---|---|---|---|\n" + ("\n".join(rows) if rows else f"| {NA} | — | — | — | — | — | — | — |"))
    lit = ("\n\n*For context, published WLASL100 top-1 (different pipelines, not re-run here): " +
           "; ".join(f"{n} {v:.1f}% ({c})" for n, v, c in LITERATURE) + ".*")

    # Figure: top-1 by model, augmentation on vs off (hands + pose), mean ± std over seeds.
    models = [m for m in ("bilstm", "gru", "timesformer") if any(k[0] == m for k in groups)]
    if models:
        series, errs = [], []
        for aug, name, color in ((True, "With temporal augmentation", BLUE), (False, "Without", ORANGE)):
            vals, e = [], []
            for m in models:
                t1 = [r["test"]["top1"] for r, _ in groups.get((m, True, aug), [])]
                vals.append(float(np.mean(t1)) if t1 else None)
                e.append(float(np.std(t1)) if len(t1) > 1 else 0.0)
            if any(v is not None for v in vals):
                series.append((name, vals, color))
                errs.append(e)
        save_bars([MODEL_NAMES[m] for m in models], series, "WLASL100 test top-1 (hands + pose, mean ± std over seeds)",
                  assets / "wlasl_models.png", errors=errs)
        figs.append(("wlasl_models.png", "WLASL100 test top-1 by model, with and without temporal augmentation"))

        fig, ax = plt.subplots(figsize=(7, 3.9))
        for m in models:
            items = groups.get((m, True, True)) or next((v for k, v in groups.items() if k[0] == m), [])
            hist = read_json(items[0][1] / "history.json", [])
            if hist:
                ax.plot([h["epoch"] for h in hist], [h["val_top1"] for h in hist], color=MODEL_COLORS[m],
                        linewidth=2, label=MODEL_NAMES[m])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation top-1")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title("Validation top-1 per epoch (seed 0)", loc="left")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(assets / "wlasl_curves.png", dpi=160)
        plt.close(fig)
        figs.append(("wlasl_curves.png", "Validation accuracy per epoch"))
    return data, table + lit, groups, rgb


def deployment_section(runs, groups, rgb, assets, figs):
    rows, points = [], []
    exports = sorted(runs.glob("**/export/export_results.json"))
    for p in exports:
        r = read_json(p)
        run_dir = p.parent.parent
        if r["kind"] == "rgb":
            continue
        cfg = read_json(run_dir / "config.json", {})
        name = MODEL_NAMES.get(cfg.get("model"), "Landmark MLP (static)") if r["kind"] == "sequence" else "Landmark MLP (static)"
        par = r.get("parity", {})
        l1, l4 = r["latency_tflite_1thread"]["median_ms"], r["latency_tflite_4thread"]["median_ms"]
        rows.append(f"| {name} | {fmt_params(r['params'])} | {r['tflite_mb']:.2f} MB | {l1:.2f} ms | {l4:.2f} ms | "
                    f"{r['latency_torch_cpu_1thread']['median_ms']:.2f} ms | "
                    f"{pct(par.get('top1_agreement'), 2)} | {par.get('max_abs_logit_diff', 0):.1e} |")
        res = read_json(run_dir / "results.json", {})
        if r["kind"] == "sequence" and res:
            points.append((name, l1, res["test"]["top1"], MODEL_COLORS.get(cfg.get("model"), BLUE)))
    table = ("| Model | Params | TFLite size | TFLite CPU, 1 thread | 4 threads | PyTorch CPU, 1 thread | "
             "PyTorch vs TFLite top-1 agreement | Max logit diff |\n|---|---|---|---|---|---|---|---|\n" +
             ("\n".join(rows) if rows else f"| {NA} | — | — | — | — | — | — | — |"))

    rgb_exp = read_json(runs / "wlasl100" / "rgb_timesformer" / "export" / "export_results.json")
    if rgb_exp and rgb:
        lat = rgb_exp["latency_torch_cpu_1thread"]["median_ms"]
        table += (f"\n\nRGB TimeSformer for comparison (PyTorch, not exported): {lat:.0f} ms per clip on 1 CPU thread"
                  + (f", {rgb_exp['latency_torch_gpu_fp16']['median_ms']:.1f} ms on GPU (fp16)" if "latency_torch_gpu_fp16" in rgb_exp else "")
                  + f"; {fmt_params(rgb_exp['params'])} parameters.")
        points.append((MODEL_NAMES["rgb_timesformer"], lat, rgb["test"]["top1"], VIOLET))

    demo = read_json(runs / "demo_benchmark.json")
    if demo and "throughput_fps" not in demo:
        print("NOTE: demo_benchmark.json is from an older version; re-run the demo benchmark.")
        demo = None
    if demo:
        table += (f"\n\n**End-to-end real-time pipeline** ({demo['backend']} landmarks + {demo['model_runtime']} model, "
                  f"{demo['frames']} frames, model run every {demo.get('stride', 1)} frames): landmarks "
                  f"{demo['landmarks_ms_median']:.1f} ms per frame, model {(demo['model_ms_median'] or 0):.1f} ms "
                  f"per prediction, so **{demo['throughput_fps']:.1f} FPS** throughput and "
                  f"{(demo['prediction_latency_ms'] or 0):.0f} ms from frame to prediction.")

    if points:
        fig, ax = plt.subplots(figsize=(7, 3.9))
        for name, lat, acc, color in points:
            ax.scatter([lat], [acc], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(name, (lat, acc), xytext=(7, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        lats = [pt[1] for pt in points]
        ax.set_xlim(min(lats) / 2, max(lats) * 4)       # room for the direct labels
        ax.set_xlabel("CPU latency per clip, ms (log scale, 1 thread)")
        ax.set_ylabel("WLASL100 test top-1")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title("Accuracy vs. speed", loc="left")
        fig.tight_layout()
        fig.savefig(assets / "accuracy_vs_latency.png", dpi=160)
        plt.close(fig)
        figs.append(("accuracy_vs_latency.png", "Accuracy versus CPU latency"))
    return table


def build(runs, assets):
    assets.mkdir(parents=True, exist_ok=True)
    figs = []
    det = detector_section(runs)
    static_det, static_acc = static_section(runs, assets, figs)
    wl_data, wl_table, groups, rgb = wlasl_section(runs, assets, figs)
    deploy = deployment_section(runs, groups, rgb, assets, figs)
    fig_md = {n: f"![{c}](assets/{n})\n" for n, c in figs}
    parts = [
        "### 1. Hand detector (YOLO, Ultralytics Hand Keypoints val set)\n", det, "",
        "### 2. Static signs: ASL alphabet (29 classes)\n", static_det, "", static_acc, "",
        fig_md.get("static_ood.png", ""),
        "### 3. Word-level signs: WLASL100 (100 words, official test split)\n", wl_data, "", wl_table, "",
        fig_md.get("wlasl_models.png", ""), fig_md.get("wlasl_curves.png", ""),
        "### 4. Deployment: TensorFlow Lite on CPU\n", deploy, "", fig_md.get("accuracy_vs_latency.png", ""),
    ]
    return "\n".join(parts), figs


def update_readme(readme, block):
    text = readme.read_text()
    if START not in text or END not in text:
        print(f"WARNING: {readme} lacks the {START} / {END} markers; not updated.")
        return False
    readme.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END),
                             lambda _: f"{START}\n{block}\n{END}", text, flags=re.S))
    return True


def run(args):
    runs = Path(args.runs_dir).expanduser()
    assets = Path(args.assets_dir).expanduser()
    block, figs = build(runs, assets)
    out = runs / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.md").write_text("# Results\n\n" + block.replace("](assets/", f"]({assets.resolve()}/"))
    print(block)
    print(f"\n{len(figs)} figure(s) written to {assets}; RESULTS.md in {out}")
    if args.readme and update_readme(Path(args.readme), block):
        print(f"Updated the results block in {args.readme}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--assets-dir", default="assets")
    p.add_argument("--readme", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
