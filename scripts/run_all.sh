#!/usr/bin/env bash
# Full pipeline on a local GPU: data -> hand detector -> landmarks -> all models -> TFLite -> report.
#
#   tmux new -s slr          # so training survives closing the terminal
#   bash scripts/run_all.sh  # re-run the same command after any interruption
#
# Every step skips work that's already done and training resumes from its last epoch, so the
# script is safe to re-run at any time. Logs: $RUNS_DIR/logs/<step>.log
#
# Settings (environment variables):
#   DATA_DIR   datasets and model files        (default ~/data/slr)
#   RUNS_DIR   checkpoints, caches, results    (default ~/runs/slr)
#   SEEDS      seeds per sequence-model config (default "0 1 2")
#   BATCH_SIZE batch size for landmark models  (default 32; lower it if the GPU is shared)
#   WORKERS    CPU processes for extraction    (default 4)
#   RUN_RGB    1 = also fine-tune the pixel TimeSformer baseline (default 1)
#   ONLY       run a single step by name, e.g. ONLY=report

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-$HOME/data/slr}"
RUNS_DIR="${RUNS_DIR:-$HOME/runs/slr}"
SEEDS="${SEEDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
RUN_RGB="${RUN_RGB:-1}"
ONLY="${ONLY:-}"
PY="${PYTHON:-python}"
LOG_DIR="$RUNS_DIR/logs"
mkdir -p "$DATA_DIR" "$RUNS_DIR" "$LOG_DIR"

LM_IMG="$RUNS_DIR/landmarks/asl_alphabet"
LM_VID="$RUNS_DIR/landmarks/wlasl100"
YOLO="$RUNS_DIR/hand_detector/weights/best.pt"
SEQ="$RUNS_DIR/wlasl100"

step() {                      # step <name> <command...>
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
  echo "==> [$(date +%H:%M:%S)] $name"
  if ! "$@" > >(tee -a "$LOG_DIR/$name.log") 2>&1; then
    echo "!!! Step '$name' failed. Full log: $LOG_DIR/$name.log"
    echo "    Fix the problem, then re-run this script: finished steps are skipped."
    exit 1
  fi
}

gpu_check() {
  "$PY" -c "from slr.common import gpu_memory_report as g; print(g())"
  df -h "$RUNS_DIR" | tail -1 | awk '{print "Free disk at RUNS_DIR: " $4 " (about 12 GB needed)"}'
}

seq_runs() {                  # every sequence-model configuration x seed
  for model in bilstm gru timesformer; do
    for seed in $SEEDS; do
      for cfg in "pose aug" "pose noaug" "hands aug"; do
        set -- $cfg
        local flags=()
        [[ "$1" == "hands" ]] && flags+=(--no-pose)
        [[ "$2" == "noaug" ]] && flags+=(--no-augment)
        local tag="${model}_${1}_${2}_s${seed}"
        step "train_${tag}" "$PY" -m slr.train_sequence --index "$LM_VID/index.json" --model "$model" \
          --out "$SEQ/$tag" --seed "$seed" --batch-size "$BATCH_SIZE" --resume "${flags[@]}"
      done
    done
  done
}

best_sequence_model() {       # chosen on validation accuracy (never on test)
  "$PY" - "$SEQ" << 'EOF'
import json, sys, glob, statistics as st
best = {}
for p in glob.glob(f"{sys.argv[1]}/*_pose_aug_s*/results.json"):
    r = json.load(open(p))
    best.setdefault(r["model"], []).append(r["best_val_top1"])
print(max(best, key=lambda m: st.mean(best[m])) if best else "timesformer")
EOF
}

step env_check gpu_check
step download "$PY" -m slr.data.download --data-dir "$DATA_DIR"
step hand_detector "$PY" -m slr.train_hand_detector --data-dir "$DATA_DIR" --out "$RUNS_DIR/hand_detector" \
  --workers "$WORKERS"

# Static signs (ASL alphabet images)
step extract_images "$PY" -m slr.extract_images --asl-dir "$DATA_DIR/asl_alphabet" \
  --ood-dir "$DATA_DIR/asl_alphabet_test" --models-dir "$DATA_DIR/models" --yolo-weights "$YOLO" --out "$LM_IMG"
step static_mlp_mediapipe "$PY" -m slr.train_static --kind mlp --landmarks "$LM_IMG/mediapipe.npz" \
  --out "$RUNS_DIR/static/mlp_mediapipe" --resume
step static_mlp_yolo "$PY" -m slr.train_static --kind mlp --landmarks "$LM_IMG/yolo_mediapipe.npz" \
  --out "$RUNS_DIR/static/mlp_yolo_mediapipe" --resume
step static_cnn "$PY" -m slr.train_static --kind cnn --manifest "$LM_IMG/manifest.json" \
  --out "$RUNS_DIR/static/cnn" --amp --resume

# Word-level signs (WLASL100 videos)
step extract_videos "$PY" -m slr.extract_videos --wlasl-dir "$DATA_DIR/wlasl" --num-classes 100 \
  --models-dir "$DATA_DIR/models" --yolo-weights "$YOLO" --out "$LM_VID" --workers "$WORKERS" \
  --rgb-frames 16 --compare-backend mediapipe --compare-n 150
seq_runs
if [[ "$RUN_RGB" == "1" ]]; then
  step train_rgb "$PY" -m slr.train_rgb --index "$LM_VID/index.json" --out "$SEQ/rgb_timesformer" --amp --resume
fi

# Deployment: TFLite export, parity, latency; then a real-time pipeline benchmark
EXPORT_RUNS=("$SEQ/bilstm_pose_aug_s0" "$SEQ/gru_pose_aug_s0" "$SEQ/timesformer_pose_aug_s0"
             "$RUNS_DIR/static/mlp_yolo_mediapipe")
[[ "$RUN_RGB" == "1" ]] && EXPORT_RUNS+=("$SEQ/rgb_timesformer")
step export_tflite "$PY" -m slr.export_tflite --runs "${EXPORT_RUNS[@]}" --index "$LM_VID/index.json" \
  --static-landmarks "$LM_IMG/yolo_mediapipe.npz" --onnx
BEST="$(best_sequence_model)"
CLIP="$("$PY" -c "import json; print(json.load(open('$LM_VID/index.json'))['splits']['test'][0]['path'])")"
step demo_benchmark "$PY" -m slr.demo --model "$SEQ/${BEST}_pose_aug_s0/export/model.tflite" \
  --models-dir "$DATA_DIR/models" --yolo-weights "$YOLO" --source "$CLIP" --no-display \
  --benchmark 300 --bench-out "$RUNS_DIR/demo_benchmark.json"

step report "$PY" -m slr.report --runs-dir "$RUNS_DIR" --assets-dir assets --readme README.md
echo "All done. README.md and assets/ are updated; review with: git diff README.md"
