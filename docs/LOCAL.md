# Running locally

The whole pipeline runs on one machine with an NVIDIA GPU. It was developed for an RTX A4000 (16 GB)
under WSL2 with Ubuntu 24.04, and should work on any Linux GPU box with Python 3.10–3.12.

## 1. One-time setup

Keep the repo and data on the **Linux filesystem** (e.g. `~/sign_language_recognition`), never under
`/mnt/c/...`. Reading thousands of video frames from the Windows drive inside WSL is many times slower.

```bash
nvidia-smi                       # must list your GPU; if not, install the Windows NVIDIA driver with WSL support
sudo apt install -y python3-venv tmux git

git clone https://github.com/tajammulbasheer/sign_language_recognition.git
cd sign_language_recognition
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-export.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest                           # ~15 s, checks the install
```

If `torch.cuda.is_available()` is False, reinstall PyTorch with the CUDA build matching your
driver from [pytorch.org](https://pytorch.org/get-started/locally/).

**Kaggle token** (needed for the three datasets): on kaggle.com go to Settings → API → *Create New
Token*, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

(From WSL, your Windows Downloads folder is `/mnt/c/Users/<you>/Downloads`.)

**Disk:** about 12 GB in total: 6.5 GB datasets, 3–4 GB RGB frame cache, plus checkpoints.

## 2. Run everything

```bash
tmux new -s slr                  # training keeps running if you close the terminal
source .venv/bin/activate
bash scripts/run_all.sh
```

Detach with `Ctrl+B` then `D`; reattach later with `tmux attach -t slr`.

**Watching progress** (in another terminal):

```bash
ls -t ~/runs/slr/logs | head -3              # which step is running
tail -f ~/runs/slr/logs/<step>.log           # its output
watch -n 2 nvidia-smi                        # GPU use and memory
```

**After any interruption** (closed laptop, crash, reboot, out-of-memory), just run
`bash scripts/run_all.sh` again. Finished steps are skipped, extraction continues from the next
uncached video, and training continues from its last saved epoch.

### Settings

Set these as environment variables, e.g. `SEEDS="0" BATCH_SIZE=16 bash scripts/run_all.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `~/data/slr` | datasets and MediaPipe models |
| `RUNS_DIR` | `~/runs/slr` | caches, checkpoints, results, logs |
| `SEEDS` | `0 1 2` | seeds per sequence-model configuration (fewer = faster, but no ± std) |
| `BATCH_SIZE` | `32` | landmark-model batch size; lower it if another process is using the GPU |
| `WORKERS` | `4` | CPU processes for landmark extraction |
| `RUN_RGB` | `1` | `0` skips the pixel TimeSformer baseline (saves ~1 h and ~4 GB) |
| `ONLY` | — | run one step by name, e.g. `ONLY=report bash scripts/run_all.sh` |

### What runs, and roughly how long

These are **estimates** for an RTX A4000 and a recent 8-core CPU; your times will differ.

| Step | What | Rough time |
|---|---|---|
| `download` | ~6.5 GB from Kaggle + MediaPipe models | 10–40 min (your connection) |
| `hand_detector` | YOLO11n-pose, 30 epochs on 18.8k images | 45–90 min |
| `extract_images` | landmarks for ~18k alphabet images, two backends | 15–30 min (CPU) |
| `static_*` | two landmark MLPs + one CNN | 15–25 min |
| `extract_videos` | landmarks for ~1.5k WLASL100 clips, plus the backend comparison | 20–40 min (CPU) |
| `train_*` | 3 models × 3 configs × 3 seeds = 27 runs | 2–4 h |
| `train_rgb` | pixel TimeSformer, 15 epochs | ~1 h |
| `export_tflite`, `demo_benchmark`, `report` | TFLite, latency, README | ~10 min |

In total it's an overnight run. The single biggest saving is `SEEDS="0"`, which cuts the sequence
training to a third (but the README then has no ± std).

## 3. After it finishes

`README.md` and `assets/` in the repo are updated automatically by the `report` step. Review and commit:

```bash
git diff --stat
git add README.md assets/
git commit -m "Add results from local training run"
git push
```

## 4. Live demo

WSL can't see USB webcams by default, so the webcam demo runs from **Windows** Python. Inside WSL,
use a video file instead.

```bash
# Inside WSL: annotate a video file
python -m slr.demo --model ~/runs/slr/wlasl100/timesformer_pose_aug_s0/export/model.tflite \
    --models-dir ~/data/slr/models --yolo-weights ~/runs/slr/hand_detector/weights/best.pt \
    --source some_clip.mp4 --output annotated.mp4 --no-display
```

On Windows: install Python 3.12, `pip install mediapipe ultralytics opencv-python ai-edge-litert`,
then copy over the `slr/` folder, the exported `model.tflite` with its `model_meta.json`, the YOLO
`best.pt`, and the two MediaPipe `.task` files. Run the same command with `--source 0`. The
`export/` folders also contain `model.onnx` if you prefer ONNX Runtime.

For alphabet letters, point `--model` at `~/runs/slr/static/mlp_yolo_mediapipe/export/model.tflite`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | `BATCH_SIZE=16 bash scripts/run_all.sh` (another process may hold GPU memory: check `nvidia-smi`) |
| Kaggle `401` / `403` | token missing or wrong: re-download `kaggle.json` into `~/.kaggle/` |
| MediaPipe model download fails | download the two URLs in `slr/data/download.py` in a browser into `~/data/slr/models/` |
| Extraction very slow | the repo or data is under `/mnt/c`; move both into the Linux home folder |
| `--resume with different settings` | you changed a setting mid-run; use the original settings, or delete that run's folder |
| A step failed | its log is in `~/runs/slr/logs/<step>.log`; fix it, then re-run `run_all.sh` |
| `litert-torch` install fails | it needs Linux and Python 3.10–3.12; everything except `export_tflite` works without it |
