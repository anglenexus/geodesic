#!/usr/bin/env bash
# setup_host_qwen3_27B.sh - one-time setup for serving Qwen3.5-27B on one GPU (A100-40GB, H100 or H200).
#
# Runs setup_host.sh (driver, CUDA 13, uv venv, SGLang + FlashInfer) with the first checkpoint, downloads
# the other weight precisions, then checks what is specific to this model: free disk, SGLang support for
# the Qwen3.5 architecture, each checkpoint's quantization config, and that each one fits this GPU with
# room for the cache.
#
# Usage:  bash setup_host_qwen3_27B.sh      then: ./serve_qwen3_27B.sh start   (or WEIGHTS=fp8 ...)
# Safe to re-run. If setup_host.sh installs a GPU driver, reboot and run this script again.
#
# Weight precisions (WEIGHTS lists which to download; serve_qwen3_27B.sh picks one with WEIGHTS=...):
#   awq   cyankiwi/Qwen3.5-27B-AWQ-4bit  ~20 GB  W4A16, Marlin kernels. The official GPTQ-Int4 is ~30 GB
#                                                (it keeps attention in BF16), too big for 40 GB with cache.
#   fp8   Qwen/Qwen3.5-27B-FP8           ~31 GB  W8A8 with 128x128 block scales; native on Hopper
#   bf16  Qwen/Qwen3.5-27B               ~56 GB  full precision; needs an 80 GB+ GPU
#
# Options (env vars):
#   WEIGHTS="awq fp8 bf16"   default: all three on GPUs with >= 75 GB, "awq" otherwise
#   SGLANG_VERSION=0.5.21    "" = latest; use that if the architecture check fails
#   WORK_DIR=~/qwen-serve    checkpoints go to ~/models/<repo name>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
export SGLANG_VERSION="${SGLANG_VERSION-0.5.21}"
ARCH="Qwen3_5ForConditionalGeneration"
MODELS_DIR="$HOME/models"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }

repo_for() {
  case "$1" in
    awq)  echo cyankiwi/Qwen3.5-27B-AWQ-4bit ;;
    fp8)  echo Qwen/Qwen3.5-27B-FP8 ;;
    bf16) echo Qwen/Qwen3.5-27B ;;
    *)    die "Unknown weight precision '$1' (awq, fp8 or bf16)" ;;
  esac
}
size_gb() { case "$1" in awq) echo 20 ;; fp8) echo 31 ;; bf16) echo 56 ;; esac; }

GPU_GB=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  GPU_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) / 1024 ))
fi
if [[ -z "${WEIGHTS:-}" ]]; then
  if (( GPU_GB >= 75 )); then WEIGHTS="awq fp8 bf16"; else WEIGHTS="awq"; fi
fi
read -r -a LIST <<< "$WEIGHTS"
echo "GPU memory ${GPU_GB} GB; weight precisions to set up: ${LIST[*]}"

# 0. Disk: every checkpoint not yet downloaded, plus ~10 GB for the venv
NEED=10
for w in "${LIST[@]}"; do
  d="$MODELS_DIR/$(basename "$(repo_for "$w")")"
  [[ -f "$d/config.json" ]] || NEED=$(( NEED + $(size_gb "$w") ))
done
mkdir -p "$MODELS_DIR"
FREE_GB=$(df -Pk "$MODELS_DIR" | awk 'NR==2 {print int($4 / 1048576)}')
(( FREE_GB >= NEED )) || die "Only ${FREE_GB} GB free under $MODELS_DIR; need ~${NEED} GB for ${LIST[*]}."

# 1-8. Shared stack + the first checkpoint
export MODEL_ID; MODEL_ID="$(repo_for "${LIST[0]}")"
export MODEL_DIR="$MODELS_DIR/${MODEL_ID##*/}"
bash "$HERE/setup_host.sh"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "setup_host.sh stopped before the model download (driver install?). Reboot if asked, then re-run $0."
  exit 0
fi

# shellcheck disable=SC1091
source "$WORK_DIR/.venv/bin/activate"

# 9. The other checkpoints
for w in "${LIST[@]:1}"; do
  repo="$(repo_for "$w")"; d="$MODELS_DIR/${repo##*/}"
  log "Weights ($w): $repo"
  if [[ -f "$d/config.json" ]]; then echo "Already present at $d"; else hf download "$repo" --local-dir "$d"; fi
done

# 10. SGLang must know the Qwen3.5 architecture
log "SGLang support for $ARCH"
SGL_DIR=$(python -c "import os, sglang; print(os.path.dirname(sglang.__file__))")
if grep -rqs "$ARCH" "$SGL_DIR/srt/models"; then
  echo "$ARCH found in sglang $(python -c 'import sglang; print(sglang.__version__)')"
else
  die "This SGLang has no $ARCH. Re-run with SGLANG_VERSION=\"\" bash $0 to install the latest release."
fi
python -c "import compressed_tensors" 2>/dev/null || uv pip install compressed-tensors

# 11. Each checkpoint against this GPU
log "Checkpoints and GPU memory"
for w in "${LIST[@]}"; do
  python - "$w" "$MODELS_DIR/$(basename "$(repo_for "$w")")" <<'EOF'
import json, pathlib, subprocess, sys
w, d = sys.argv[1], pathlib.Path(sys.argv[2])
cfg = json.loads((d / "config.json").read_text())
q = cfg.get("quantization_config") or {}
expect = {"awq": "compressed-tensors", "fp8": "fp8", "bf16": None}[w]
if q.get("quant_method") != expect:
    sys.exit(f"{w}: expected quant_method {expect}, found {q.get('quant_method')} in {d}")
weights_gb = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 2**30
gpu_gb = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).split()[0]) / 1024
left = gpu_gb * 0.88 - weights_gb  # serve_qwen3_27B.sh uses MEM_FRACTION 0.88
verdict = "OK" if left >= 8 else ("tight: few concurrent requests" if left > 2 else "does NOT fit this GPU")
print(f"{w:5s} {q.get('quant_method') or 'bf16':18s} weights {weights_gb:5.1f} GiB, ~{left:5.1f} GiB left for cache "
      f"on {gpu_gb:.0f} GiB -> {verdict}")
EOF
done

log "Setup complete. Start serving with: WEIGHTS=${LIST[0]} ./serve_qwen3_27B.sh start"
