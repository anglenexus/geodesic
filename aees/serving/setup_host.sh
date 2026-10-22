#!/usr/bin/env bash
# setup_host.sh - Phases 1-2: turn a fresh Ubuntu 22.04 NVIDIA GPU host into a Qwen serving
# stack: R580+ driver, CUDA 13 toolkit, uv venv, SGLang + version-matched FlashInfer wheels,
# model weights, and a FlashInfer correctness check. Every host (A100, H200) uses this one stack.
#
# Usage:  bash setup_host.sh
# Safe to re-run: finished steps are skipped. If it has to install a GPU driver it
# stops and asks you to reboot, then run it again.
#
# Options (env vars):
#   MODEL_ID=Qwen/Qwen3-4B-Instruct-2507   model to download
#   SGLANG_VERSION=0.5.21                  version validated in the plan; "" = latest
#   WORK_DIR=~/qwen-serve                  venv + logs live here
#   UPGRADE_DRIVER=1                       replace a pre-R580 driver (CUDA 13 needs R580+)
#   HF_TOKEN=...                           only for gated models
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/${MODEL_ID##*/}}"
WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
SGLANG_VERSION="${SGLANG_VERSION-0.5.21}"
CUDA_PKG="cuda-toolkit-13-0"
CUDA_HOME_DIR="/usr/local/cuda-13.0"
MIN_DRIVER=580
UBUNTU=22.04

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }

# 0. OS check: one stack everywhere (Ubuntu 22.04, NVIDIA's ubuntu2204 packages)
. /etc/os-release
if [[ "${VERSION_ID:-}" != "$UBUNTU" ]]; then
  die "This stack targets Ubuntu $UBUNTU; this host is ${PRETTY_NAME:-unknown}. Re-image it with Ubuntu $UBUNTU."
fi
REPO_ARCH=x86_64
if [[ "$(uname -m)" == "aarch64" ]]; then REPO_ARCH=sbsa; fi

# 1. Base packages (Python 3.12 comes from uv, headers included, so no python3.12-dev here)
log "Base packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git curl wget \
  htop tmux jq ca-certificates
# Kernel headers are only needed to build a new GPU driver; some cloud kernels have none in the archive.
HEADERS="linux-headers-$(uname -r)"
if apt-cache policy "$HEADERS" 2>/dev/null | grep -q "Candidate: [0-9]"; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$HEADERS"
else
  warn "No $HEADERS package for this kernel; fine unless a driver has to be installed."
fi

# NVIDIA's apt repository for Ubuntu 22.04: R580 driver, fabric manager and CUDA 13 toolkit
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  wget -qO /tmp/cuda-keyring.deb \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -y
fi

install_driver() {
  # Driver and fabric manager must be the exact same R580 version; HGX (NVSwitch) boxes need the latter.
  if [[ ! -d "/lib/modules/$(uname -r)/build" ]]; then
    die "Can't build an R580 driver: no kernel headers for $(uname -r). Either pick a provider image that
       already has driver >= 580 (e.g. a CUDA 13 / R580 image), or move to Ubuntu's generic kernel:
       sudo apt-get install -y linux-generic && sudo reboot   # then re-run with UPGRADE_DRIVER=1"
  fi
  local v
  v=$(apt-cache madison nvidia-driver-580-open | awk '{print $3}' | head -1)
  [[ -n "$v" ]] || die "nvidia-driver-580-open not found in NVIDIA's ubuntu2204 repository."
  local pkgs=("nvidia-driver-580-open=$v")
  if systemctl list-unit-files 2>/dev/null | grep -q nvidia-fabricmanager || ls /dev/nvidia-nvswitch* >/dev/null 2>&1; then
    pkgs+=("nvidia-fabricmanager=$v")
  fi
  echo "Installing ${pkgs[*]}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades "${pkgs[@]}"
  [[ " ${pkgs[*]} " == *fabricmanager* ]] && sudo systemctl enable nvidia-fabricmanager
  echo
  echo "Driver $v installed. Reboot now (sudo reboot), then run this script again."
  exit 0
}

# 2. NVIDIA driver (R580+ is required by CUDA 13)
log "NVIDIA driver"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if (( ${DRV%%.*} < MIN_DRIVER )); then
    [[ "${UPGRADE_DRIVER:-0}" == 1 ]] || die "Driver $DRV is older than R$MIN_DRIVER. Re-run with UPGRADE_DRIVER=1 to replace it with R580, then reboot."
    install_driver
  fi
  echo "Driver $DRV OK"
else
  install_driver
fi

# 3. CUDA 13 toolkit (provides nvcc for FlashInfer's JIT path)
log "CUDA 13 toolkit"
if [[ -x "$CUDA_HOME_DIR/bin/nvcc" ]]; then
  echo "Found $("$CUDA_HOME_DIR/bin/nvcc" --version | tail -1)"
else
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$CUDA_PKG"
fi
ENV_SNIPPET="export CUDA_HOME=$CUDA_HOME_DIR; export PATH=\$CUDA_HOME/bin:\$PATH; export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
grep -qF "CUDA_HOME=$CUDA_HOME_DIR" ~/.bashrc || echo "$ENV_SNIPPET" >> ~/.bashrc
eval "$ENV_SNIPPET"

# 4. GPU settings
log "GPU settings"
sudo nvidia-smi -pm 1 >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version,mig.mode.current --format=csv
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -qi enabled; then
  warn "MIG is enabled; disable it to give the server the whole GPU."
fi

# 5. uv + Python 3.12 venv
log "Python environment in $WORK_DIR"
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORK_DIR/logs"
cd "$WORK_DIR"
[[ -d .venv ]] || uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

# 6. SGLang + FlashInfer, with kernel wheels pinned to the flashinfer-python SGLang pulls in
log "SGLang ${SGLANG_VERSION:-latest} + FlashInfer"
uv pip install --prerelease=allow "sglang${SGLANG_VERSION:+==$SGLANG_VERSION}"
FI_VER=$(python -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
echo "flashinfer-python $FI_VER"
# Remove flashinfer* packages at any other version (e.g. leftovers of an unpinned install)
STRAY=$(uv pip list --format=json | python -c "
import json, sys
v = sys.argv[1]
print(' '.join(p['name'] for p in json.load(sys.stdin)
               if p['name'].startswith('flashinfer') and p['name'] != 'flashinfer-python'
               and not p['version'].startswith(v)))" "$FI_VER")
if [[ -n "$STRAY" ]]; then
  warn "Removing mismatched packages: $STRAY"
  # shellcheck disable=SC2086
  uv pip uninstall $STRAY
fi
uv pip install "flashinfer-cubin==$FI_VER" --index-url https://flashinfer.ai/whl
uv pip install "flashinfer-jit-cache==$FI_VER" --index-url https://flashinfer.ai/whl/cu130 \
  || warn "No flashinfer-jit-cache $FI_VER for cu130; kernels will JIT-compile on first use instead."
uv pip install openai "huggingface_hub[cli]" ninja pandas plotly  # ninja: FlashInfer JIT; pandas/plotly: reports
uv pip list 2>/dev/null | grep -i '^flashinfer' || true

# 7. Model weights
log "Model weights: $MODEL_ID"
if [[ -f "$MODEL_DIR/config.json" ]]; then
  echo "Already present at $MODEL_DIR"
else
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

# 8. Verify the stack and FlashInfer kernels (GQA decode at Qwen3-4B shapes vs a PyTorch reference)
log "Verify"
python -c "import torch, flashinfer, sglang; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| flashinfer', flashinfer.__version__, '| sglang', sglang.__version__)"
python - <<'EOF'
import torch, flashinfer
q = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(4096, 8, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn_like(k)
o = flashinfer.single_decode_with_kv_cache(q, k, v)
kr, vr = k.repeat_interleave(4, dim=1).float(), v.repeat_interleave(4, dim=1).float()
p = (torch.einsum("hd,lhd->hl", q.float(), kr) / 128**0.5).softmax(-1)
err = (o.float() - torch.einsum("hl,lhd->hd", p, vr)).abs().max().item()
print(f"FlashInfer GQA decode check: max abs err {err:.2e}")
assert err < 5e-2, "FlashInfer output does not match the PyTorch reference"
EOF

log "Setup complete. Start serving with: ./serve.sh start"
