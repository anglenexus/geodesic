#!/usr/bin/env bash
# One-time setup of a 1x H200 Ubuntu 22.04 box for LoRA SFT + GRPO of Qwen3.6-27B (post_training/PLAN.md).
# The host runs Docker (task sandboxes) and Harbor's Miles agent server; the trainer, SGLang and the
# session server run inside the pinned Miles image. Run as a sudo-capable user. Safe to re-run.
#
#   bash aees/post_training/setup.sh
#
# Afterwards: log out and back in (docker group), `docker login`, `hf auth login`. See the summary at the end.
set -euo pipefail

HARBOR_REF=${HARBOR_REF:-harbor-miles-v0.20.0}                           # Miles' Harbor branch (agent server, TITO fixes)
DATA=${DATA:-/data}                  # models, checkpoints, trials, task images: needs ~400 GB free
MIN_DRIVER=580                       # the Miles image is built on CUDA 13.0
MIN_DISK_GB=400
MODELS=${MODELS:-"Qwen/Qwen3.6-27B Qwen/Qwen3.5-4B"}   # policy + the Phase 1 smoke-test model

HERE=$(cd "$(dirname "$0")" && pwd)          # aees/post_training
source "$HERE/configs/pins.sh"               # MILES_IMAGE, MILES_COMMIT, HARBOR_COMMIT
HARBOR_STATE="$HERE/../harbor/state"         # task lists (training / validation / test)

echo "== 1/9 preflight: OS, CPU arch, GPU driver, disk"
. /etc/os-release
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = "22.04" ] || echo "warning: tested on Ubuntu 22.04, found $PRETTY_NAME"
[ "$(uname -m)" = x86_64 ] || { echo "expected x86_64 (task images are amd64-only)"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: install the NVIDIA driver (>= $MIN_DRIVER) first"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${driver%%.*}" -ge "$MIN_DRIVER" ] || { echo "driver $driver < $MIN_DRIVER (needed for CUDA 13.0)"; exit 1; }
sudo mkdir -p "$DATA" && sudo chown "$USER" "$DATA"
free_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_DISK_GB" ] || echo "warning: $free_gb GB free under $DATA, plan assumes >= $MIN_DISK_GB GB"
mkdir -p "$DATA"/{models,ckpt,trials,tasks,src,data}

echo "== 2/9 system packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git git-lfs gnupg jq python3 tmux

echo "== 3/9 docker engine + daemon config"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"
# Harbor creates one compose network per trial; Docker's default /16 pools run out after ~30 networks.
# Keep task images and containers on the data disk. Same limits as aees/harbor/ec2/setup_ec2.sh.
sudo mkdir -p /etc/docker "$DATA/docker"
sudo tee /etc/docker/daemon.json >/dev/null <<JSON
{
  "data-root": "$DATA/docker",
  "default-address-pools": [{"base": "10.200.0.0/14", "size": 24}],
  "log-driver": "json-file",
  "log-opts": {"max-size": "20m", "max-file": "2"},
  "default-ulimits": {"nofile": {"Name": "nofile", "Hard": 65536, "Soft": 65536}}
}
JSON
sudo tee /etc/sysctl.d/90-harbor.conf >/dev/null <<'CONF'
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
fs.file-max = 2097152
CONF
sudo sysctl --system >/dev/null

echo "== 4/9 NVIDIA container toolkit (GPU inside the Miles container)"
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker    # merges the nvidia runtime into daemon.json
sudo systemctl restart docker
sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L

echo "== 5/9 uv + Harbor ($HARBOR_REF) for the agent server"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d "$DATA/src/harbor/.git" ]; then
  git clone https://github.com/harbor-framework/harbor.git "$DATA/src/harbor"
fi
git -C "$DATA/src/harbor" fetch origin "$HARBOR_REF"
git -C "$DATA/src/harbor" checkout -- agent_server/trial_runner.py 2>/dev/null || true   # drop an earlier patch
git -C "$DATA/src/harbor" checkout --detach "$HARBOR_COMMIT"   # the branch can move; the commit can't
git -C "$DATA/src/harbor" apply "$HERE/grpo/harbor_parser.patch"   # JSON parser via env var; exception type in replies
(cd "$DATA/src/harbor" && uv sync)       # pip can't resolve the branch's uv-workspace dependency

echo "== 6/9 Miles image + pinned Miles source (mounted over /root/miles at launch)"
sudo docker pull "$MILES_IMAGE"
if [ ! -d "$DATA/src/miles/.git" ]; then
  git clone https://github.com/radixark/miles.git "$DATA/src/miles"
fi
git -C "$DATA/src/miles" fetch origin
git -C "$DATA/src/miles" checkout --detach "$MILES_COMMIT"
(cd "$HERE" && uv sync --python 3.12)    # our host-side scripts (export, split, tests)

echo "== 7/9 model weights"
for m in $MODELS; do
  uvx --from huggingface_hub hf download "$m" --local-dir "$DATA/models/${m#*/}"
done

echo "== 8/9 TB 2.1 task definitions + image pre-pull (training, validation, test tasks)"
[ -d "$DATA/tasks/terminal-bench-2-1" ] || \
  (cd "$DATA/src/harbor" && uv run harbor datasets download terminal-bench/terminal-bench-2-1 -o "$DATA/tasks")
cat "$HARBOR_STATE"/tb21_selected_tasks.txt "$HARBOR_STATE"/tb21_val8_tasks.txt \
    "$HARBOR_STATE"/tb21_test20_tasks.txt > "$DATA/tasks/all_used.txt"
python3 "$HERE/../harbor/scripts/task_images.py" "$DATA/tasks/terminal-bench-2-1" "$DATA/tasks/all_used.txt" \
  | sort -u | xargs -r -P 8 -n 1 sudo docker pull -q

echo "== 9/9 HF -> torch_dist checkpoints (native LoRA loads the base from torch_dist)"
for m in $MODELS; do
  name=${m#*/}
  args_name=$(echo "$name" | sed 's/^Qwen//')          # Qwen3.6-27B -> 3.6-27B
  [ -d "$DATA/models/${name}_torch_dist" ] && continue
  sudo docker run --rm --gpus all --ipc=host --shm-size=32g \
    -v "$DATA/models:/root/models" -v "$DATA/src/miles:/root/miles" "$MILES_IMAGE" bash -c "
      set -euo pipefail
      cd /root/miles && pip install -e . --no-deps -q
      read -ra MODEL_ARGS <<< \"\$(python3 miles/utils/external_utils/model_args_utils.py qwen$args_name)\"
      PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \"\${MODEL_ARGS[@]}\" \
        --hf-checkpoint /root/models/$name --save /root/models/${name}_torch_dist"
done

cat <<NEXT

Setup done. Data root: $DATA
  models:  $(ls "$DATA/models" | tr '\n' ' ')
  tasks:   $DATA/tasks/terminal-bench-2-1
  harbor:  $DATA/src/harbor ($HARBOR_REF @ ${HARBOR_COMMIT:0:7})    miles: $DATA/src/miles ($MILES_COMMIT)
Next:
  1. Log out and back in (docker group), then: docker info
  2. docker login                        (avoids Docker Hub pull rate limits)
  3. hf auth login                       (token with write access to the private model repo, for uploads)
  4. bash $HERE/scripts/agent_server.sh   then the Phase 0 baseline (PLAN.md §4)
NEXT
