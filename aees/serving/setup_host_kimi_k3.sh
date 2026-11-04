#!/usr/bin/env bash
# setup_host_kimi_k3.sh - on the 8x B300 VM (started by nebius_kimi_k3.sh up): Docker + the NVIDIA container
# toolkit, the same SGLang image as the Modal runs, then Phase 4 kernels while the weights download, then the
# rest of run_kimi_k3.sh (server, Phase 3, export, hold) once the weights are complete.
#
# Safe to rerun after a preemption (nebius_kimi_k3.sh resume): every step skips or resumes what is done.
# Kill switch: the first run writes a deadline (KILL_MIN from then) to /data; every run powers the VM off at it.
#
# Layout: /data/models/Kimi-K3 (weights), /data/results (results, logs; served on :8000), ~/serving (scripts).
# Inside the container they are /models, /results and /root/serving, the same paths as on Modal.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE_BASE="lmsysorg/sglang:v0.5.23-cu130"
IMAGE="kimi-sglang:v0.5.23"
HOLD_MIN="${HOLD_MIN:-15}"
stamp() { printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

stamp "Host: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | xargs)"
sudo mkdir -p /data/models /data/results && sudo chown -R "$USER" /data
[[ -f /data/kill_deadline ]] || echo $(( $(date +%s) + ${KILL_MIN:-150} * 60 )) > /data/kill_deadline
left=$(( ($(cat /data/kill_deadline) - $(date +%s)) / 60 ))
(( left > 0 )) || { echo "Kill-switch deadline passed; powering off"; sudo shutdown -h now; exit 1; }
sudo shutdown -c 2>/dev/null; sudo shutdown -h "+$left" "kimi-k3 kill switch"
stamp "Kill switch: power-off in $left min"

stamp "Docker + NVIDIA container toolkit"
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sudo sh; fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y -q && sudo apt-get install -y -q nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null && sudo systemctl restart docker
D=(sudo docker)

stamp "Image $IMAGE (from $IMAGE_BASE + tmux, pandas, plotly)"
"${D[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || printf 'FROM %s\nRUN apt-get update -q && apt-get install -y -q tmux curl && rm -rf /var/lib/apt/lists/*\nRUN pip install -q pandas plotly\n' \
  "$IMAGE_BASE" | "${D[@]}" build -q -t "$IMAGE" - || { echo "image build failed"; exit 1; }

RUN=("${D[@]}" run --rm --gpus all --ipc=host --network host --shm-size 64g --ulimit memlock=-1
     -v /data/models:/models -v /data/results:/results -v "$HERE":/root/serving -w /root/serving
     -e PYTHONUNBUFFERED=1 -e HF_XET_HIGH_PERFORMANCE=1 "$IMAGE")

stamp "Weights download (background) and Phase 4 kernels (GPUs idle) at the same time"
"${RUN[@]}" python -c "
import time; from huggingface_hub import snapshot_download
t = time.time()
snapshot_download('moonshotai/Kimi-K3', local_dir='/models/Kimi-K3', max_workers=24,
                  ignore_patterns=['.eval_results/*', 'assets/*'])
print(f'weights done in {(time.time() - t) / 60:.1f} min', flush=True)
" > /data/results/download.log 2>&1 &
dl=$!
if ls /data/results/results/fi_kimi-k3_*/fp4.json >/dev/null 2>&1; then
  echo "Phase 4 results already on disk; skipping the kernels"
else
  "${RUN[@]}" bash run_kimi_k3.sh kernels
fi
wait $dl || { echo "weights download failed:"; tail -20 /data/results/download.log; exit 1; }
tail -1 /data/results/download.log
du -sh /data/models/Kimi-K3

stamp "Server, Phase 3, export, hold (${HOLD_MIN} min)"
# Named, so the hold window can be ended early: sudo docker exec kimi-run touch /tmp/STOP
"${D[@]}" rm -f kimi-run >/dev/null 2>&1
"${RUN[@]:0:3}" --name kimi-run "${RUN[@]:3}" -e HOLD_MIN="$HOLD_MIN" bash run_kimi_k3.sh serve,phase3,export,hold
stamp "Done. From the laptop: bash nebius_kimi_k3.sh fetch, then bash nebius_kimi_k3.sh down"
