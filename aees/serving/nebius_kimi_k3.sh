#!/usr/bin/env bash
# nebius_kimi_k3.sh - from the laptop: the 8x B300 Nebius VM for the Kimi K3 tests (on-demand, not preemptible).
#
#   bash nebius_kimi_k3.sh up        # create the VM (~2 min), wait for ssh, copy this folder, start setup_host_kimi_k3.sh
#   bash nebius_kimi_k3.sh status    # VM state, setup/run progress (tail of the logs)
#   bash nebius_kimi_k3.sh sync      # every 2 min: results -> laptop -> S3, and a PREEMPTED line if the VM stops
#   bash nebius_kimi_k3.sh resume    # after a preemption: start the same VM (disk and weights kept), rerun setup
#   bash nebius_kimi_k3.sh tunnel    # ssh -N with :30000 (API) and :8000 (results) forwarded; leave it running
#   bash nebius_kimi_k3.sh fetch     # results -> bench_results/ (export) and a raw copy for the S3 backup
#   bash nebius_kimi_k3.sh down      # delete the VM and its disk (weights included; nothing is kept on Nebius)
#
# Platform gpu-b300-sxm, preset 8gpu-192vcpu-2768gb (8 x B300, 2.7 TiB RAM: the whole 1.56 TB checkpoint fits in the
# page cache) in us-north1, where the tenant has B300 quota. $9.50/GPU-hour on demand: ~$76/h.
# Kill switch: cloud-init schedules `shutdown -h` KILL_MIN (default 150) minutes after boot, so the VM powers off
# even if the laptop goes away. With --recovery-policy fail a guest shutdown leaves the VM stopped (Nebius doesn't
# bill a stopped VM's compute, only its disk) instead of restarting it; `down` then deletes VM and disk.
# Uses the `aees` Nebius CLI profile (service account).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="${STATE:-$HERE/.nebius}"
N=(nebius --profile "${NEBIUS_PROFILE:-aees}")
command -v nebius >/dev/null || N[0]=$HOME/.nebius/bin/nebius
PROJECT="${PROJECT:-project-u02n20nqig001ntg5jdh7b}"   # default-project-us-north1 (B300 quota 8)
SUBNET="${SUBNET:-vpcsubnet-u02n7ccew5fnhz00nw}"
NAME="${NAME:-kimi-k3-b300-$(date +%m%d)}"
SSH_USER="${SSH_USER:-willzou}"
PUBKEY="${PUBKEY:-$HOME/.ssh/id_rsa.pub}"
DISK_GB="${DISK_GB:-2560}"        # weights 1.56 TB + SGLang image 16 GB + results
KILL_MIN="${KILL_MIN:-150}"       # total VM time across resumes (setup_host_kimi_k3.sh enforces it)
PREEMPTIBLE="${PREEMPTIBLE:-1}"   # B300 on Nebius is preemptible-only today; preemption STOPS the VM, keeps its disk
SYNC_SEC="${SYNC_SEC:-120}"
S3="${S3:-s3://aees-grpo-backups-611728564241/serving/kimi-k3/nebius-$(date +%Y%m%d)}"
mkdir -p "$STATE"
[[ -f "$STATE/vm.env" ]] && source "$STATE/vm.env"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -x)

up() {
  [[ -n "${VM_ID:-}" ]] && { echo "VM already recorded ($VM_ID at $VM_IP); run: $0 status, or $0 down first"; exit 1; }
  local user_data PRE=()
  (( PREEMPTIBLE )) && PRE=(--preemptible-on-preemption stop --follows-spot-price)
  user_data=$(printf '#cloud-config\nusers:\n  - name: %s\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n      - %s\nruncmd:\n  - [shutdown, -h, "+%s", "kimi-k3 kill switch"]\n' \
    "$SSH_USER" "$(head -1 "$PUBKEY")" "$KILL_MIN")
  echo "Creating $NAME: gpu-b300-sxm 8gpu-192vcpu-2768gb, ${DISK_GB} GiB disk, kill switch at ${KILL_MIN} min"
  "${N[@]}" compute instance create --parent-id "$PROJECT" --name "$NAME" \
    --resources-platform gpu-b300-sxm --resources-preset 8gpu-192vcpu-2768gb \
    --recovery-policy fail ${PRE[@]+"${PRE[@]}"} \
    --boot-disk-managed-disk-name "$NAME-boot" --boot-disk-managed-disk-type network_ssd \
    --boot-disk-managed-disk-size-gibibytes "$DISK_GB" \
    --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 --boot-disk-attach-mode READ_WRITE \
    --network-interfaces "[{\"name\": \"eth0\", \"ip_address\": {}, \"public_ip_address\": {}, \"subnet_id\": \"$SUBNET\"}]" \
    --cloud-init-user-data "$user_data" > "$STATE/instance.yaml"
  VM_ID=$(grep -m1 -oE 'computeinstance-[a-z0-9]+' "$STATE/instance.yaml")
  VM_IP=$(sed -n '/public_ip_address:/,/address:/p' "$STATE/instance.yaml" | grep -oE '[0-9]+(\.[0-9]+){3}' | tail -1)
  printf 'VM_ID=%s\nVM_IP=%s\nVM_CREATED=%s\n' "$VM_ID" "$VM_IP" "$(date +%s)" > "$STATE/vm.env"
  echo "Created $VM_ID at $VM_IP; waiting for ssh"
  for _ in $(seq 60); do
    "${SSH[@]}" -o BatchMode=yes "$SSH_USER@$VM_IP" \
      'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sort | uniq -c' 2>/dev/null && break
    sleep 10
  done
  rsync -az --exclude bench_results --exclude '*.pdf' --exclude __pycache__ --exclude .nebius \
    "$HERE/" "$SSH_USER@$VM_IP:serving/"
  start_setup
}

start_setup() {
  "${SSH[@]}" "$SSH_USER@$VM_IP" "KILL_MIN=$KILL_MIN nohup bash serving/setup_host_kimi_k3.sh >> setup.log 2>&1 &"
  echo "Setup started on the VM (log: ~/setup.log). Next: $0 sync (background), $0 status, $0 tunnel"
}

vm_state() {
  "${N[@]}" compute instance get --id "$VM_ID" --format json 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',{}).get('state','UNKNOWN'))" 2>/dev/null || echo GONE
}

resume() {
  [[ -n "${VM_ID:-}" ]] || { echo "No VM recorded"; exit 1; }
  echo "VM state: $(vm_state); starting it again (disk with weights and results is kept)"
  "${N[@]}" compute instance start --id "$VM_ID" </dev/null
  VM_IP=$("${N[@]}" compute instance get --id "$VM_ID" --format json \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status']['network_interfaces'][0]['public_ip_address']['address'].split('/')[0])")
  sed -i '' "s/^VM_IP=.*/VM_IP=$VM_IP/" "$STATE/vm.env"
  for _ in $(seq 60); do "${SSH[@]}" -o BatchMode=yes "$SSH_USER@$VM_IP" true 2>/dev/null && break; sleep 10; done
  rsync -az --exclude bench_results --exclude '*.pdf' --exclude __pycache__ --exclude .nebius "$HERE/" "$SSH_USER@$VM_IP:serving/"
  start_setup
}

sync_loop() {
  local raw="$STATE/results_raw" last=""
  mkdir -p "$raw"
  while true; do
    st=$(vm_state)
    if [[ "$st" == RUNNING ]]; then
      rsync -az --timeout=60 "$SSH_USER@$VM_IP:/data/results/" "$raw/" 2>/dev/null
      rsync -az --timeout=60 "$SSH_USER@$VM_IP:setup.log" "$raw/" 2>/dev/null
      aws s3 sync "$raw" "$S3" --only-show-errors 2>&1 | tail -2
    fi
    [[ "$st" != "$last" ]] && echo "[$(date +%H:%M:%S)] VM $st$([[ $st == STOPPED || $st == STOPPING ]] && echo ' -> PREEMPTED or kill switch; resume with: bash nebius_kimi_k3.sh resume')"
    [[ "$st" == GONE ]] && { echo "VM deleted; last sync in $raw and $S3"; return; }
    last=$st
    sleep "$SYNC_SEC"
  done
}

status() {
  [[ -n "${VM_ID:-}" ]] || { echo "No VM recorded in $STATE/vm.env"; return; }
  "${N[@]}" compute instance get --id "$VM_ID" --format json 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('VM', d['metadata']['name'], d.get('status',{}).get('state'))" || true
  echo "Created $(( ($(date +%s) - VM_CREATED) / 60 )) min ago (kill switch: $KILL_MIN min of VM time); state $(vm_state)"
  "${SSH[@]}" "$SSH_USER@$VM_IP" 'tail -n 8 setup.log; ls -t /data/results/logs/run_*.log 2>/dev/null | head -1 | xargs -r tail -n 6' || true
}

tunnel() {
  echo "API http://127.0.0.1:30000/v1 (python smoke_test.py --model kimi-k3 --no-think)   results http://127.0.0.1:8000/"
  exec "${SSH[@]}" -N -L 30000:127.0.0.1:30000 -L 8000:127.0.0.1:8000 "$SSH_USER@$VM_IP"
}

fetch() {
  local raw="$STATE/results_raw"
  mkdir -p "$raw"
  rsync -az "$SSH_USER@$VM_IP:/data/results/" "$raw/"
  rsync -az "$raw/export/" "$HERE/bench_results/"
  echo "Raw copy: $raw ($(du -sh "$raw" | cut -f1)); exported folders now in bench_results/:"
  ls -t "$HERE/bench_results" | head -5
}

down() {
  [[ -n "${VM_ID:-}" ]] || { echo "No VM recorded"; return; }
  echo "Deleting $VM_ID (and its boot disk with the weights)"
  "${N[@]}" compute instance delete --id "$VM_ID" </dev/null
  "${N[@]}" compute disk list --parent-id "$PROJECT" --format json 2>/dev/null \
    | python3 -c "import json,sys; print('disks left in project:', [d['metadata']['name'] for d in json.load(sys.stdin).get('items',[])])" || true
  mv "$STATE/vm.env" "$STATE/vm.env.deleted-$(date +%Y%m%d-%H%M%S)"
}

case "${1:-}" in
  up) up ;; status) status ;; tunnel) tunnel ;; fetch) fetch ;; down) down ;; resume) resume ;; sync) sync_loop ;;
  ssh) exec "${SSH[@]}" "$SSH_USER@$VM_IP" ;;
  *) echo "Usage: $0 {up|status|tunnel|ssh|fetch|down}"; exit 1 ;;
esac
