#!/usr/bin/env bash
# From the laptop: create the H200 with the Nebius CLI (service-account profile), then print its IP for kickoff.sh.
#
#   bash scripts/nebius_up.sh              # ~1 min to RUNNING, ~1 more for ssh; writes state/nebius.env
#
# One-time profile (summary.md, step 0):
#   nebius profile create aees --endpoint api.nebius.cloud --service-account-id <serviceaccount-...> \
#     --public-key-id <publickey-...> --private-key-file-path <key.pem> --tenant-id <tenant-...> --parent-id <project-...>
# H200s are in us-central1 only (the tenant's us-north1 project has B300 and CPU platforms). The image family
# ubuntu24.04-cuda13.0 ships driver 580, which the Miles image needs (CUDA 13.0).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
STATE=${STATE:-$HERE/state}
N=(nebius --profile "${NEBIUS_PROFILE:-aees}")
command -v nebius >/dev/null || N[0]=$HOME/.nebius/bin/nebius
PROJECT=${PROJECT:-project-u00wkf1akc00c0hjpbz059}       # default-project-us-central1
SUBNET=${SUBNET:-vpcsubnet-u00zdx57rq9phnvedk}
NAME=${NAME:-aees-h200-$(date +%m%d)}
SSH_USER=${SSH_USER:-willzou1211}
PUBKEY=${PUBKEY:-$HOME/.ssh/id_rsa.pub}
DISK_GB=${DISK_GB:-1536}                                 # weights 52 GB, Miles image, 4.5 GB per checkpoint, trials
mkdir -p "$STATE"
# Only the first key: ~/.ssh/id_rsa.pub here holds two lines, and a bare second line breaks the YAML.
user_data=$(printf '#cloud-config\nusers:\n  - name: %s\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n      - %s\n' \
  "$SSH_USER" "$(head -1 "$PUBKEY")")
"${N[@]}" compute instance create --parent-id "$PROJECT" --name "$NAME" \
  --resources-platform gpu-h200-sxm --resources-preset 1gpu-16vcpu-200gb \
  --boot-disk-managed-disk-name "$NAME-boot" --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes "$DISK_GB" \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 --boot-disk-attach-mode READ_WRITE \
  --network-interfaces "[{\"name\": \"eth0\", \"ip_address\": {}, \"public_ip_address\": {}, \"subnet_id\": \"$SUBNET\"}]" \
  --cloud-init-user-data "$user_data" > "$STATE/nebius_instance.yaml"
id=$(grep -m1 -oE 'computeinstance-[a-z0-9]+' "$STATE/nebius_instance.yaml")
ip=$(sed -n '/public_ip_address:/,/address:/p' "$STATE/nebius_instance.yaml" | grep -oE '[0-9]+(\.[0-9]+){3}' | tail -1)
printf 'NEBIUS_INSTANCE_ID=%s\nH200_IP=%s\nH200_USER=%s\n' "$id" "$ip" "$SSH_USER" > "$STATE/nebius.env"
echo "created $id at $ip; waiting for ssh"
for _ in $(seq 60); do
  ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -x "$SSH_USER@$ip" \
    'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader' 2>/dev/null && break
  sleep 10
done
echo "next: bash scripts/kickoff.sh $ip $SSH_USER      (NEBIUS_INSTANCE_ID=$id for launch.sh)"
