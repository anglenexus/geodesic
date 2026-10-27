#!/usr/bin/env bash
# Launch and prepare the EC2 sandbox box for the overlap prototype (PLAN.md §5b). Runs on the laptop.
#
#   bash proto/ec2_up.sh            # ~15-20 min (Docker, Harbor, task images); then bash proto/kickoff.sh <h200-ip>
#
# Uses the laptop's AWS CLI credentials (AWS_PROFILE; nothing is copied to the instance). SSH is open only
# to this laptop's public IP. The instance terminates itself after TTL_MIN minutes whatever happens here
# (shutdown in user-data + terminate-on-shutdown). Writes proto/state/ec2.env.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)                      # aees/post_training/proto
STATE=$HERE/state
REGION=${REGION:-us-east-2}                              # where the aees-ec2 key pair lives
TYPE=${TYPE:-m7i.8xlarge}                                # 32 vCPU, 128 GB: 8 concurrent trials with room
KEY_NAME=${KEY_NAME:-aees-ec2}
KEY=${KEY:-$HOME/.ssh/aees-ec2.pem}
TTL_MIN=${TTL_MIN:-240}
mkdir -p "$STATE"
[ ! -f "$STATE/ec2.env" ] || { echo "already up ($(tr '\n' ' ' < "$STATE/ec2.env")); run proto/ec2_down.sh first"; exit 1; }

source "$HERE/../scripts/ec2_lib.sh"
ec2_identity
AMI=$(ec2_ami "$REGION")
SG=$(ec2_sg "$REGION" aees-proto "aees overlap prototype: SSH from the laptop only")
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "$(printf '#!/bin/bash\nshutdown -h +%s\n' "$TTL_MIN")" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aees-proto},{Key=Project,Value=aees}]' \
  --query 'Instances[0].InstanceId' --output text)
printf 'REGION=%s\nEC2_ID=%s\n' "$REGION" "$ID" > "$STATE/ec2.env"
echo "launched $ID ($TYPE, $AMI); terminates itself after $TTL_MIN min"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "EC2_IP=$IP" >> "$STATE/ec2.env"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
ec2_wait_ssh "$KEY" "$IP"

echo "== copying aees/harbor + prototype scripts"
"${SSH[@]}" 'mkdir -p geodesic/aees proto'          # rsync creates only the last folder
RSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -az -e "$RSH" --exclude .venv --exclude '*.sqlite' --exclude 'state/remote/' --exclude '*.log' \
  --exclude __pycache__ "$HERE/../../harbor/" "ubuntu@$IP:geodesic/aees/harbor/"
rsync -az -e "$RSH" "$HERE"/{ec2_prepare.sh,ec2_trials.sh,llm_log_proxy.py,ec2_tasks.txt} "ubuntu@$IP:proto/"
"${SSH[@]}" 'bash ~/proto/ec2_prepare.sh' 2>&1 | tail -25
echo "EC2 ready: ubuntu@$IP   next: bash proto/kickoff.sh <h200-ip> [h200-user]"
