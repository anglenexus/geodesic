#!/usr/bin/env bash
# Terminate the prototype's EC2 box (proto/state/ec2.env). Runs on the laptop. Fetch results first (proto/fetch.sh).
set -euo pipefail
STATE=$(cd "$(dirname "$0")" && pwd)/state
[ -f "$STATE/ec2.env" ] || { echo "no proto/state/ec2.env: nothing to terminate"; exit 0; }
source "$STATE/ec2.env"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$EC2_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
rm "$STATE/ec2.env"
