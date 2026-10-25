#!/usr/bin/env bash
# Pull the prototype's results to the laptop and write the report (PLAN.md §5b):
#   H200 /data/proto/$RUN -> data/proto/$RUN ;  EC2 ~/proto/out -> data/proto/$RUN/ec2 ;
#   report.md + report.json + dashboard.html. Safe to run mid-run (partial results).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                    # aees/post_training
source "$HERE/proto/state/run.env"
DEST=$HERE/data/proto/$RUN
mkdir -p "$DEST/ec2"
rsync -az -e "ssh ${H200_KEY:+-i $H200_KEY}" --exclude adapter/ "$H200_USER@$H200:/data/proto/$RUN/" "$DEST/"
if [ -f "$HERE/proto/state/ec2.env" ]; then
  source "$HERE/proto/state/ec2.env"
  rsync -az -e "ssh -i ${KEY:-$HOME/.ssh/aees-ec2.pem}" "ubuntu@$EC2_IP:proto/out/" "$DEST/ec2/" || echo "EC2 results not fetched"
  rsync -az -e "ssh -i ${KEY:-$HOME/.ssh/aees-ec2.pem}" --include '*/' --include result.json --exclude '*' \
    "ubuntu@$EC2_IP:proto/jobs/" "$DEST/ec2/jobs/" || true
fi
cd "$HERE" && uv run python -m proto.report "$DEST"
