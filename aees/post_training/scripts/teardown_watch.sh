#!/usr/bin/env bash
# On the laptop, for a whole run: delete the H200 once finish.sh has published BACKUP_OK to S3 (its status file,
# <run>/monitor/backup_status). Uses the laptop's nebius profile, so the H200 needs no Nebius credentials.
# BACKUP_INCOMPLETE: deletes nothing and exits 1 for review.
#
#   bash scripts/teardown_watch.sh <run> <nebius-instance-id> [every-seconds]     # e.g. in the background
set -uo pipefail
RUN=${1:?usage: $0 <run> <nebius-instance-id> [every-seconds]}; ID=${2:?nebius instance id}; EVERY=${3:-300}
DEST=${DEST:-s3://aees-grpo-backups-611728564241}
PROFILE=${NEBIUS_PROFILE:-aees}
N=nebius; command -v nebius >/dev/null || N=$HOME/.nebius/bin/nebius
while :; do
  s=$(aws s3 cp "$DEST/$RUN/monitor/backup_status" - 2>/dev/null | head -1)
  case "$s" in
    BACKUP_OK)
      echo "$(date +%T) $RUN: BACKUP_OK in S3, deleting H200 $ID"
      "$N" --profile "$PROFILE" compute instance delete --id "$ID" 2>&1 | tail -2
      left=$(aws ec2 describe-instances --region "${REGION:-us-east-2}" --filters Name=tag:Project,Values=aees \
        Name=instance-state-name,Values=pending,running --query 'length(Reservations[].Instances[])' --output text)
      echo "EC2 boxes still running: ${left:-?}"
      exit 0 ;;
    BACKUP_INCOMPLETE)
      echo "$(date +%T) $RUN: BACKUP_INCOMPLETE in S3, H200 $ID NOT deleted; review $DEST/$RUN/monitor/backup_status"
      exit 1 ;;
  esac
  sleep "$EVERY"
done
