#!/usr/bin/env bash
# On the H200, after a run (dist_setup.md, Teardown): gather everything into the run folder, upload it with
# scripts/backup.sh (optimizer state included, for RESUME_FROM), and check every file's size in the destination.
# Writes $DATA/monitor/$RUN/backup_status: BACKUP_OK, or BACKUP_INCOMPLETE with the missing files. On BACKUP_OK,
# terminates the EC2 sandbox boxes listed in ~/aees_box_ids (their trials are in the backup), then, if
# NEBIUS_INSTANCE_ID is set and the nebius CLI is authorized, deletes this H200 permanently.
#
#   RUN=grpo10h_1026_1924 DEST=s3://aees-grpo-backups-611728564241 bash scripts/finish.sh          # now
#   RUN=... DEST=... WAIT_FOR=train bash scripts/finish.sh     # first wait until tmux session "train" ends
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
DATA=${DATA:-/data}
RUN=${RUN:?set RUN}; DEST=${DEST:?set DEST, e.g. s3://<bucket>}
RUN_DIR=$DATA/ckpt/$RUN
STATUS=$DATA/monitor/$RUN/backup_status
if [ -n "${WAIT_FOR:-}" ]; then
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done
  sleep "${SETTLE_SEC:-150}"                             # two trial-sync rounds after the last step
fi
echo "running $(date -u +%FT%TZ)" > "$STATUS"
sudo chown -R "$USER" "$RUN_DIR"                         # containers write as root (some files mode 600)
mkdir -p "$RUN_DIR/monitor" "$RUN_DIR/trials"
rsync -a "$DATA/monitor/$RUN/" "$RUN_DIR/monitor/"       # dashboard history, preflight, box logs, train log
mkdir -p "$RUN_DIR/monitor/dashboard"                    # final tracker state: open index.html, "Open snapshot"
curl -sf "127.0.0.1:${TRACKER_PORT:-8770}/status.json" -o "$RUN_DIR/monitor/dashboard/${RUN}_status.json" \
  || echo "tracker not answering: no dashboard snapshot" >&2
cp "$HERE/monitor/index.html" "$RUN_DIR/monitor/dashboard/index.html"
rm -f "$RUN_DIR/trials/$RUN.tar.gz"                      # re-archive: trials may have arrived since
tar -C "$DATA/trials" -czf "$RUN_DIR/trials/$RUN.tar.gz" "$RUN"   # every trial of the run + dispatch.jsonl
WITH_OPTIMIZER=1 bash "$HERE/scripts/backup.sh" "$RUN_DIR" "$DEST"

python3 - "$RUN_DIR" "$DEST" "$RUN" > "$STATUS.tmp" <<'PY'
import fnmatch, os, subprocess, sys
run_dir, dest, run = sys.argv[1:]
include = ["iter_*/adapter/*", "*.sh", "*.json", "*.yaml", "*.log", "trials/*.tar.gz", "monitor/*"]  # = backup.sh
include += ["monitor/dashboard/*"]
local = {}
for root, _, files in os.walk(run_dir):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), run_dir)
        if any(fnmatch.fnmatch(rel, p) for p in include):
            local[rel] = os.path.getsize(os.path.join(root, f))
out = subprocess.run(["aws", "s3", "ls", "--recursive", f"{dest}/{run}/"], capture_output=True, text=True).stdout
remote = {}
for line in out.splitlines():
    parts = line.split(None, 3)
    if len(parts) == 4:
        remote[parts[3].split(f"{run}/", 1)[1]] = int(parts[2])
bad = [f"{k} local {v} remote {remote.get(k)}" for k, v in sorted(local.items()) if remote.get(k) != v]
total = sum(local.values()) / 1e9
print("BACKUP_OK" if not bad else "BACKUP_INCOMPLETE")
print(f"{len(local)} files, {total:.1f} GB checked against {dest}/{run}/")
print("\n".join(bad))
PY
mv "$STATUS.tmp" "$STATUS"
cat "$STATUS"
if head -1 "$STATUS" | grep -qx BACKUP_OK && [ -s "${BOX_IDS_FILE:-$HOME/aees_box_ids}" ]; then
  aws ec2 terminate-instances --region "${REGION:-us-east-2}" --instance-ids $(cat "${BOX_IDS_FILE:-$HOME/aees_box_ids}") \
    --query 'TerminatingInstances[].CurrentState.Name' --output text | sed 's/^/EC2 boxes: /' | tee -a "$STATUS"
fi
if head -1 "$STATUS" | grep -qx BACKUP_OK && [ -n "${NEBIUS_INSTANCE_ID:-}" ]; then
  if nebius compute instance get --id "$NEBIUS_INSTANCE_ID" >/dev/null 2>&1; then
    echo "deleting H200 $NEBIUS_INSTANCE_ID $(date -u +%FT%TZ)" | tee -a "$STATUS"
    aws s3 cp "$STATUS" "$DEST/$RUN/monitor/backup_status" --only-show-errors || true   # last word, kept in S3
    nebius compute instance delete --id "$NEBIUS_INSTANCE_ID" --async </dev/null 2>&1 | tail -2 | tee -a "$STATUS"   # also its managed disks
  else
    echo "H200 not deleted: nebius CLI not authorized (nebius profile create)" | tee -a "$STATUS"
  fi
fi
