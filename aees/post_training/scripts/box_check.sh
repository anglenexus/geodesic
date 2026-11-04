#!/usr/bin/env bash
# On the H200, before Miles starts rollouts (launch.sh, after links.sh start): every EC2 sandbox box answers and stays
# up long enough for the steps left (integration_issues.md: the boxes' 15 h shutdown ended step 10 of auxr_1103_0033).
# For box i in ~/aees_boxes:
#   ping   its agent server through tunnel i, 127.0.0.1:1100i/health (box up, tunnel up, server up)
#   time   its scheduled shutdown (shutdown_at, written by ec2_box.sh serve, read with the read-only sync key)
#          is at least NEED_MIN minutes away
#
#   NEED_MIN=220 bash scripts/box_check.sh       # one line per box; exits 1 if any box fails
set -uo pipefail
NEED_MIN=${NEED_MIN:?set NEED_MIN, the minutes the run still needs the boxes}
BOXES_FILE=${BOXES_FILE:-$HOME/aees_boxes}
PING_TRIES=${PING_TRIES:-30}                             # seconds: tunnels come up a few seconds after links.sh start
read -ra BOXES < "$BOXES_FILE"
SSHO="-o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10"
now=$(date +%s); fail=0
for i in "${!BOXES[@]}"; do
  n=$((i + 1)); ip=${BOXES[$i]}; ping="no answer"
  for _ in $(seq "$PING_TRIES"); do
    curl -sf --max-time 5 "127.0.0.1:1100$n/health" >/dev/null 2>&1 && { ping=ok; break; }
    sleep 1
  done
  tmp=$(mktemp); ok=1
  if ! rsync -e "ssh -i ${SYNC_KEY:-$HOME/.ssh/aees_sync} $SSHO" "ubuntu@$ip:shutdown_at" "$tmp" >/dev/null 2>&1; then
    time="shutdown time unreadable (ec2_box.sh serve writes it)"; ok=0
  elif [ -z "$(tr -dc 0-9 < "$tmp")" ]; then time="no shutdown scheduled"
  else
    at=$(tr -dc 0-9 < "$tmp"); left=$(((at - now) / 60))
    time="shuts down $(date -u -d "@$at" '+%F %T' 2>/dev/null || date -u -r "$at" '+%F %T') UTC, in $left min"
    [ "$left" -ge "$NEED_MIN" ] || ok=0
  fi
  rm -f "$tmp"
  [ "$ping" = ok ] || ok=0
  echo "box $n ($ip): ping $ping; $time; need $NEED_MIN min: $([ $ok = 1 ] && echo OK || echo FAIL)"
  [ $ok = 1 ] || fail=1
done
exit "$fail"
