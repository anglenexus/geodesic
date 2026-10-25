#!/usr/bin/env bash
# Live view of a running bench (PLAN.md §5b), on the laptop after proto/kickoff.sh:
#   bash proto/watch.sh            # Ctrl-C to stop
# Every INTERVAL_S: proto/fetch.sh with LIVE=1, which pulls the results so far and rewrites
# data/proto/$RUN/dashboard.html with the running phase included. The page reloads itself every minute.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                    # aees/post_training
INTERVAL_S=${INTERVAL_S:-60}
source "$HERE/proto/state/run.env"
PAGE=$HERE/data/proto/$RUN/dashboard.html
opened=0
while true; do
  if LIVE=1 bash "$HERE/proto/fetch.sh" > "$HERE/proto/state/watch.log" 2>&1; then
    echo "$(date +%T) updated $PAGE"
    [ "$opened" = 1 ] || { command -v open >/dev/null && open "$PAGE"; opened=1; }
  else
    echo "$(date +%T) fetch failed (proto/state/watch.log)"
  fi
  sleep "$INTERVAL_S"
done
