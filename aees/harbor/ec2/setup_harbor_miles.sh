#!/usr/bin/env bash
# The Tmax-aligned Harbor build next to the stock one: Harbor's harbor-miles branch at the commit pinned in
# ../post_training/configs/pins.sh, plus our patches (JSON parser via setting; 120 s per-command limit; Responses API
# reasoning summaries and the max_seq_len context stop). This is
# the build the full TB 2.1 run of Qwen3.6-27B used (post_training/serving_speed_on_TB2.md). Run after setup_ec2.sh.
#
#   bash ec2/setup_harbor_miles.sh [task-list ...]     # default task list: state/tb21_selected_tasks.txt
#
# Result: ~/harbor-miles/.venv/bin/harbor (plans point "harbor_bin" at it), and the listed tasks' images pre-pulled.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                 # aees/harbor
PT="$HERE/../post_training"
export PATH="$HOME/.local/bin:$PATH"
source "$PT/configs/pins.sh"                           # HARBOR_COMMIT
DEST=${HARBOR_MILES_DIR:-$HOME/harbor-miles}
echo "== Harbor (harbor-miles @ ${HARBOR_COMMIT:0:7}) + patches -> $DEST"
[ -d "$DEST/.git" ] || git clone -q https://github.com/harbor-framework/harbor.git "$DEST"
git -C "$DEST" fetch -q origin harbor-miles-v0.20.0
git -C "$DEST" checkout -q -- . 2>/dev/null || true    # drop earlier patches
git -C "$DEST" clean -fdq -- src tests                # and files they added
git -C "$DEST" checkout -q --detach "$HARBOR_COMMIT"
git -C "$DEST" apply "$PT/grpo/harbor_parser.patch"
git -C "$DEST" apply "$PT/grpo/command_timeout.patch"
git -C "$DEST" apply "$PT/grpo/responses_reasoning.patch"   # Responses API: reasoning summaries, stateless; 65,536 stop
(cd "$DEST" && uv sync -q)
"$DEST/.venv/bin/harbor" --version
grep -q "command_timeout_sec" "$DEST/src/harbor/agents/terminus_2/terminus_2.py" && echo "command_timeout patch: ok"
grep -q "reasoning_summary" "$DEST/src/harbor/llms/lite_llm.py" && echo "responses_reasoning patch: ok"

echo "== pre-pull task images"
LISTS=("$@"); [ ${#LISTS[@]} -gt 0 ] || LISTS=("$HERE/state/tb21_selected_tasks.txt")
TB=$HOME/tb21-tasks
[ -d "$TB/terminal-bench-2-1" ] || "$DEST/.venv/bin/harbor" datasets download terminal-bench/terminal-bench-2-1 -o "$TB" >/dev/null
cat "${LISTS[@]}" | sort -u > /tmp/tasks_used.txt
python3 "$HERE/scripts/task_images.py" "$TB/terminal-bench-2-1" /tmp/tasks_used.txt > /tmp/images.txt
xargs -P 8 -n 1 docker pull -q < /tmp/images.txt >/dev/null
echo "pulled $(wc -l < /tmp/images.txt) images; disk free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
