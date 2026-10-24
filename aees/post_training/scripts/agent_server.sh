#!/usr/bin/env bash
# Harbor agent server for GRPO, on the host (not in the Miles container). Miles POSTs each trial to
# http://127.0.0.1:11000/run; the server starts the task's container, runs terminus-2 against Miles'
# session server, runs the verifier, and returns {"reward", "exit_status", ...} (agent_server/results.py).
# Start it before scripts/run.sh grpo, and check: curl 127.0.0.1:11000/health
#
#   RUN=grpo_pilot1 bash scripts/agent_server.sh          # trial folders -> $DATA/trials/$RUN
set -euo pipefail
DATA=${DATA:-/data}
RUN=${RUN:?set RUN to the GRPO run name, e.g. grpo_pilot1}
MAX_LEN=${MAX_LEN:-49152}                          # same as configs/common.sh (D10)

export HARBOR_TASKS_DIR=$DATA/tasks/terminal-bench-2-1   # one folder per instance_id (grpo/split_tasks.py)
export HARBOR_TERMINUS_2_PARSER=json               # as SFT data, baseline, evaluation (grpo/harbor_parser.patch)
export HARBOR_TERMINUS_2_ENABLE_SUMMARIZE=false    # summarization off (D10)
export AGENT_MAX_INPUT_TOKENS=$MAX_LEN             # terminus-2's context size; = Miles' --max-seq-len
export AGENT_MAX_OUTPUT_TOKENS=8192                # = MAX_TURN_TOKENS in configs/grpo.sh
export HARBOR_TIMEOUT_MULTIPLIER=1.0               # task.toml timeouts as written, as in the ledger (server default 2.0)
export DOCKER_PRUNE_INTERVAL_SEC=1e9               # off: its `docker image prune -a` deletes the pre-pulled task images

mkdir -p "$DATA/trials/$RUN"
cd "$DATA/src/harbor"
uv run python miles_agent_server.py --host 127.0.0.1 --port 11000 --max-concurrent "${CONCURRENCY:-16}" \
  --dashboard-port 0 --trials-dir "$DATA/trials/$RUN" 2>&1 | tee -a "$DATA/trials/$RUN/agent_server.log"
# 16 at once (D14); no --agent-timeout, so each task's own timeout applies
