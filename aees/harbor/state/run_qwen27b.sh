#!/bin/bash
# 20 selected TB 2.1 tasks x 20 attempts with Qwen3.5-27B (SGLang over the SSH tunnel on :30000), on the Mac.
# Server: aees/serving/serve_qwen3_27B.sh. Thinking is on by default; THINK=0 turns it off per request.
#
#   bash state/run_qwen27b.sh                  # thinking on
#   THINK=0 bash state/run_qwen27b.sh          # thinking off (chat_template_kwargs.enable_thinking=false)
#   CTX=65536 bash state/run_qwen27b.sh        # must equal the server's CONTEXT_LEN
#   N=8 K=20                                   # concurrent trials (keep <= server MAX_RUNNING), attempts per task
#
# Terminus-2 sends no max_tokens, so a turn may generate up to CTX minus its prompt. Thinking needs that
# headroom: max_input_tokens is half of CTX, which makes terminus-2 summarize history before the prompt
# grows past it. A turn that still runs out raises "hit max_tokens limit" and the agent is told to retry.
set -euo pipefail
cd "$(dirname "$0")/.."
THINK="${THINK:-1}"
CTX="${CTX:-32768}"
N="${N:-8}"
K="${K:-20}"
args=()
while read -r t; do [ -n "$t" ] && args+=(-i "terminal-bench/$t"); done < state/tb21_selected_tasks.txt
extra=()
if [ "$THINK" = 0 ]; then
  extra+=(--ak 'llm_call_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}')
  mode=nothink
else
  mode=think
fi
export HOSTED_VLLM_API_KEY=none DOCKER_DEFAULT_PLATFORM=linux/amd64
exec caffeinate -i harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
  -m hosted_vllm/qwen3.5-27b \
  --ak api_base='"http://127.0.0.1:30000/v1"' \
  --ak "model_info={\"max_input_tokens\": $((CTX / 2)), \"max_output_tokens\": $((CTX / 2)), \"input_cost_per_token\": 0.0, \"output_cost_per_token\": 0.0}" \
  --ak 'trajectory_config={"raw_content": true, "linear_history": true}' --ak store_all_messages=true \
  ${extra[@]+"${extra[@]}"} \
  --env docker -o ~/code/jobs --job-name "tb21_qwen27b_${mode}_ctx$((CTX / 1024))k_k${K}" -n "$N" -k "$K" -r 0 -y "${args[@]}"
