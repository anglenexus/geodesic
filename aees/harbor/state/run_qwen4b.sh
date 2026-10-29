#!/bin/bash
# 20 selected TB 2.1 tasks x 20 attempts with Qwen3-4B (SGLang over the SSH tunnel on :30000), -n 15 on the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
args=()
while read -r t; do [ -n "$t" ] && args+=(-i "terminal-bench/$t"); done < state/tb21_selected_tasks.txt
export HOSTED_VLLM_API_KEY=none DOCKER_DEFAULT_PLATFORM=linux/amd64
exec caffeinate -i harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
  -m hosted_vllm/qwen3-4b \
  --ak api_base='"http://127.0.0.1:30000/v1"' \
  --ak 'model_info={"max_input_tokens": 28672, "max_output_tokens": 4096, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}' \
  --ak 'trajectory_config={"raw_content": true, "linear_history": true}' --ak store_all_messages=true \
  --env docker -o ~/code/jobs --job-name tb21_qwen4b_k20 -n 15 -k 20 -r 0 -y "${args[@]}"
