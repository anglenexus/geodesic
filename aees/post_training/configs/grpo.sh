# GRPO set (PLAN.md §3d). Sourced after configs/common.sh by scripts/run.sh grpo.
# Rollouts: SGLang serves the policy; the Harbor agent server (scripts/agent_server.sh, on the host)
# runs terminus-2 in Docker and returns the verifier reward. Miles trains on the sampled tokens (TITO, D9).

PROMPT_DATA=${PROMPT_DATA:-/root/post_training/data/grpo_train.jsonl}   # from grpo/split_tasks.py
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR to a new folder per run, e.g. /root/ckpt/grpo_pilot1}
NUM_ROLLOUT=${NUM_ROLLOUT:-50}                  # steps; the Phase 3 pilot stops at 50 (§4)
MAX_TURN_TOKENS=8192                            # output cap per model call (Harbor's default)

GRPO_ARGS=(
  --prompt-data "$PROMPT_DATA" --input-key prompt --metadata-key metadata   # rows: {"prompt", "metadata"}
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "${ROLLOUT_BATCH:-20}" --n-samples-per-prompt "${N_SAMPLES:-8}"   # all 20 tasks x G = 8 (§3f)
  --global-batch-size "${GLOBAL_BATCH:-160}"      # one optimizer update per step: fully on-policy
  --rollout-temperature 1.0 --rollout-top-p 1.0  # sample from the policy as-is (§3d)
  --max-seq-len "$MAX_LEN"                        # 48k per trajectory: Harbor ends the trial there and
                                                  #   the session server cuts the sample there (D10)
  --rollout-max-response-len "$MAX_TURN_TOKENS"
  --apply-chat-template-kwargs '{"enable_thinking": false}'   # thinking off (D13); reaches the session server
  --advantage-estimator grpo                      # (r - group mean) / (group std + 1e-6)
  --eps-clip 0.2 --loss-type policy_loss          # no --eps-clip-high (§3d)
  --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0   # no extras (D7)
)
OPTIM_ARGS=(
  --optimizer adam --lr 1e-5 --lr-decay-style constant     # §3a
  --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --weight-decay 0
  --clip-grad 0                                            # off (§3d); the grad norm is still logged
)
SAVE_ARGS=(
  --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-10}"   # adapter + optimizer state every 10 steps (Phase 3)
)
# Optional warm start from an SFT adapter (D15): its weights only. The SFT folder also holds SFT's step
# count, which Miles would restore and then skip that many GRPO steps; copying the weights avoids that.
INIT_ADAPTER=${INIT_ADAPTER:-}                             # e.g. /root/ckpt/sft_smoke1/iter_0000007/adapter
if [ -n "$INIT_ADAPTER" ]; then
  mkdir -p "$SAVE_DIR/init_adapter"
  cp "$INIT_ADAPTER"/adapter_config.json "$INIT_ADAPTER"/adapter_megatron_rank*.pt "$SAVE_DIR/init_adapter/"
  GRPO_ARGS+=(--lora-adapter-path "$SAVE_DIR/init_adapter")   # same LoRA shape: both use common.sh
fi
# Continue a run from a saved iteration: adapter, Adam, scheduler and step count (Miles restore_lora_adapter).
# NUM_ROLLOUT stays the total, so the run goes on from the saved step. common.sh drops --no-load-optim for it.
RESUME_FROM=${RESUME_FROM:-}                               # e.g. /root/ckpt/grpo10h_1026_1924/iter_0000007/adapter
if [ -n "$RESUME_FROM" ]; then
  [ -z "$INIT_ADAPTER" ] || { echo "set RESUME_FROM or INIT_ADAPTER, not both"; exit 2; }
  GRPO_ARGS+=(--lora-adapter-path "$RESUME_FROM")
fi
SGLANG_MEM=${SGLANG_MEM:-0.85}                   # GPU share for SGLang weights + KV pool; Phase 2 tunes it
SGLANG_SERVER=(                                  # SGLang's own flags; Miles takes each as --sglang-<flag>, test T2 as is
  --mem-fraction-static "$SGLANG_MEM"             # 0.85 x 141 = 120 GB: 55 weights + adapter, ~65 KV pool
  --context-length $((MAX_LEN + MAX_TURN_TOKENS)) # 56k, so the turn that crosses 48k can finish
  --dtype bfloat16 --lora-backend triton          # as Miles' native LoRA example; rank = --lora-rank
  ${SGLANG_EXTRA:-}                               # set from integration test T2 (integration_tests.md)
)
SGLANG_ARGS=(--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 "${SGLANG_SERVER[@]/#--/--sglang-}")
AGENT_ARGS=(
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate   # agent loop, TITO
  --custom-agent-function-path swe_agent_function.run      # POSTs each trial to $AGENT_SERVER_URL/run
  --custom-rm-path generate.reward_func                    # reward = the verifier's, from the agent server
  --rollout-function-path generate.RolloutFn               # Miles' rollout + agent timing metrics
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
  --custom-reward-post-process-path grpo.infra_filter.post_process_rewards   # infra out per attempt (§8 item 6)
  --tito-model qwen36                                      # Qwen3.6 token stream with preserve_thinking (D9)
  --use-session-server --session-server-port 30000 --session-server-workers 16   # ports 30000-30015
  --session-server-ip 0.0.0.0 --session-server-external-host 127.0.0.1   # agents on EC2 reach them through the
                                                  #   H200's reverse tunnels; eth0 is firewalled (dist_setup.md)
  --sglang-router-port 31000
)
GRPO_ARGS+=("${SGLANG_ARGS[@]}" "${AGENT_ARGS[@]}")
