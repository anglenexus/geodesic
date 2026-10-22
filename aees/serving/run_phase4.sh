#!/usr/bin/env bash
# run_phase4.sh - Phase 4 FlashInfer and kernel deep dive, end to end, on the GPU host.
#
#   ./run_phase4.sh            # everything: kernels -> jit -> trace -> report (capped at BUDGET_MIN, default 30)
#   ./run_phase4.sh kernels    # 2a sparse KV + 2c load balancing + 2d weight precision (server stopped)
#   ./run_phase4.sh jit        # 2b JIT compile timings in a cold venv (server stopped)
#   ./run_phase4.sh trace      # 2a live KV index capture under a few minutes of real traffic
#   ./run_phase4.sh report     # rebuild results/fi*/fi_report.html from whatever has run
#   MODEL=qwen3.5-27b ./run_phase4.sh   # Qwen3.5-27B shapes and server; results in results/fi_qwen3.5-27b
#
# Options (env vars): MODEL=qwen3-4b|qwen3.5-27b  BUDGET_MIN=30 (0 = no cap)  RESTORE=1 (restart the
# normal server at the end; off by default to save time)  WEIGHTS=awq|fp8|bf16 (27B checkpoint for the trace)
#
# Every step has a time limit, and a step that would start after the budget is used is skipped and
# logged; the report shows which sections have data. Run inside tmux (tmux new -s phase4).
# Run from the folder holding the scripts (geodesic/aees/serving/). Venv, logs and results stay in
# WORK_DIR (~/qwen-serve), outside the repo. A failed step is logged and the next one still runs.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
PY="${PY:-$WORK_DIR/.venv/bin/python}"
COLD_VENV="${COLD_VENV:-$HOME/fi-cold}"
MODEL="${MODEL:-qwen3-4b}"  # qwen3-4b | qwen3.5-27b: attention shapes, serve script, results folder
BUDGET_MIN="${BUDGET_MIN:-30}"
RESTORE="${RESTORE:-0}"
case "$MODEL" in
  qwen3-4b)    SERVE=serve.sh;           OUT="${FI_OUT:-$WORK_DIR/results/fi}" ;;
  qwen3.5-27b) SERVE=serve_qwen3_27B.sh; OUT="${FI_OUT:-$WORK_DIR/results/fi_$MODEL}" ;;
  *) echo "Unknown MODEL=$MODEL (qwen3-4b or qwen3.5-27b)"; exit 1 ;;
esac
export WORK_DIR FI_OUT="$OUT" FI_MODEL="$MODEL"
# Output goes through tee into the log; without this Python buffers prints and progress appears in bursts.
export PYTHONUNBUFFERED=1
# FlashInfer's JIT calls the `ninja` and `nvcc` executables, so the venv's bin and CUDA must be on PATH
# even though we call the venv python directly instead of activating it.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
VENV_BIN="$(dirname "$PY")"
export PATH="$VENV_BIN:$CUDA_HOME/bin:$PATH"
mkdir -p "$OUT"
LOG="$OUT/phase4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

START=$(date +%s)
elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }
step() { printf '\n\033[1;34m==> [%s, %s min] %s\033[0m\n' "$(date +%H:%M:%S)" "$(elapsed_min)" "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; FAILED+=("$*"); }
FAILED=()
# run_limited <minutes> <description> <command...>: skip if the budget is used, else run with a time limit
run_limited() {
  local mins="$1" what="$2"; shift 2
  if (( BUDGET_MIN > 0 )); then
    local left=$(( BUDGET_MIN - $(elapsed_min) ))
    if (( left <= 1 )); then warn "$what skipped: ${BUDGET_MIN}-minute budget used"; return 0; fi
    (( mins > left )) && mins=$left
  fi
  timeout --signal=INT --kill-after=30 "$(( mins * 60 ))" "$@"
  local rc=$?
  (( rc == 124 )) && { warn "$what stopped at its ${mins}-minute limit (partial results kept)"; return 0; }
  (( rc == 0 )) || warn "$what failed (see log)"
  return 0
}
cd "$HERE" || exit 1

for f in "$SERVE" run_bench.py make_dashboard.py fi_bench.py fi_jit_setup.sh fi_trace.py; do
  [[ -e "$HERE/$f" ]] || { echo "Missing $HERE/$f"; exit 1; }
done
[[ -x "$PY" ]] || { echo "No serving venv python at $PY (run setup_host.sh first)"; exit 1; }

# Python packages the benchmark scripts need, beyond what SGLang installs (report charts, JIT builds).
if ! "$PY" -c "import pandas, plotly" 2>/dev/null || ! [[ -x "$VENV_BIN/ninja" ]]; then
  step "Installing pandas, plotly and ninja into the serving venv"
  VIRTUAL_ENV="$(dirname "$VENV_BIN")" "$HOME/.local/bin/uv" pip install pandas plotly ninja \
    || warn "could not install pandas/plotly/ninja"
fi

kernels() {
  step "Stopping the server so the microbenchmarks get the whole, quiet GPU"
  bash "$SERVE" stop
  step "2a sparse paged KV: page size x layout x KV dtype (BF16, FP8 e5m2, FP8 e4m3)"
  run_limited 5 "2a sparse benchmark" "$PY" fi_bench.py sparse
  step "2c load balancing: skewed batches, split-KV on/off, plan() cost"
  run_limited 4 "2c balance benchmark" "$PY" fi_bench.py balance
  step "2d weight precision: linear layers in BF16, FP8 and AWQ 4-bit, with SGLang's kernels"
  # FP8 is measured with SGLang's prebuilt CUTLASS block-FP8 kernel; DeepGEMM (what the server picks on
  # Hopper) would first JIT-compile for 10-20 minutes.
  run_limited 6 "2d precision benchmark" env SGLANG_ENABLE_JIT_DEEPGEMM=0 "$PY" fi_bench.py precision
}

jit() {
  step "Stopping the server (JIT test needs the GPU)"
  bash "$SERVE" stop
  step "2b building the cold venv (same torch + FlashInfer, no prebuilt kernels, empty JIT cache)"
  if run_limited 8 "2b cold venv setup" bash fi_jit_setup.sh && [[ -f "$COLD_VENV/jit_env.sh" ]]; then
    step "2b timing first-use compilation of 10 attention variants"
    # shellcheck disable=SC1091
    run_limited 8 "2b JIT benchmark" bash -c "source '$COLD_VENV/jit_env.sh' && export PATH='$COLD_VENV/bin':\$PATH \
      && '$COLD_VENV/bin/python' fi_bench.py jit"
  fi
}

trace() {
  step "2a live trace: server with the capture hook, a few minutes of baseline traffic"
  rm -rf "$OUT/trace"
  # Python auto-loads a module named sitecustomize from PYTHONPATH; link fi_trace.py under that name
  # in a scratch folder (outside the repo) so only the traced server picks it up.
  HOOK="$OUT/.trace_hook"
  mkdir -p "$HOOK" && ln -sf "$HERE/fi_trace.py" "$HOOK/sitecustomize.py"
  run_limited 12 "trace traffic run" env "SERVER_ENV=PYTHONPATH=$HOOK FI_TRACE_DIR=$OUT/trace" \
    "$PY" run_bench.py --model "$MODEL" --trace --no-restore
  n=$(find "$OUT/trace" -name '*.npz' 2>/dev/null | wc -l)
  echo "Captured $n index snapshots in $OUT/trace"
  [[ "$n" -gt 0 ]] || warn "no trace snapshots captured; check the server log for [fi_trace] lines"
  if [[ "$RESTORE" == 1 ]]; then
    step "Restoring the normal server (no capture hook)"
    SERVER_ENV="" bash "$SERVE" restart || warn "server restart failed; run ./$SERVE start"
  else
    step "Stopping the traced server (RESTORE=1 restarts a normal one instead)"
    bash "$SERVE" stop
  fi
}

report() {
  step "Building the Phase 4 report"
  "$PY" fi_bench.py report || warn "report build failed (see log)"
  bash serve.sh dashboard >/dev/null 2>&1 || true
  echo
  echo "Report: http://127.0.0.1:${DASH_PORT:-8000}/$(basename "$OUT")/fi_report.html  (through the tunnel from ./serve.sh tunnel)"
  echo "For git: $PY export_results.py --phase4 $OUT   (add --phase3 <run dir> to put both in one folder)"
}

case "${1:-all}" in
  all)     kernels; jit; trace; report ;;
  kernels) kernels; report ;;
  jit)     jit; report ;;
  trace)   trace; report ;;
  report)  report ;;
  *)       echo "Usage: $0 [all|kernels|jit|trace|report]"; exit 1 ;;
esac

echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "Phase 4 finished cleanly in $(elapsed_min) min. Log: $LOG"
else
  echo "Phase 4 finished in $(elapsed_min) min with ${#FAILED[@]} problem(s); paste the lines below and the log tail to debug:"
  printf '  - %s\n' "${FAILED[@]}"
  echo "Log: $LOG"
fi
