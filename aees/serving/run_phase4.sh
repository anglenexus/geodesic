#!/usr/bin/env bash
# run_phase4.sh - Phase 4 FlashInfer deep dive, end to end (~1.5 hours), on the GPU host.
#
#   ./run_phase4.sh            # everything: kernels -> jit -> trace -> report
#   ./run_phase4.sh kernels    # 2a sparse KV + 2c load balancing microbenchmarks (~10 min, server stopped)
#   ./run_phase4.sh jit        # 2b JIT compile timings in a cold venv            (~15-25 min, server stopped)
#   ./run_phase4.sh trace      # 2a live KV index capture under real traffic      (~10 min, server running)
#   ./run_phase4.sh report     # rebuild results/fi/fi_report.html from whatever has run
#
# Run inside tmux (tmux new -s phase4) so a dropped SSH session doesn't stop it.
# Run from the folder holding the scripts (e.g. geodesic/aees/serving/): serve.sh, run_bench.py,
# make_dashboard.py, fi_bench.py, fi_jit_setup.sh and fi_trace.py all sit side by side there.
# Venv, logs and results stay in WORK_DIR (~/qwen-serve), outside the repo.
# A failed step is logged and the next one still runs; the report shows which sections have data.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
PY="${PY:-$WORK_DIR/.venv/bin/python}"
COLD_VENV="${COLD_VENV:-$HOME/fi-cold}"
OUT="${FI_OUT:-$WORK_DIR/results/fi}"
export WORK_DIR FI_OUT="$OUT"
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

step() { printf '\n\033[1;34m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; FAILED+=("$*"); }
FAILED=()
cd "$HERE" || exit 1

for f in serve.sh run_bench.py make_dashboard.py fi_bench.py fi_jit_setup.sh fi_trace.py; do
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
  bash serve.sh stop
  step "2a sparse paged KV: page size x layout x KV dtype"
  "$PY" fi_bench.py sparse || warn "2a sparse benchmark failed (see log)"
  step "2c load balancing: skewed batches, split-KV on/off, plan() cost"
  "$PY" fi_bench.py balance || warn "2c balance benchmark failed (see log)"
}

jit() {
  step "Stopping the server (JIT test needs the GPU)"
  bash serve.sh stop
  step "2b building the cold venv (same torch + FlashInfer, no prebuilt kernels, empty JIT cache)"
  if bash fi_jit_setup.sh; then
    step "2b timing first-use compilation of 10 attention variants"
    # shellcheck disable=SC1091
    ( source "$COLD_VENV/jit_env.sh" && export PATH="$COLD_VENV/bin:$PATH" && "$COLD_VENV/bin/python" fi_bench.py jit ) \
      || warn "2b JIT benchmark failed (see log)"
  else
    warn "2b cold venv setup failed (see log)"
  fi
}

trace() {
  step "2a live trace: baseline server with the capture hook, quick baseline traffic"
  rm -rf "$OUT/trace"
  # Python auto-loads a module named sitecustomize from PYTHONPATH; link fi_trace.py under that name
  # in a scratch folder (outside the repo) so only the traced server picks it up.
  HOOK="$OUT/.trace_hook"
  mkdir -p "$HOOK" && ln -sf "$HERE/fi_trace.py" "$HOOK/sitecustomize.py"
  SERVER_ENV="PYTHONPATH=$HOOK FI_TRACE_DIR=$OUT/trace" \
    "$PY" run_bench.py --quick --arms baseline --no-restore || warn "trace traffic run failed (see log)"
  n=$(find "$OUT/trace" -name '*.npz' 2>/dev/null | wc -l)
  echo "Captured $n index snapshots in $OUT/trace"
  [[ "$n" -gt 0 ]] || warn "no trace snapshots captured; check the server log for [fi_trace] lines"
  step "Restoring the normal server (no capture hook)"
  SERVER_ENV="" bash serve.sh restart || warn "server restart failed; run ./serve.sh start"
}

report() {
  step "Building the Phase 4 report"
  "$PY" fi_bench.py report || warn "report build failed (see log)"
  bash serve.sh dashboard >/dev/null 2>&1 || true
  echo
  echo "Report: http://127.0.0.1:${DASH_PORT:-8000}/fi/fi_report.html  (through the tunnel from ./serve.sh tunnel)"
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
  echo "Phase 4 finished cleanly. Log: $LOG"
else
  echo "Phase 4 finished with ${#FAILED[@]} problem(s); paste the lines below and the log tail to debug:"
  printf '  - %s\n' "${FAILED[@]}"
  echo "Log: $LOG"
fi
