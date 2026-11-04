#!/usr/bin/env bash
# run_kimi_k3.sh - the Kimi K3 test run inside the Modal container (started by modal_kimi_k3.py::run):
#   kernels (Phase 4, GPUs idle) -> serve (Balanced recipe, ~13 min) -> phase3 (server reused) -> export -> hold
#
#   bash run_kimi_k3.sh                      # everything
#   bash run_kimi_k3.sh all+low_latency      # everything, plus extra Phase 3 settings (~15 min each, restarts)
#   bash run_kimi_k3.sh serve,phase3,hold    # a subset, comma-separated
#
# Env: PHASE_MIN=15 (cap per phase)  HOLD_MIN=15 (smoke tests + dashboards before exiting; `touch /tmp/STOP`
# ends it early)  RESULTS_DIR=/results/results (Volume, served on :8000)  WORK_DIR=/root/kimi-serve
# A failed step is logged and the next one still runs. The Modal function's timeout is the outer kill switch.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
PHASE_MIN="${PHASE_MIN:-15}"
HOLD_MIN="${HOLD_MIN:-15}"
WORK_DIR="${WORK_DIR:-/root/kimi-serve}"
RESULTS_DIR="${RESULTS_DIR:-/results/results}"
EXPORT_DIR="${EXPORT_DIR:-/results/export}"
TS="$(date +%Y%m%d-%H%M%S)"
export WORK_DIR RESULTS_DIR PYTHONUNBUFFERED=1
# Phase 4 results: this run's folder, or (when the kernels step is skipped) the latest earlier one.
FI_OUT="$RESULTS_DIR/fi_kimi-k3_$TS"
if [[ "$*" != *kernels* && "${1:-all}" != all* ]]; then
  prev=$(ls -dt "$RESULTS_DIR"/fi_kimi-k3_* 2>/dev/null | head -1)
  [[ -n "$prev" ]] && FI_OUT="$prev"
fi
export FI_OUT
mkdir -p "$WORK_DIR/logs" "$RESULTS_DIR" "$EXPORT_DIR" /results/logs
LOG="/results/logs/run_$TS.log"
exec > >(tee -a "$LOG") 2>&1

SPEC="${1:-all}"
STEPS="${SPEC%%+*}"
ARMS="baseline"
[[ "$SPEC" == *+* ]] && ARMS="baseline ${SPEC#*+}" && ARMS="${ARMS//,/ }"
has() { [[ "$STEPS" == all || ",$STEPS," == *",$1,"* ]]; }

START=$(date +%s)
elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }
step() { printf '\n\033[1;34m==> [%s, %s min] %s\033[0m\n' "$(date +%H:%M:%S)" "$(elapsed_min)" "$*"; }
FAILED=()
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; FAILED+=("$*"); }
# limited <minutes> <description> <command...>: run with a time limit; a timeout keeps partial results
limited() {
  local mins="$1" what="$2"; shift 2
  timeout --signal=INT --kill-after=30 "$(( mins * 60 ))" "$@"
  local rc=$?
  (( rc == 124 )) && { warn "$what stopped at its ${mins}-minute limit (partial results kept)"; return 0; }
  (( rc == 0 )) || warn "$what failed (exit $rc, see $LOG)"
  return 0
}

step "Run $TS: steps=$STEPS settings=$ARMS phase cap=${PHASE_MIN} min hold=${HOLD_MIN} min"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python -c "import torch, flashinfer, sglang; print('torch', torch.__version__, 'CUDA', torch.version.cuda, \
'FlashInfer', flashinfer.__version__, 'SGLang', sglang.__version__)"
# Results folder on :8000 from the start, so reports can be watched as they are written (through the tunnel).
python -m http.server 8000 --bind 127.0.0.1 --directory "$RESULTS_DIR" >/dev/null 2>&1 &

done_fi=$(ls -dt "$RESULTS_DIR"/fi_kimi-k3_*/fp4.json 2>/dev/null | head -1)
if has kernels && [[ -n "$done_fi" ]]; then
  FI_OUT="$(dirname "$done_fi")"; export FI_OUT
  step "Phase 4 results already in $FI_OUT (earlier attempt); skipping the kernels"
elif has kernels; then
  step "Phase 4: kernel microbenchmarks with the GPUs idle (cap $PHASE_MIN min)"
  t0=$(date +%s)
  limited $(( PHASE_MIN * 45 / 100 )) "4a MLA decode" python fi_bench_kimi.py mla
  left=$(( PHASE_MIN - ($(date +%s) - t0) / 60 - 1 ))
  (( left > 1 )) && limited "$left" "4b FP4 GEMM" python fi_bench_kimi.py fp4 || warn "4b FP4 GEMM skipped: phase cap used"
  python fi_bench_kimi.py report || warn "Phase 4 report failed"
fi

if has serve; then
  step "Server start: Kimi K3, TP8 + DCP8 (weights from the Volume)"
  t0=$(date +%s)
  bash serve_kimi_k3.sh start 2>&1 | tee "$WORK_DIR/logs/server_start.txt"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || warn "server start failed"
  echo "Server start took $(( $(date +%s) - t0 )) s" | tee -a "$WORK_DIR/logs/server_start.txt"
fi

P3=""
if has phase3; then
  n_extra=$(( $(wc -w <<<"$ARMS") - 1 ))
  budget=$(( PHASE_MIN + 15 * n_extra ))
  step "Phase 3: serving tests on the running server (cap $budget min; settings: $ARMS)"
  # shellcheck disable=SC2086
  # An unfinished earlier Phase 3 (preempted VM) is resumed: points already done are skipped.
  prev=$(ls -dt "$RESULTS_DIR"/kimi-k3_* 2>/dev/null | head -1)
  resume=()
  [[ -n "$prev" ]] && ! grep -q "^Finished after" "$prev/run_log.md" 2>/dev/null && resume=(--resume "$prev") \
    && echo "Resuming unfinished Phase 3 in $prev"
  limited $(( budget + 2 )) "Phase 3" python run_bench.py --model kimi-k3 --reuse-server --no-restore \
    --budget-min "$budget" --arms $ARMS --out "$RESULTS_DIR" "${resume[@]}"
  P3=$(ls -dt "$RESULTS_DIR"/kimi-k3_* 2>/dev/null | head -1)
  cp "$WORK_DIR"/logs/*.log "$WORK_DIR"/logs/*.cmd "$P3/" 2>/dev/null || true
fi

if has export; then
  step "Export: small, committable results into $EXPORT_DIR"
  args=()
  [[ -n "$P3" && -d "$P3" ]] && args+=(--phase3 "$P3")
  [[ -d "$FI_OUT" ]] && args+=(--phase4 "$FI_OUT")
  if (( ${#args[@]} )); then
    python export_results.py "${args[@]}" --dest "$EXPORT_DIR" || warn "export failed"
  else
    warn "nothing to export"
  fi
fi

if has hold; then
  step "Hold window: $HOLD_MIN min for smoke tests and dashboards (touch /tmp/STOP to end now)"
  echo "  python smoke_test.py --model kimi-k3 --no-think        http://127.0.0.1:8000/"
  end=$(( $(date +%s) + HOLD_MIN * 60 ))
  while (( $(date +%s) < end )) && [[ ! -e /tmp/STOP ]]; do
    sleep 15
  done
fi

step "Stopping the server"
bash serve_kimi_k3.sh stop || true
echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "Run finished cleanly in $(elapsed_min) min. Log: $LOG"
else
  echo "Run finished in $(elapsed_min) min with ${#FAILED[@]} problem(s):"
  printf '  - %s\n' "${FAILED[@]}"
  echo "Log: $LOG"
fi
