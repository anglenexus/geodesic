# Qwen3-4B serving on SGLang + FlashInfer: setup, benchmarks and FlashInfer deep dive

## Summary

This PR adds `geodesic/aees/serving/`: nine scripts that take a blank Ubuntu 24.04 GPU host to a running Qwen3-4B server, benchmark it, and investigate FlashInfer's key mechanisms. Everything runs from this one flat folder; the venv, logs and results live in `~/qwen-serve` outside the repo.

- **Phases 1–2, serving:** one script installs the driver, CUDA 13, SGLang and version-matched FlashInfer wheels; another runs the server with FlashInfer attention, the prefix cache and Prometheus metrics.
- **Phase 3, benchmarks:** an unattended matrix covering latency, throughput, QPS, p99.9 tails, burst and cold start, across five server settings, plus a tool-call quality check. The output is one interactive HTML dashboard.
- **Phase 4, FlashInfer deep dive:** microbenchmarks and a live capture for block-sparse KV memory, JIT compilation of attention variants, and split-KV load balancing. The output is one HTML report with explanatory diagrams above the measured charts.

Both phases have been run end to end (Phase 3 on an A100-40GB, Phase 4 on an A10).

This PR also includes `geodesic/aees/harbor/`: the experiment and set-up guide for running agentic evaluations (e.g. SWE-bench Verified) with [Harbor](https://github.com/laude-institute/harbor), using either a hosted API model or the self-served Qwen3-4B from `aees/serving/`. It covers one-time setup, an oracle sanity check, task and concurrency choices, LLM agent runs, the ATIF trajectory outputs, troubleshooting and reference measurements.

## Files

| File | Purpose | Runs on |
|---|---|---|
| `setup_host.sh` | Phases 1–2: driver (R580+), CUDA 13.0 toolkit, uv venv, SGLang (pinned 0.5.21), FlashInfer wheels pinned to the installed `flashinfer-python`, model download, FlashInfer correctness check. Idempotent | GPU host |
| `serve.sh` | Server lifecycle in tmux: `start`, `stop`, `restart`, `status`, `logs`, plus `dashboard` (serves results on :8000) and `tunnel` (prints the laptop SSH command). Settings via env vars (`EXTRA_ARGS`, `SERVER_ENV`, `PORT`, `DASH_PORT`, …) | GPU host |
| `smoke_test.py` | Validation over the tunnel: plain chat, a tool call, a prefix-cache hit, then an optional interactive stdin/stdout chat with streaming, TTFT and a demo tool loop | Laptop |
| `run_bench.py` | Phase 3 matrix: restarts the server per setting, runs `sglang.bench_serving` sweeps, samples `/metrics` every second, runs the tool-call quality check. Supports `--quick`, `--dry-run`, `--resume`, `--arms` | GPU host |
| `make_dashboard.py` | Phase 3 dashboard from per-request data: KPI cards, settings-vs-baseline table, capacity curves, percentiles to p99.9, QPS sweep, cold vs warm, KV usage over time, quality table, GPU telemetry | GPU host |
| `run_phase4.sh` | Phase 4 in one command: `kernels`, `jit`, `trace`, `report` (or `all`). Keeps going past a failed stage and logs a problem summary | GPU host |
| `fi_bench.py` | Phase 4 microbenchmarks (`sparse`, `balance`, `jit`) and the report builder (`report`), at Qwen3-4B attention shapes | GPU host |
| `fi_jit_setup.sh` | Builds a "cold" venv (same torch and FlashInfer, no prebuilt kernels, private JIT cache) so compilation can be measured | GPU host |
| `fi_trace.py` | Opt-in server hook that saves FlashInfer's paged-KV index arrays every 50th `plan()` call. Loaded only by `run_phase4.sh trace`; inert unless `FI_TRACE_DIR` is set | GPU host (server process) |

## How to run

```bash
# GPU host, from geodesic/aees/serving/
bash setup_host.sh            # once per host; reboot and rerun if it installs a driver
./serve.sh start              # API on 127.0.0.1:30000
./serve.sh dashboard          # results on 127.0.0.1:8000
./serve.sh tunnel             # prints the laptop tunnel command

# Laptop
ssh -N -L 30000:127.0.0.1:30000 -L 8000:127.0.0.1:8000 ubuntu@<gpu-host-ip>
python smoke_test.py          # tests, then an optional interactive chat

# GPU host, inside tmux
python run_bench.py --quick   # ~15 min pipeline check; full run without --quick (~1.5 h)
./run_phase4.sh               # ~45-60 min
```

Dashboard: `http://127.0.0.1:8000/<run-id>/dashboard.html`. Phase 4 report: `http://127.0.0.1:8000/fi/fi_report.html`.

## Findings

### Phase 3 (A100-40GB, quick run, agent workload at 64 concurrent requests)

| Setting | Output tok/s | TTFT p99 | ITL p99 |
|---|---|---|---|
| Prefix cache off (`--disable-radix-cache`) | −59% | +424% (≈20 s) | +18% |
| FP8 KV cache (`--kv-cache-dtype fp8_e5m2`) | −5% | +2% | +17% |
| KV page size 16 | −1% | +1% | −1% |
| Triton attention instead of FlashInfer | −17% | +21% | +27% |

- **The prefix cache is the biggest lever** for agent traffic with shared system prompts and tool schemas.
- **FlashInfer beats Triton** by 17% throughput and 21–27% tail latency.
- **FP8 KV is a capacity tool, not a speed tool, on Ampere.** It roughly doubles KV pool capacity, but costs speed because the A100 has no FP8 math.
- **Page size makes no measurable difference.** FlashInfer reads scattered single-token pages efficiently.
- **TTFT, not streaming speed, limits concurrency.** ITL p99 stayed within the 50 ms SLO at every level tested.

### Phase 4 (A10)

- **JIT compilation:** each new attention variant costs about 7–15 s to compile on first use and about 1 ms afterwards. `prefill_causal` reused the module that `decode_tensorcores` had just built, since tensor-core decode runs on the prefill kernel. Production hosts avoid the compile cost by installing the prebuilt `flashinfer-cubin` and `flashinfer-jit-cache` wheels.
- **Block-sparse KV and load balancing:** see `fi_report.html` for scattered-vs-contiguous bandwidth, FP8 vs BF16 kernel time, split-KV speedups on skewed batches, `plan()` cost per layer, and the live KV index map captured from the running server.

## Design notes

- **Measurement hygiene:**
  - The prefix cache is flushed before every point.
  - Agent points are warmed with the same prompts before measuring, and a separate "cold" point keeps the empty-cache burst.
  - All percentiles are computed from per-request data.
  - Each setting differs from the baseline by exactly one flag.
- **Ports:** API on 30000, dashboard on 8000, both bound to 127.0.0.1 and reached through one SSH tunnel. Lambda's firewall only opens port 22.
- **Version pinning:** the FlashInfer kernel wheels must match `flashinfer-python` exactly, including the `-smXX` provider packages, or FlashInfer refuses to import.
- **Robustness:**
  - Scripts call each other via `bash`, so a lost execute bit doesn't break them.
  - `run_phase4.sh` puts the venv's `bin` and CUDA on `PATH`, since FlashInfer's JIT shells out to `ninja` and `nvcc`.
  - Python output is unbuffered so progress is visible.
- **Trace hook isolation:** `fi_trace.py` is linked as `sitecustomize.py` in a scratch folder outside the repo, so only the traced server process loads it.

## Testing

- Phases 1–4 run end to end on Lambda A100-40GB (Phases 1–3) and A10 (Phase 4) hosts.
- Shell scripts pass `bash -n` and `shellcheck -S warning`; Python files pass `pyflakes`.
- Dashboard and report rendering checked in headless Chromium against synthetic results, with no page errors.
- `smoke_test.py`'s streaming tool-call loop and interactive prompt tested on a pseudo-terminal against a mock OpenAI-compatible server.
- `fi_trace.py` capture tested through the symlinked `sitecustomize.py` path.

## Known limitations and follow-ups

- Phase 3 findings come from one `--quick` run. Rerun the full matrix before quoting numbers externally, and treat differences under about 5% as noise.
- Phase 4 absolute numbers are from an A10 (about 600 GB/s, 72 SMs). The patterns should carry over to the A100, but its kernels run roughly 2.5× faster.
- The proposed SLO (TTFT p99 ≤ 500 ms, ITL p99 ≤ 50 ms) still needs product sign-off. Rebuild the dashboard with different SLOs via `make_dashboard.py --slo-*`, without rerunning anything.
- Warm agent points repeat identical prompts, so even the question tokens are cached, which makes warm TTFT slightly optimistic (by a few ms).

## Reviewer checklist

- [ ] `results/` and `logs/` are not committed (they live in `~/qwen-serve`; add them to `.gitignore` if you ever point `WORK_DIR` into the repo)
- [ ] Shell scripts are executable in git: `git update-index --chmod=+x geodesic/aees/serving/*.sh`
- [ ] No hostnames, IPs or keys in the scripts (the tunnel command uses `<gpu-host-ip>`)
