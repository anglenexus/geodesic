# Serving benchmark results

One folder per run, written by `export_results.py` on the GPU host and committed from there.
Folder names are `<model>_<gpu>_<date-time>` (Phase 3 and 4 together) or `<model>_<gpu>_fi` (Phase 4 alone).
Raw per-request data stays on the GPU host (`~/qwen-serve/results/`); these files are enough to rebuild
every chart and table.

| File | What it holds |
|---|---|
| `summary.md` | Start here. The results in plain language and Markdown tables: each setting at a glance, change vs baseline, per-concurrency tables, stress tests, and the Phase 4 kernel results |
| `p3_run_log.md` | Phase 3 as it ran: server start facts per setting (start time, KV cache size, weight memory), one row per test point, and anything skipped by the 30-minute budget |
| `meta.json` | Model, GPU, software versions, the settings and their exact server flags, the test plan, export time, git commit |
| `p3_points.csv` | One row per test point (columns below) |
| `p3_tails.json` | Latency survival curves from the long p99.9 run: for each fraction of requests, the latency they exceeded |
| `p3_quality.json` | Tool-call quality per setting: 37 prompts at temperature 0 (thinking off), right tool and right arguments |
| `p3_server.json` | Per setting: KV cache size in tokens, prefix-cache hit rate, weight memory, server start time, launch command, key startup log lines |
| `p3_timeseries.csv` | SGLang's live gauges every 5 s: KV cache in use, prefix-cache hit rate, running and queued requests |
| `p3_gpu.csv` | GPU utilization, memory and power every 5 s |
| `p4_sparse.json` | FlashInfer decode kernel time and bandwidth by page size, page layout and KV format (BF16, FP8 e5m2, FP8 e4m3) |
| `p4_balance.json` | FlashInfer split-KV load balancing on skewed batches, and scheduling (`plan()`) cost |
| `p4_precision.json` | Linear-layer (GEMM) time per forward pass for BF16, FP8 and AWQ 4-bit weights, with SGLang's kernels |
| `p4_jit.json` | First-use compile time of each attention kernel variant, and what changes between two variants |
| `p4_trace.json` | Statistics of the KV-cache page indices captured from the live server, plus one binned index map |

## `p3_points.csv` columns

- `arm`: the setting (see `meta.json` → `arm_labels` / `arm_flags`); `baseline` is the default configuration
- `workload`: `agent` (16 shared 4K-token system prompts + 128-token question), `chat` (1K-token prompt), `long` (8K-token prompt); every answer is 256 tokens
- `mode` / `value`: `conc` = fixed concurrency (value = requests in flight), `rate` = Poisson arrivals (value = requests/s), `cold` = empty prefix cache, `p999` = long run for tail latency, `burst` = all requests at once (value = count)
- `num_prompts`, `completed`, `failed`, `duration_s`
- `req_s`, `out_tok_s`, `in_tok_s`: throughput summed over all requests
- `ttft_*`: time to first token, ms; `itl_*`: time between streamed tokens, ms; `e2e_*`: whole request, ms. Suffixes `p50`, `p90`, `p99`, `p999` are percentiles (p999 = 99.9th)
- `n_ttft`, `n_itl`: samples behind the percentiles (p99.9 from fewer than 1,000 requests is indicative)

## Writing about these results

Give `summary.md` and `p3_run_log.md` from each run (for example the A100 and H200 folders) to Claude as the
source material; every number in a write-up should trace back to one of these two files.
