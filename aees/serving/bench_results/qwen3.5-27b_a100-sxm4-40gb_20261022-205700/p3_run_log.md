# Phase 3 run log: Qwen3.5-27B on NVIDIA A100-SXM4-40GB

- Started 2026-10-22 20:57, plan `full`, time budget 30 min
- SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0)
- Server: `serve_qwen3_27B.sh` with MAX_RUNNING=32, flags `--cuda-graph-max-bs-decode 32`
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache; **no_radix**: Prefix (radix) cache off; **mtp**: MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts)
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache

Server ready in 402 s (restart included); KV pool 134,392 tokens; weights 18.8 GB; free after load 20.1 GB. Elapsed 7.0 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 61 | 0.24 | 86 | 540 | 16.0 | 20.6 | 81 |
| agent | sweep | 16 | 32 | 173 | 0.67 | 15,920 | 22,366 | 19.7 | 25.0 | 61 |
| agent | sweep | 32 | 64 | 177 | 0.69 | 38,300 | 44,427 | 19.6 | 24.6 | 107 |
| chat | sweep | 32 | 64 | 283 | 1.11 | 23,065 | 26,124 | 18.6 | 25.9 | 71 |
| long | sweep | 8 | 16 | 82 | 0.32 | 13,832 | 20,815 | 21.4 | 28.4 | 65 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

Skipped the cold-start, QPS, p99.9 and burst tests: 13 min left, the other 2 setting(s) need ~16.

## no_radix: Prefix (radix) cache off

Server ready in 81 s (restart included); KV pool 174,325 tokens; weights 18.8 GB; free after load 20.1 GB. Elapsed 18.3 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 47 | 0.19 | 1,321 | 1,336 | 16.0 | 19.1 | 100 |
| agent | sweep | 16 | 32 | 153 | 0.60 | 15,201 | 20,662 | 23.9 | 31.5 | 67 |
| agent | sweep | 32 | 64 | 158 | 0.62 | 24,529 | 50,765 | 32.9 | 39.8 | 118 |
| chat | sweep | 32 | 64 | 425 | 1.66 | 6,130 | 17,595 | 27.9 | 37.3 | 51 |
| long | sweep | 8 | 16 | 78 | 0.31 | 14,267 | 20,773 | 21.4 | 28.5 | 67 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

## mtp: skipped

Time budget used (29 of 30 min).

Finished after 29.4 min.
