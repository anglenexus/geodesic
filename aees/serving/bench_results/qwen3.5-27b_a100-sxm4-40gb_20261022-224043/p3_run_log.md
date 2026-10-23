# Phase 3 run log: Qwen3.5-27B on NVIDIA A100-SXM4-40GB

- Started 2026-10-22 22:41, plan `full`, time budget 40 min
- SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0)
- Server: `serve_qwen3_27B.sh` with MAX_RUNNING=32, flags `--cuda-graph-max-bs-decode 32`
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache; **mamba80**: Larger DeltaNet state cache (80 slots: ~16 running requests, less KV cache); **mtp**: MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts)
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache

Server ready in 74 s (restart included); KV pool 134,392 tokens; weights 18.8 GB; free after load 20.1 GB. Elapsed 1.5 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 61 | 0.24 | 86 | 540 | 16.0 | 20.4 | 80 |
| agent | sweep | 16 | 32 | 173 | 0.68 | 15,797 | 22,383 | 19.7 | 24.3 | 61 |
| agent | sweep | 32 | 64 | 178 | 0.69 | 38,199 | 44,176 | 19.5 | 29.3 | 105 |
| chat | sweep | 32 | 64 | 284 | 1.11 | 23,035 | 26,092 | 18.5 | 27.2 | 71 |
| long | sweep | 8 | 16 | 82 | 0.32 | 13,210 | 20,824 | 21.4 | 30.5 | 66 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).
| agent | cold start | 16 | 32 | 151 | 0.59 | 18,726 | 22,912 | 19.7 | 26.5 | 69 |
| agent | 1 req/s | open | 32 | 167 | 0.65 | 11,307 | 19,409 | 20.3 | 1,205.4 | 63 |
| agent | 4 req/s | open | 64 | 174 | 0.68 | 37,167 | 75,707 | 19.6 | 132.4 | 107 |
| agent | p99.9 run | 32 | 608 | 181 | 0.71 | 38,821 | 45,664 | 19.5 | 27.4 | 885 |
| agent | burst | all | 160 | 185 | 0.72 | 99,726 | 216,178 | 19.5 | 25.7 | 238 |

## mamba80: skipped

Time budget used (43 of 40 min).

## mtp: skipped

Time budget used (43 of 40 min).

Finished after 42.6 min.
