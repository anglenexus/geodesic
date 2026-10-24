# Phase 3 run log: Qwen3.5-27B on NVIDIA H200

- Started 2026-10-22 20:50, plan `full`, time budget 30 min
- SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0)
- Server: `serve_qwen3_27B.sh` with MAX_RUNNING=64, flags `--cuda-graph-max-bs-decode 64`
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache; **fp8_weights**: FP8 weights (official Qwen FP8: W8A8, 128x128 block scales); **bf16_weights**: BF16 weights (official Qwen checkpoint, full precision)
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache

Server ready in 65 s (restart included); KV pool 894,521 tokens; weights 18.8 GB; free after load 120.3 GB. Elapsed 1.3 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 116 | 0.45 | 66 | 106 | 8.4 | 9.3 | 46 |
| agent | sweep | 16 | 64 | 1,228 | 4.80 | 300 | 469 | 11.6 | 16.2 | 24 |
| agent | sweep | 64 | 192 | 1,970 | 7.70 | 1,248 | 2,024 | 22.7 | 27.4 | 38 |
| chat | sweep | 64 | 128 | 934 | 3.65 | 5,948 | 16,368 | 21.3 | 25.2 | 45 |
| long | sweep | 16 | 32 | 161 | 0.63 | 13,223 | 22,802 | 12.8 | 18.1 | 62 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).
| agent | cold start | 16 | 48 | 470 | 1.84 | 2,650 | 9,122 | 11.6 | 16.0 | 38 |
| agent | 4 req/s | open | 64 | 979 | 3.83 | 98 | 281 | 12.3 | 61.4 | 26 |
| agent | 16 req/s | open | 160 | 1,915 | 7.48 | 3,868 | 7,842 | 22.6 | 172.4 | 34 |
| agent | p99.9 run | 64 | 1008 | 2,044 | 7.98 | 1,591 | 2,832 | 22.7 | 29.2 | 150 |
| agent | burst | all | 304 | 1,958 | 7.65 | 18,025 | 33,818 | 22.5 | 28.5 | 54 |

## fp8_weights: FP8 weights (official Qwen FP8: W8A8, 128x128 block scales)

Server ready in 223 s (restart included); KV pool 809,196 tokens; weights 28.5 GB; free after load 110.7 GB. Elapsed 18.6 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 96 | 0.38 | 64 | 124 | 10.2 | 10.8 | 52 |
| agent | sweep | 16 | 64 | 1,180 | 4.61 | 269 | 334 | 12.3 | 16.9 | 24 |
| agent | sweep | 64 | 192 | 2,449 | 9.57 | 901 | 5,963 | 17.5 | 23.4 | 30 |
| chat | sweep | 64 | 128 | 1,753 | 6.85 | 2,091 | 8,094 | 16.1 | 21.8 | 28 |
| long | sweep | 16 | 32 | 352 | 1.38 | 4,873 | 8,410 | 13.6 | 17.9 | 33 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

## bf16_weights: BF16 weights (official Qwen checkpoint, full precision)

Server ready in 63 s (restart included); KV pool 615,355 tokens; weights 51.0 GB; free after load 88.1 GB. Elapsed 23.7 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 67 | 0.26 | 68 | 105 | 14.7 | 15.5 | 71 |
| agent | sweep | 16 | 64 | 859 | 3.36 | 293 | 361 | 17.3 | 21.3 | 29 |
| agent | sweep | 64 | 192 | 1,527 | 5.96 | 1,214 | 8,716 | 21.9 | 26.5 | 43 |
| chat | sweep | 64 | 128 | 1,202 | 4.70 | 3,503 | 11,468 | 20.3 | 27.0 | 38 |
| long | sweep | 16 | 32 | 242 | 0.94 | 7,385 | 12,588 | 18.6 | 23.0 | 45 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

Finished after 29.1 min.
