# Phase 3 run log: Qwen3.5-27B on NVIDIA A100-SXM4-40GB

- Started 2026-10-23 02:57, plan `full`, time budget 30 min
- SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0)
- Server: `serve_qwen3_27B.sh` with MAX_RUNNING=32, flags `--cuda-graph-max-bs-decode 32`
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **mamba80**: Larger DeltaNet state cache (80 slots: ~16 running requests, less KV cache); **mtp**: MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts)
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## mamba80: Larger DeltaNet state cache (80 slots: ~16 running requests, less KV cache)

Server ready in 75 s (restart included); KV pool 61,573 tokens; weights 18.8 GB; free after load 20.1 GB. Elapsed 1.5 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 48 | 0.19 | 1,326 | 1,348 | 16.0 | 21.0 | 99 |
| agent | sweep | 16 | 32 | 188 | 0.73 | 7,746 | 21,623 | 22.8 | 30.3 | 58 |
| agent | sweep | 32 | 64 | 228 | 0.89 | 21,928 | 35,528 | 22.6 | 34.4 | 85 |
| chat | sweep | 32 | 64 | 358 | 1.40 | 13,393 | 21,433 | 21.0 | 31.2 | 59 |
| long | sweep | 8 | 16 | 76 | 0.30 | 11,828 | 26,400 | 20.2 | 24.4 | 69 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

## mtp: MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts)

Server ready in 99 s (restart included); KV pool 90,869 tokens; weights 18.8 GB; free after load 20.1 GB. Elapsed 12.8 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 74 | 0.29 | 1,368 | 1,388 | 6.4 | 25.7 | 70 |
| agent | sweep | 16 | 32 | 123 | 0.48 | 27,491 | 29,933 | 7.2 | 295.5 | 80 |
| agent | sweep | 32 | 64 | 128 | 0.50 | 57,295 | 62,921 | 7.2 | 296.3 | 141 |
| chat | sweep | 32 | 64 | 236 | 0.92 | 31,121 | 33,780 | 7.0 | 95.7 | 82 |
| long | sweep | 8 | 16 | 72 | 0.28 | 20,542 | 26,070 | 7.4 | 402.0 | 72 |

Tool-call quality: right tool 100%, right tool and arguments 100% (37 prompts, temperature 0).

Finished after 24.6 min.
