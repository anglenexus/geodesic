# Qwen3.5-27B on A100-SXM4-40GB

Measured 2026-10-23 with SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0). Plan: full, capped at 30 min.

How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; **ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, **p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.

## Settings tested

- **mamba80**: Larger DeltaNet state cache (80 slots: ~16 running requests, less KV cache) (`--max-mamba-cache-size 80`)
- **mtp**: MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts) (`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`)

## Each setting at a glance

Agent workload (4K-token shared system prompt, 128-token question, 256-token answer) unless noted. 1 user = concurrency 1. Peak = the highest concurrency tested for that setting.

| setting | 1 user: tok/s | 1 user: TTFT ms | peak agent throughput tok/s | at concurrency | TTFT p99 there ms | ITL p99 there ms | 8K-prompt TTFT p50 ms | KV cache tokens | weights GB | server start s | tool calls right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mamba80 | 62.5 | 1,326 | 228 | 32 | 35,528 | 34.4 | 11,828 | 61,573 | 18.8 | 75 | 100% |
| mtp | 156 | 1,368 | 128 | 32 | 62,921 | 296 | 20,542 | 90,869 | 18.8 | 90 | 100% |

## mamba80: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 48.0 | 0.2 | 1,326 | 1,348 | 16.0 | 21.0 | 5,409 | 0 |
| agent | 16 | 32 | 188 | 0.7 | 7,746 | 21,623 | 22.8 | 30.3 | 17,649 | 0 |
| agent | 32 | 64 | 228 | 0.9 | 21,928 | 35,528 | 22.6 | 34.4 | 31,238 | 0 |
| chat | 32 | 64 | 358 | 1.4 | 13,393 | 21,433 | 21.0 | 31.2 | 20,032 | 0 |
| long | 8 | 16 | 76.0 | 0.3 | 11,828 | 26,400 | 20.2 | 24.4 | 19,382 | 0 |

## mtp: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 73.7 | 0.3 | 1,368 | 1,388 | 6.4 | 25.7 | 3,533 | 0 |
| agent | 16 | 32 | 123 | 0.5 | 27,491 | 29,933 | 7.2 | 296 | 31,504 | 0 |
| agent | 32 | 64 | 128 | 0.5 | 57,295 | 62,921 | 7.2 | 296 | 62,387 | 0 |
| chat | 32 | 64 | 236 | 0.9 | 31,121 | 33,780 | 7.0 | 95.7 | 33,519 | 0 |
| long | 8 | 16 | 72.3 | 0.3 | 20,542 | 26,070 | 7.4 | 402 | 23,721 | 0 |
