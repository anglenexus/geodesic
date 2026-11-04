# Kimi K3 on B300 SXM6 AC

Measured 2026-11-03 with SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0). Plan: full.

How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; **ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, **p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.

## Settings tested

- **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe (`(baseline)`)

## Each setting at a glance

Agent workload (4K-token shared system prompt, 128-token question, 256-token answer) unless noted. 1 user = concurrency 1. Peak = the highest concurrency tested for that setting.

| setting | 1 user: tok/s | 1 user: TTFT ms | peak agent throughput tok/s | at concurrency | TTFT p99 there ms | ITL p99 there ms | 8K-prompt TTFT p50 ms | KV cache tokens | weights GB | server start s | tool calls right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 115 | 239 | 1,935 | 64 | 7,885 | 49.0 | 5,307 | 4,813,312 | 195 | 970 | 97% |

## baseline: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 104 | 0.4 | 239 | 262 | 8.7 | 9.3 | 2,459 | 0 |
| agent | 16 | 48 | 908 | 3.5 | 477 | 732 | 15.5 | 41.6 | 4,467 | 0 |
| agent | 64 | 192 | 1,935 | 7.6 | 1,103 | 7,885 | 21.2 | 49.0 | 6,585 | 0 |
| chat | 64 | 128 | 1,483 | 5.8 | 2,662 | 10,563 | 22.6 | 46.1 | 8,370 | 0 |
| long | 16 | 32 | 441 | 1.7 | 5,307 | 5,452 | 15.3 | 40.7 | 9,265 | 0 |

## Stress tests (agent workload)

p99.9 from fewer than 1,000 requests is indicative only.

| setting | test | concurrency or rate | requests | failed | tok/s | TTFT p50 ms | TTFT p99 ms | TTFT p99.9 ms | ITL p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline | cold start (empty prefix cache) | 16 | 32 | 0 | 663 | 1,991 | 2,617 | 2,617 | 35.7 |
| baseline | long run for p99.9 | 64 | 400 | 0 | 2,062 | 684 | 7,207 | 7,210 | 50.0 |
| baseline | Poisson arrivals, req/s | 4 | 64 | 0 | 875 | 565 | 1,866 | 1,898 | 248 |
