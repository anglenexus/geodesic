# Qwen3.5-27B on A100-SXM4-40GB

Measured 2026-10-22 with SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0). Plan: full, capped at 30 min.

How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; **ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, **p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.

## Settings tested

- **baseline**: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache (`(baseline)`)
- **no_radix**: Prefix (radix) cache off (`--disable-radix-cache`)

## Each setting at a glance

Agent workload (4K-token shared system prompt, 128-token question, 256-token answer) unless noted. 1 user = concurrency 1. Peak = the highest concurrency tested for that setting.

| setting | 1 user: tok/s | 1 user: TTFT ms | peak agent throughput tok/s | at concurrency | TTFT p99 there ms | ITL p99 there ms | 8K-prompt TTFT p50 ms | KV cache tokens | weights GB | server start s | tool calls right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 62.5 | 86.3 | 177 | 32 | 44,427 | 24.6 | 13,832 | 134,392 | 18.8 | 390 | 100% |
| no_radix | 62.6 | 1,321 | 158 | 32 | 50,765 | 39.8 | 14,267 | 174,325 | 18.8 | 80 | 100% |

## Change vs baseline at each concurrency

Positive throughput is better; negative latency is better. Differences under ~5% are within run-to-run noise.

| workload | concurrency | setting | throughput | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|---|---|
| agent | 1 | no_radix | -22% | +1429% | +147% | -0% | -7% |
| agent | 16 | no_radix | -11% | -5% | -8% | +21% | +26% |
| agent | 32 | no_radix | -11% | -36% | +14% | +68% | +62% |
| chat | 32 | no_radix | +50% | -73% | -33% | +50% | +44% |
| long | 8 | no_radix | -4% | +3% | -0% | -0% | +1% |

## baseline: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 60.9 | 0.2 | 86.3 | 540 | 16.0 | 20.6 | 4,165 | 0 |
| agent | 16 | 32 | 173 | 0.7 | 15,920 | 22,366 | 19.7 | 25.0 | 23,693 | 0 |
| agent | 32 | 64 | 177 | 0.7 | 38,300 | 44,427 | 19.6 | 24.6 | 46,154 | 0 |
| chat | 32 | 64 | 283 | 1.1 | 23,065 | 26,124 | 18.6 | 25.9 | 28,911 | 0 |
| long | 8 | 16 | 82.0 | 0.3 | 13,832 | 20,815 | 21.4 | 28.4 | 24,968 | 0 |

## no_radix: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 47.4 | 0.2 | 1,321 | 1,336 | 16.0 | 19.1 | 5,396 | 0 |
| agent | 16 | 32 | 153 | 0.6 | 15,201 | 20,662 | 23.9 | 31.5 | 26,700 | 0 |
| agent | 32 | 64 | 158 | 0.6 | 24,529 | 50,765 | 32.9 | 39.8 | 48,369 | 0 |
| chat | 32 | 64 | 425 | 1.7 | 6,130 | 17,595 | 27.9 | 37.3 | 16,916 | 0 |
| long | 8 | 16 | 78.4 | 0.3 | 14,267 | 20,773 | 21.4 | 28.5 | 26,089 | 0 |

## Phase 4: weight precision in the linear layers

Time the model's weight matrices take per forward pass (all layers), measured with the kernels SGLang uses: BF16 (W16A16) = torch F.linear (cuBLAS). 1-64 tokens is decoding (reading weights dominates), 1024+ is prefill (math dominates).

| tokens per forward pass | bf16 ms |
|---|---|
| 1 | 37.8 |
| 16 | 38.5 |
| 64 | 43.5 |
| 256 | 62.0 |
| 1024 | 202 |
| 4096 | 764 |

## Phase 4: attention decode kernel time by KV-cache format (µs)

FlashInfer batch decode at qwen3.5-27b attention shapes {'qh': 24, 'kvh': 4, 'hd': 256, 'layers': 16}; peak memory bandwidth 1555 GB/s.

| batch | kv_len | page_size | layout | bf16 | fp8_e4m3 | fp8_e5m2 |
|---|---|---|---|---|---|---|
| 16 | 16384 | 1 | contiguous | 856 | 10,413 | 2,062 |
| 16 | 16384 | 1 | scattered | 860 | 10,417 | 2,063 |
| 16 | 16384 | 16 | contiguous | 855 | 10,418 | 2,074 |
| 16 | 16384 | 16 | scattered | 864 | 10,411 | 2,076 |
| 64 | 4096 | 1 | contiguous | 1,011 | 10,419 | 2,738 |
| 64 | 4096 | 1 | scattered | 901 | 10,422 | 2,128 |
| 64 | 4096 | 16 | contiguous | 899 | 10,419 | 2,127 |
| 64 | 4096 | 16 | scattered | 900 | 10,413 | 2,127 |
| 256 | 1024 | 1 | contiguous | 872 | 10,563 | 1,946 |
| 256 | 1024 | 1 | scattered | 865 | 10,588 | 1,952 |
| 256 | 1024 | 16 | contiguous | 871 | 10,493 | 1,940 |
| 256 | 1024 | 16 | scattered | 872 | 10,636 | 1,961 |

## Phase 4: JIT compile cost per attention variant

| variant | what changes | first call s (compile) | later calls s |
|---|---|---|---|
| decode_base | Batch decode, BF16, head_dim 256 (Qwen3.5-27B) | 10.4 | 0.0 |
| decode_softcap | + logits soft-cap | 10.3 | 0.0 |
| decode_rope | + fused RoPE (ROPE_LLAMA) | 10.3 | 0.0 |
| decode_window | + sliding window (256) | 10.2 | 0.0 |
| decode_fp8kv | FP8 e5m2 KV cache | 11.8 | 0.0 |
| decode_hd64 | head_dim 64 | 10.3 | 0.0 |
| decode_tensorcores | Decode on tensor cores (prefill kernel) | 17.8 | 0.0 |
| prefill_causal | Batch prefill, causal, paged KV | 0.0 | 0.0 |
| single_decode | Single-request decode | 8.8 | 0.0 |
| single_prefill | Single-request prefill, causal | 15.9 | 0.0 |
