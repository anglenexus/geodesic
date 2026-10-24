# Qwen3.5-27B on H200

Measured 2026-10-22 with SGLang 0.5.21, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0). Plan: full, capped at 30 min.

How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; **ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, **p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.

## Settings tested

- **baseline**: AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache (`(baseline)`)
- **fp8_weights**: FP8 weights (official Qwen FP8: W8A8, 128x128 block scales) (`WEIGHTS=fp8`)
- **bf16_weights**: BF16 weights (official Qwen checkpoint, full precision) (`WEIGHTS=bf16`)

## Each setting at a glance

Agent workload (4K-token shared system prompt, 128-token question, 256-token answer) unless noted. 1 user = concurrency 1. Peak = the highest concurrency tested for that setting.

| setting | 1 user: tok/s | 1 user: TTFT ms | peak agent throughput tok/s | at concurrency | TTFT p99 there ms | ITL p99 there ms | 8K-prompt TTFT p50 ms | KV cache tokens | weights GB | server start s | tool calls right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 119 | 66.4 | 1,970 | 64 | 2,024 | 27.4 | 13,223 | 894,521 | 18.8 | 65 | 100% |
| fp8_weights | 98.2 | 64.3 | 2,449 | 64 | 5,963 | 23.4 | 4,873 | 809,196 | 28.5 | 220 | 100% |
| bf16_weights | 68.2 | 67.9 | 1,527 | 64 | 8,716 | 26.5 | 7,385 | 615,355 | 51.0 | 60 | 100% |

## Change vs baseline at each concurrency

Positive throughput is better; negative latency is better. Differences under ~5% are within run-to-run noise.

| workload | concurrency | setting | throughput | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|---|---|
| agent | 1 | fp8_weights | -17% | -3% | +17% | +21% | +16% |
| agent | 1 | bf16_weights | -42% | +2% | -1% | +74% | +66% |
| agent | 16 | fp8_weights | -4% | -11% | -29% | +6% | +4% |
| agent | 16 | bf16_weights | -30% | -2% | -23% | +50% | +31% |
| agent | 64 | fp8_weights | +24% | -28% | +195% | -23% | -14% |
| agent | 64 | bf16_weights | -22% | -3% | +331% | -3% | -3% |
| chat | 64 | fp8_weights | +88% | -65% | -51% | -25% | -13% |
| chat | 64 | bf16_weights | +29% | -41% | -30% | -5% | +7% |
| long | 16 | fp8_weights | +118% | -63% | -63% | +6% | -1% |
| long | 16 | bf16_weights | +50% | -44% | -45% | +45% | +27% |

## baseline: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 116 | 0.5 | 66.4 | 106 | 8.4 | 9.3 | 2,213 | 0 |
| agent | 16 | 64 | 1,228 | 4.8 | 300 | 469 | 11.6 | 16.2 | 3,330 | 0 |
| agent | 64 | 192 | 1,970 | 7.7 | 1,248 | 2,024 | 22.7 | 27.4 | 7,769 | 0 |
| chat | 64 | 128 | 934 | 3.6 | 5,948 | 16,368 | 21.3 | 25.2 | 16,269 | 0 |
| long | 16 | 32 | 161 | 0.6 | 13,223 | 22,802 | 12.8 | 18.1 | 25,388 | 0 |

## fp8_weights: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 96.1 | 0.4 | 64.3 | 124 | 10.2 | 10.8 | 2,658 | 0 |
| agent | 16 | 64 | 1,180 | 4.6 | 269 | 334 | 12.3 | 16.9 | 3,477 | 0 |
| agent | 64 | 192 | 2,449 | 9.6 | 901 | 5,963 | 17.5 | 23.4 | 5,575 | 0 |
| chat | 64 | 128 | 1,753 | 6.8 | 2,091 | 8,094 | 16.1 | 21.8 | 7,690 | 0 |
| long | 16 | 32 | 352 | 1.4 | 4,873 | 8,410 | 13.6 | 17.9 | 11,616 | 0 |

## bf16_weights: throughput and latency by concurrency

| workload | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | request p50 ms | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 1 | 16 | 67.2 | 0.3 | 67.9 | 105 | 14.7 | 15.5 | 3,805 | 0 |
| agent | 16 | 64 | 859 | 3.4 | 293 | 361 | 17.3 | 21.3 | 4,769 | 0 |
| agent | 64 | 192 | 1,527 | 6.0 | 1,214 | 8,716 | 21.9 | 26.5 | 6,834 | 0 |
| chat | 64 | 128 | 1,202 | 4.7 | 3,503 | 11,468 | 20.3 | 27.0 | 9,325 | 0 |
| long | 16 | 32 | 242 | 0.9 | 7,385 | 12,588 | 18.6 | 23.0 | 16,945 | 0 |

## Stress tests (agent workload)

p99.9 from fewer than 1,000 requests is indicative only.

| setting | test | concurrency or rate | requests | failed | tok/s | TTFT p50 ms | TTFT p99 ms | TTFT p99.9 ms | ITL p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| baseline | burst (all requests at once) | 300 | 304 | 0 | 1,958 | 18,025 | 33,818 | 33,823 | 28.5 |
| baseline | cold start (empty prefix cache) | 16 | 48 | 0 | 470 | 2,650 | 9,122 | 9,122 | 16.0 |
| baseline | long run for p99.9 | 64 | 1008 | 0 | 2,044 | 1,591 | 2,832 | 8,464 | 29.2 |
| baseline | Poisson arrivals, req/s | 4 | 64 | 0 | 979 | 98.3 | 281 | 309 | 61.4 |
| baseline | Poisson arrivals, req/s | 16 | 160 | 0 | 1,915 | 3,868 | 7,842 | 7,854 | 172 |

## Phase 4: weight precision in the linear layers

Time the model's weight matrices take per forward pass (all layers), measured with the kernels SGLang uses: BF16 (W16A16) = torch F.linear (cuBLAS); FP8 W8A8, 128x128 blocks = sglang triton_w8a8_block_fp8_linear; AWQ 4-bit W4A16 (Marlin) = sglang apply_gptq_marlin_linear (Marlin W4A16). 1-64 tokens is decoding (reading weights dominates), 1024+ is prefill (math dominates).

| tokens per forward pass | bf16 ms | fp8 ms | fp8 speedup | awq ms | awq speedup |
|---|---|---|---|---|---|
| 1 | 12.9 | 24.2 | 0.53x | 7.2 | 1.78x |
| 16 | 13.1 | 19.4 | 0.68x | 7.6 | 1.73x |
| 64 | 13.6 | 19.4 | 0.70x | 12.7 | 1.07x |
| 256 | 18.0 | 26.2 | 0.69x | 35.6 | 0.50x |
| 1024 | 72.6 | 77.2 | 0.94x | 144 | 0.50x |
| 4096 | 283 | 291 | 0.97x | 577 | 0.49x |

## Phase 4: attention decode kernel time by KV-cache format (µs)

FlashInfer batch decode at qwen3.5-27b attention shapes {'qh': 24, 'kvh': 4, 'hd': 256, 'layers': 16}; peak memory bandwidth 4800 GB/s.

| batch | kv_len | page_size | layout | bf16 | fp8_e4m3 | fp8_e5m2 |
|---|---|---|---|---|---|---|
| 16 | 16384 | 1 | contiguous | 586 | 800 | 800 |
| 16 | 16384 | 1 | scattered | 586 | 800 | 799 |
| 16 | 16384 | 16 | contiguous | 587 | 807 | 807 |
| 16 | 16384 | 16 | scattered | 589 | 807 | 808 |
| 64 | 4096 | 1 | contiguous | 593 | 792 | 792 |
| 64 | 4096 | 1 | scattered | 593 | 792 | 792 |
| 64 | 4096 | 16 | contiguous | 592 | 798 | 797 |
| 64 | 4096 | 16 | scattered | 593 | 798 | 798 |
| 256 | 1024 | 1 | contiguous | 568 | 775 | 775 |
| 256 | 1024 | 1 | scattered | 568 | 776 | 775 |
| 256 | 1024 | 16 | contiguous | 568 | 775 | 776 |
| 256 | 1024 | 16 | scattered | 568 | 776 | 775 |

## Phase 4: JIT compile cost per attention variant

| variant | what changes | first call s (compile) | later calls s |
|---|---|---|---|
| decode_base | Batch decode, BF16, head_dim 256 (Qwen3.5-27B) | 8.1 | 0.0 |
| decode_softcap | + logits soft-cap | 7.9 | 0.0 |
| decode_rope | + fused RoPE (ROPE_LLAMA) | 8.0 | 0.0 |
| decode_window | + sliding window (256) | 8.0 | 0.0 |
| decode_fp8kv | FP8 e5m2 KV cache | 8.4 | 0.0 |
| decode_hd64 | head_dim 64 | 7.9 | 0.0 |
| decode_tensorcores | Decode on tensor cores (prefill kernel) | 16.6 | 0.0 |
| prefill_causal | Batch prefill, causal, paged KV | 69.4 | 0.0 |
| single_decode | Single-request decode | 6.5 | 0.0 |
| single_prefill | Single-request prefill, causal | 52.4 | 0.0 |
