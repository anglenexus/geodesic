# qwen3.5-27b on A100-SXM4-40GB

Measured ? with SGLang ?, FlashInfer 0.6.18, torch 2.13.0+cu130 (CUDA 13.0). Plan: full.

How to read this: **TTFT** (time to first token) is how long a user waits before the answer starts; **ITL** (inter-token latency) is the gap between streamed tokens, so 1000 / ITL is one user's reading speed in tokens/s; **throughput** is output tokens/s summed over all users. **p50** is the typical request, **p99** the slowest 1 in 100. **Concurrency** is how many requests are in flight at once.

## Phase 4: weight precision in the linear layers

Time the model's weight matrices take per forward pass (all layers), measured with the kernels SGLang uses: BF16 (W16A16) = torch F.linear (cuBLAS); AWQ 4-bit W4A16 (Marlin) = sglang apply_gptq_marlin_linear (Marlin W4A16). 1-64 tokens is decoding (reading weights dominates), 1024+ is prefill (math dominates).

| tokens per forward pass | bf16 ms | awq ms | awq speedup |
|---|---|---|---|
| 1 | 37.8 | 13.6 | 2.79x |
| 16 | 38.4 | 14.1 | 2.72x |
| 64 | 44.9 | 21.3 | 2.11x |
| 256 | 63.0 | 64.7 | 0.97x |
| 1024 | 199 | 249 | 0.80x |
| 4096 | 759 | 994 | 0.76x |

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
