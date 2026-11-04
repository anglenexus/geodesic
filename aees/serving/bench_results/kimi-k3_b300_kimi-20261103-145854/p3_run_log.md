# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:35, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 1 | 16 | 104 | 0.41 | 239 | 262 | 8.7 | 9.3 | 54 |

Finished after 1.8 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:37, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 16 | 48 | 908 | 3.55 | 477 | 732 | 15.5 | 41.6 | 29 |

Finished after 1.2 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:39, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | sweep | 64 | 192 | 1,935 | 7.56 | 1,103 | 7,885 | 21.2 | 49.0 | 41 |

Finished after 1.4 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:41, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| chat | sweep | 64 | 128 | 1,483 | 5.79 | 2,662 | 10,563 | 22.6 | 46.1 | 37 |

Finished after 1.0 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:42, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| long | sweep | 16 | 32 | 441 | 1.72 | 5,307 | 5,452 | 15.3 | 40.7 | 33 |

Finished after 0.9 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:44, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|

Tool-call quality: right tool 97%, right tool and arguments 97% (37 prompts, temperature 0).

Finished after 0.5 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:45, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | cold start | 16 | 32 | 663 | 2.59 | 1,991 | 2,617 | 15.5 | 35.7 | 27 |

Finished after 0.8 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:46, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | 4 req/s | open | 64 | 875 | 3.42 | 565 | 1,866 | 21.4 | 248.1 | 34 |

Finished after 1.3 min.

---
# Phase 3 run log: Kimi K3 on NVIDIA B300 SXM6 AC

- Started 2026-11-03 23:48, plan `full`, time budget none
- SGLang 0.5.23, FlashInfer 0.7.2, torch 2.14.1+cu130 (CUDA 13.0)
- Server: `serve_kimi_k3.sh` with MAX_RUNNING=128
- Workloads: agent (4K shared system prompt + 128-token question); chat (1K in); long context (8K in); 256 output tokens each
- Settings: **baseline**: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe
- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile


## baseline: Native MXFP4 experts (rest BF16), TP8 + DCP8: SGLang's B300 Balanced recipe

Reused the server that was already running; KV pool 4,813,312 tokens; weights 195.3 GB; free after load 69.3 GB. Elapsed 0.4 min.

| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| agent | p99.9 run | 64 | 400 | 2,062 | 8.05 | 684 | 7,207 | 21.5 | 50.0 | 66 |

Finished after 1.9 min.
