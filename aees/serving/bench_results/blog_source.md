# Blog source: practical Qwen3.5-27B serving on an A100-40GB and an H200

This file is source material for a Medium post. Upload it to Claude on the web with a request such as
"Write the Medium post described in the brief below, using only the facts and numbers in this file."
Every number here comes from the committed results in `aees/serving/bench_results/` (branch
`aees-serving-qwen27b`, PR #22) and the interactive dashboard (link in section 9).

## 1. Brief for the writer

- **Working title:** Serving a capable open-source agent model on GPUs you can actually rent: Qwen3.5-27B on an A100-40GB and an H200
- **Thesis:** a capable open-source model with strong agentic (tool-calling) skills can be served on widely
  available GPUs with a short, reproducible setup. This is not a toy: the service has a prefix-aware KV cache, just-in-time compiled attention kernels,
  memory and device management, tool-call and reasoning parsing, and metrics, built on SGLang and FlashInfer.
  From a GPU machine to a serving endpoint takes under 10 minutes (section 4 lists what we measured).
  We measure it the way you would before putting it in front of users: latency, throughput, concurrency,
  weight precision, stress tests and the trade-offs between them.
- **Audience:** ML and platform engineers who want to self-host an open model for agents, and technical leads deciding which GPU to rent.
- **Tone:** practical, plain, confident; show the numbers and explain why they come out that way. No hype.
- **Length:** 1,800–2,500 words plus 5–7 figures and 2–3 tables.
- **Sign-off:** "With love from the Geodesic team."
- **Suggested structure:**
  1. Why this model, why these GPUs (hook: a 27B model that scores in the 40s on Terminal-Bench, served from one GPU).
  2. The stack in one picture: SGLang (scheduler, caches, parsers) on top of FlashInfer (attention kernels).
  3. Setup: two scripts, a running OpenAI-compatible endpoint (section 4).
  4. One user: TTFT and ITL on each GPU (T1).
  5. Many users: throughput and waiting time under concurrency (T2, figures F2–F3).
  6. Weight precision on the H200: AWQ 4-bit vs FP8 vs BF16 (T2, T5).
  7. The A100's memory budget: KV cache vs DeltaNet state vs speculative decoding (T3, F5). This is the most original finding.
  8. Stress tests: cold start, arrival rates, p99.9, burst (T4).
  9. Under the hood: FlashInfer KV formats and JIT compilation (T6, T7).
  10. Recommendations: which GPU and settings for which job (section 7).
  11. How to reproduce.

## 2. The model

- **Qwen3.5-27B** (Hugging Face `Qwen/Qwen3.5-27B`; the user-facing name in our scripts is "Qwen3-27B"): a 27B-parameter dense hybrid model.
  - 64 layers: **48 Gated DeltaNet (linear attention) layers** and **16 full-attention layers** (24 query heads, 4 KV heads, head size 256).
    Hidden size 5,120, MLP size 17,408, vocabulary 248,320. It ships a multi-token-prediction (MTP) head usable for speculative decoding.
  - Why the hybrid matters for serving: only the 16 attention layers keep a KV cache, about **64 KiB per token**,
    so long contexts are cheap. Each DeltaNet layer keeps a fixed-size recurrent state per request instead, about **150 MB per request** in total.
    That state becomes the binding limit on a 40 GB card (section 6.4).
- **Published agentic scores (thinking mode, from Qwen's release):** SWE-bench Verified 72.4, Terminal-Bench 2.0 41.6 (256K context, 80K max output tokens).
  Writer: cite Qwen's model card for these and check them before publishing.
- **Tool calling in our runs:** 37 tool-call prompts at temperature 0, scored for the right tool and the right arguments:
  **100% / 100% in every one of the 7 settings** (3 precisions on the H200, 4 settings on the A100). Thinking was off for this check.
- **Checkpoints used:**
  - AWQ 4-bit: `cyankiwi/Qwen3.5-27B-AWQ-4bit` (W4A16, compressed-tensors, Marlin kernels), 18.8 GB loaded. The only one that leaves room for a cache on 40 GB.
    (The official GPTQ-Int4 is ~30 GB because it keeps attention in BF16: too big for 40 GB with a cache.)
  - FP8: `Qwen/Qwen3.5-27B-FP8` (W8A8, 128×128 block scales), 28.5 GB loaded.
  - BF16: `Qwen/Qwen3.5-27B`, 51.0 GB loaded.

## 3. The serving stack

- **SGLang 0.5.21** is the server: an OpenAI-compatible API, a continuous-batching scheduler, and:
  - **Prefix (radix) cache:** requests that share a prompt prefix (an agent's system prompt and tool schemas) reuse its KV cache instead of recomputing it.
    On the H200 the agent workload hit the cache for 96.6% of prompt tokens.
  - **Chunked prefill** (4,096 tokens): long prompts are processed in slices between decode steps, so one 8K prompt does not freeze everyone else's stream.
  - **CUDA graphs** for decode batches up to the tested maximum (32 on the A100, 64 on the H200): removes per-step CPU launch overhead.
  - **Memory manager:** `--mem-fraction-static 0.88` gives 88% of GPU memory to weights plus caches. SGLang splits what is left after the weights between the KV pool
    (for the attention layers) and a pool of DeltaNet state slots ("mamba cache" in SGLang's logs), and sets the running-request cap from them.
  - **Parsers:** `--reasoning-parser qwen3` puts thinking in `reasoning_content`; `--tool-call-parser qwen3_coder` turns the model's tool-call text into OpenAI `tool_calls`.
  - **Metrics:** Prometheus gauges (running and queued requests, cache hit rate, token usage, speculative acceptance) that we sampled every 5 s.
  - **Speculative decoding (MTP / NEXTN):** the model's own MTP head drafts tokens that the full model verifies in one pass.
- **FlashInfer 0.6.18** provides the attention kernels and sampling:
  - **Paged KV cache:** the KV cache is a pool of pages indexed per request, so memory is not reserved for the worst case and shared prefixes are just shared page indices.
    Our trace of the live index confirmed whole-request runs of contiguous pages (mean run length 1,500–2,400 tokens) under load.
  - **Plan / run split and load balancing:** a planning step splits long sequences across SMs (split-KV) so a batch of mixed lengths keeps the GPU busy.
  - **JIT compilation:** each kernel variant (head size, KV format, masking, positional encoding) is compiled on first use and cached on disk (T7).
    This is why the very first server start on a fresh A100 took 402 s while later restarts take 65–90 s.
  - **KV-cache formats:** BF16, FP8 e5m2 and FP8 e4m3 (T6).
- **Device stack:** NVIDIA driver R580 (open kernel modules, headless), CUDA 13.0, torch 2.13, Python 3.12, Ubuntu 22.04 on both machines.
  One GPU each, no NVSwitch, no tensor parallelism.

## 4. Setup

```bash
git clone <repo> && cd geodesic/aees/serving
bash setup_host_qwen3_27B.sh      # driver (if needed), CUDA 13, uv venv, SGLang + FlashInfer, downloads the weights,
                                  # checks SGLang knows the architecture and that each checkpoint fits this GPU
./serve_qwen3_27B.sh start        # starts SGLang in tmux; WEIGHTS=fp8 or bf16 on an 80 GB+ GPU
python smoke_test.py              # streams an answer (thinking shown dimmed) and checks a tool call
```

The launch command the script runs (H200, AWQ):

```bash
sglang serve ~/models/Qwen3.5-27B-AWQ-4bit \
  --served-model-name qwen3.5-27b --host 127.0.0.1 --port 30000 \
  --attention-backend flashinfer --sampling-backend flashinfer \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --context-length 32768 --mem-fraction-static 0.88 \
  --max-running-requests 64 --chunked-prefill-size 4096 \
  --cuda-graph-max-bs-decode 64 --enable-metrics --enable-cache-report
```

**The "under 10 minutes" claim.** Use this wording: "from a GPU machine to a serving endpoint in under 10 minutes".
Facts we measured, and what we did not:
- The first server start on a fresh A100, including FlashInfer JIT compilation and CUDA-graph capture, took **402 s (6.7 min)**. This is the largest single step.
- Later restarts, with the JIT cache warm, took **65 s (H200, AWQ) and 75 s (A100, AWQ)**, and 60–220 s across all 7 settings.
  FP8's 220 s includes DeepGEMM kernel compilation on first use.
- The weights' load time inside a start was 4.2 s for AWQ (from SGLang's log).
- We did not time:
  - the download of the AWQ weights (about 20 GB, a few minutes on a typical cloud link)
  - the install of SGLang and FlashInfer into the venv
  - the GPU driver install (not needed on images that ship an R580 driver)
- The 6.7-minute first start leaves about 3 minutes for the download and the install, so "under 10 minutes" is a fair claim for a cloud image that already has a recent NVIDIA driver.
  We have not timed the whole sequence in one go. A machine that needs a driver install and a reboot takes longer. Writer: say this in one sentence.
- Clients reach it like any OpenAI endpoint (`base_url=http://<host>:30000/v1`, model `qwen3.5-27b`). We drive Harbor's terminus-2 agent with it for Terminal-Bench runs.

## 5. Hardware

| | A100-SXM4-40GB | H200 (SXM, 141 GB) |
|---|---|---|
| Architecture | Ampere (sm80), 108 SMs | Hopper (sm90), 132 SMs |
| Memory | 40 GB HBM2 | 141 GB HBM3e |
| Memory bandwidth | 1.56 TB/s | 4.8 TB/s |
| FP8 tensor cores | no | yes |
| Fits | AWQ 4-bit only, with a cache | all three precisions |
| Role in this post | the widely available, inexpensive card | the high-end single GPU |

| On-demand price we paid | **$1.99 / hour (Lambda)** | **$6.56 / hour (Nebius)** |

### Cost per million output tokens

Cost = hourly price ÷ (measured output tok/s × 3,600) × 1,000,000. This is output tokens only, at the measured workload, with the prompt processing included in the GPU time.
Treat these as a comparison between settings, not a quote.

| GPU | setting | workload, concurrency | output tok/s | $ per 1M output tokens |
|---|---|---|---|---|
| H200 | FP8 | agent ×64 | 2,449 | **$0.74** |
| H200 | AWQ 4-bit | agent ×64 | 1,970 | $0.93 |
| H200 | FP8 | chat ×64 | 1,753 | $1.04 |
| H200 | BF16 | agent ×64 | 1,527 | $1.19 |
| A100 | prefix cache off | chat ×32 | 425 | $1.30 |
| A100 | state cache 80 | agent ×32 | 228 | **$2.42** |
| A100 | AWQ 4-bit default | agent ×32 | 178 | $3.11 |
| A100 | MTP | agent ×1 | 74 | $7.50 |
| A100 | AWQ 4-bit default | agent ×1 | 61 | $9.08 |
| H200 | AWQ 4-bit | agent ×1 | 116 | $15.74 |

Reading it:
- **Under load, the H200 is the cheaper GPU per token.** It costs 3.3× more per hour but serves 10.7× more agent tokens (FP8 at 64 vs the best A100 setting at 32).
  That gives $0.74 vs $2.42 per million.
- **For one user or a small team, the A100 is the cheaper GPU.** A single stream doesn't fill either GPU, so the hourly price dominates: $1.99/h vs $6.56/h.
  The A100 gives a perfectly usable 16 ms per token (62 tokens/s, faster than anyone reads), or 6.4 ms with MTP.

## 6. Results and what they mean

Workloads (all answers 256 tokens, thinking off, synthetic prompts):
- **agent:** 16 different 4K-token system prompts (like an agent's instructions plus tool schemas) shared across requests, plus a 128-token question. Exercises the prefix cache.
- **chat:** 1K-token prompts, no sharing.
- **long:** 8K-token prompts, no sharing.

Terms: **TTFT** = time to first token (the wait). **ITL** = inter-token latency (1000 / ITL = one reader's tokens per second).
**Throughput** = output tokens/s summed over all requests. **Concurrency** = requests in flight. p50 = median; p99 = slowest 1 in 100.

### 6.1 One user (T1)
- Same AWQ weights: ITL **8.4 ms on the H200 vs 16.0 ms on the A100** (119 vs 62 tokens/s per reader). That is 1.9×, less than the 3.1× memory-bandwidth ratio, because part of each step (the DeltaNet recurrence, kernel launches, sampling) does not speed up with bandwidth.
  Decoding one user is limited by how fast the weights are read from memory.
- TTFT is short on both (66 ms vs 86 ms) because the 4K system prompt is already in the prefix cache.
- On the H200, smaller weights stream faster: AWQ 8.4 ms, FP8 10.2 ms, BF16 14.7 ms.
- MTP on the A100 cuts ITL to **6.4 ms** (2.5× faster streaming): the draft head's guesses were accepted for ~3 tokens per verification step.

### 6.2 Many users (T2; figures F2, F3)
- **H200:** throughput climbs to **1,970 tok/s (AWQ), 2,449 tok/s (FP8), 1,527 tok/s (BF16)** at 64 concurrent agent requests, with TTFT p50 around 1 s.
  At 16 concurrent, TTFT p50 stays at ~300 ms for all three.
- **A100 (AWQ default):** throughput flattens at **~175 tok/s** from 16 concurrent requests on, and TTFT p50 jumps to **15.8 s at 16 and 38 s at 32**.
  ITL for the requests that are running stays ~19.5 ms. The wait is queueing, not slow compute; section 6.4 explains the cause.

### 6.3 Weight precision on the H200 (T2, T5)
- **Decoding (few tokens per pass) is memory-bound:** fewer bytes per weight wins. AWQ is fastest for one user.
  The kernel benchmark agrees: Marlin AWQ is 1.78× faster than BF16 at 1 token on the H200 (2.79× on the A100, where bandwidth is scarcer).
- **Prefill and big batches are compute-bound:** 4-bit weights must be dequantized, and Marlin falls to **0.5× of BF16 at 1,024+ tokens** on the H200 (0.76× on the A100).
  FP8 runs on FP8 tensor cores and reaches **1.76× of BF16 at 4,096 tokens** (DeepGEMM).
- That is why the served results cross over: for 8K prompts at 16 concurrent, **FP8 gives 352 tok/s and 4.9 s TTFT, BF16 242 tok/s and 7.4 s, AWQ 161 tok/s and 13.2 s**.
  For chat at 64 concurrent: FP8 1,753, BF16 1,202, AWQ 934 tok/s.
- **Recommendation on Hopper: FP8** is the best all-rounder (best throughput and long-prompt TTFT, near-AWQ single-user speed, and the official checkpoint). AWQ only when single-stream latency is all that matters.
- Caveat for the writer: the FP8 kernel microbenchmark at 1–64 tokens (0.63–0.70× BF16) is slower than the served FP8 ITL implies (10.2 ms vs BF16 14.7 ms).
  The server runs these kernels inside CUDA graphs with fused activation quantization; treat the served numbers as the truth for decoding.

### 6.4 The A100's memory budget (T3, F5): the most interesting finding
- Qwen3.5-27B's DeltaNet layers need a fixed state per request. With the prefix cache on, SGLang keeps **5 state slots per running request**
  (the live state plus snapshots at cached prefix boundaries, so a later request can resume from a cached prefix).
- On 40 GB, after 18.8 GB of AWQ weights, SGLang's default split gave **49 state slots, so a cap of 9 running requests**, alongside a 134K-token KV cache that sat mostly unused.
  SGLang says so in its startup log: "max_running_requests is capped to 9 by the mamba state cache".
- Three ways to spend the same memory differently, measured:

| A100 setting | running cap | KV cache tokens | 1-user TTFT | 1-user ITL | agent ×32 tok/s | agent ×32 TTFT p50 | chat ×32 tok/s |
|---|---|---|---|---|---|---|---|
| default (prefix cache on) | 9 | 134,392 | 86 ms | 16.0 ms | 178 | 38.2 s | 284 |
| state cache 80 slots (`--max-mamba-cache-size 80`) | 16 | 61,573 | 1.3 s | 16.0 ms | 228 | 21.9 s | 358 |
| MTP speculative decoding | 3 | 90,869 | 1.4 s | 6.4 ms | 128 | 57.3 s | 236 |
| prefix cache off (`--disable-radix-cache`) | 32 | 174,325 | 1.3 s | 16.0 ms | 158 | 24.5 s | 425 |

- Reading it:
  - **More state slots** double the cap and raise throughput 28% (agent) to 26% (chat) and halve the 16-user wait (15.8 s → 7.7 s).
    But the KV cache shrank to 61K tokens, below the ~65K the 16 shared 4K prompts need, so the prefix cache evicts prompts just before they are reused
    and even a single user now waits 1.3 s.
  - **MTP** makes one stream 2.5× faster but its draft head's memory leaves room for only 3 running requests: the best single-user setting and the worst under load.
  - **Prefix cache off** removes the 5-slot cost and runs 31 requests at once, which is best for chat (425 tok/s), but every agent prompt is recomputed (1.3 s TTFT).
- The lesson: on a 40 GB card, a hybrid linear-attention model turns serving into a three-way memory budget (KV cache vs recurrent state vs speculative-decoding head).
  Pick the setting for the job: the default or MTP for one interactive user, the larger state cache for an agent fleet that shares prompts, prefix cache off for unrelated chats.
  SGLang's own hint, `--mamba-ssm-dtype bfloat16` (halves the state), is the next thing to try; we did not measure it.
- The H200 has the same rule but so much memory that it barely binds: the cap was 64 (AWQ), 61 (FP8) and 46 (BF16).

### 6.5 Stress tests (T4)
- **No request failed** in any test on either GPU.
- **H200, AWQ:** a 1,008-request run at 64 concurrent held 2,044 tok/s with TTFT p50 1.6 s, p99 2.8 s. At 4 requests/s arriving at random, TTFT p50 was 98 ms.
  A burst of 304 simultaneous requests drained at 1,958 tok/s, the last first-token arriving at ~34 s (p99).
- **A100, AWQ default:** the same tests sit behind the 9-request cap: ~180 tok/s throughout, p99.9 TTFT 45.8 s at 32 concurrent, and a 150-request burst's p99 first token at 216 s.
  Even 1 request/s arriving at random queues (TTFT p50 11.3 s): with 9 slots the A100 completes at most ~0.7 agent requests/s, so arrivals outpace it.
- Cold start (empty prefix cache, 16 concurrent): H200 TTFT p50 2.65 s vs 300 ms warm; A100 18.7 s vs 15.8 s warm.

### 6.6 FlashInfer under the hood (T6, T7)
- **FP8 KV cache does not pay off here:** halving the KV bytes made decode attention **about 1.35× slower on the H200, and 2.2–2.7× (e5m2) to 10–12× (e4m3) slower on the A100**,
  which has no FP8 hardware and converts in software. With only 16 attention layers and 4 KV heads, KV reads are a small share of decode time, so the conversion cost dominates.
  Keep the BF16 KV cache for this model.
- Scattered vs contiguous pages and page size 1 vs 16 changed decode time by under ~10% (mostly under 2%): paging costs almost nothing here.
- **JIT:** most attention variants compile once in 6–18 s (the prefill kernels took 52–69 s on the H200), then are cached on disk. A production image should ship a warm JIT cache.

## 7. Recommendations

- **Cost:** for steady multi-user or agent traffic, the H200 at $6.56/h is about 3× cheaper per token than the A100 at $1.99/h. For a single developer or a demo, the A100 costs less than a third as much per hour.
- **One H200, agents at scale:** FP8 weights, defaults otherwise. 2,449 tok/s at 64 concurrent agent requests with ~1 s TTFT; one reader streams at ~98 tokens/s (10.2 ms ITL).
- **One H200, lowest single-stream latency:** AWQ 4-bit (8.4 ms ITL, 119 tokens/s per reader).
- **One A100-40GB, a few interactive users or a long thinking session:** AWQ with MTP (6.4 ms ITL), at most ~3 at once.
- **One A100-40GB, an agent fleet sharing prompts (e.g. 8–16 parallel Terminal-Bench trials):** AWQ with `--max-mamba-cache-size 80`. Expect queueing beyond 16 requests.
- **Either GPU:** BF16 KV cache, FlashInfer attention, prefix cache on for agents, warm JIT cache in the image.

## 8. Method and caveats (for a short "How we measured" box)

- Each phase was capped at 30 minutes per GPU (a few A100 follow-ups at 30–40 minutes); one run per point, so differences under ~5% are noise.
- Synthetic prompts (random tokens in the shared-prefix structure above); real agent text will differ, especially MTP acceptance.
- Thinking was off for throughput tests and the tool-call check, to keep output lengths fixed. Harbor runs use thinking on.
- Context length 32K; the published Terminal-Bench numbers used 256K context.
- Results come from 4 runs; their raw logs, per-request data and GPU telemetry are archived.
- Not yet measured: a full fresh-host setup time end to end (see section 4); `--mamba-ssm-dtype bfloat16`; H100; multi-GPU.

## 9. Figures (from the dashboard)

Dashboard: https://claude.ai/artifact/6EvZ6nFaWp8v1ZAfCAHnEZ (private until shared from its Share menu).

- **F1** Headline tiles: per-user ITL on each GPU, peak throughput, server start time, tool-call accuracy.
- **F2** "Agent throughput by concurrency": tok/s vs concurrency, H200 solid, A100 dashed.
- **F3** "Wait for the first token by concurrency": TTFT p50 (log scale).
- **F4** "Served throughput by workload" (H200 AWQ/FP8/BF16) next to "Linear-layer kernels vs BF16".
- **F5** "Requests running and waiting" on the A100: running requests pinned at the cap while the queue grows, for each setting.
- **F6** Stress-test table.
- **F7** FlashInfer KV-format and JIT tables.

## 10. Data tables

### T1. One user (agent workload, concurrency 1)

| GPU | weights / setting | tok/s | TTFT p50 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|
| H200 | AWQ 4-bit | 116 | 66 ms | 8.4 ms | 9.3 ms |
| H200 | FP8 | 96 | 64 ms | 10.2 ms | 10.8 ms |
| H200 | BF16 | 67 | 68 ms | 14.7 ms | 15.5 ms |
| A100 | AWQ 4-bit | 61 | 86 ms | 16.0 ms | 20.4 ms |
| A100 | State cache 80 | 48 | 1.3 s | 16.0 ms | 21.0 ms |
| A100 | MTP | 74 | 1.4 s | 6.4 ms | 25.7 ms |
| A100 | Prefix cache off | 47 | 1.3 s | 16.0 ms | 19.1 ms |

### T2. Concurrency sweep (all settings)

| GPU | setting | workload | concurrency | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|---|---|---|
| H200 | AWQ 4-bit | agent | 1 | 116 | 66 ms | 106 ms | 8.4 ms | 9.3 ms |
| H200 | AWQ 4-bit | agent | 16 | 1,228 | 300 ms | 469 ms | 11.6 ms | 16.2 ms |
| H200 | AWQ 4-bit | agent | 64 | 1,970 | 1.2 s | 2.0 s | 22.7 ms | 27.4 ms |
| H200 | AWQ 4-bit | chat | 64 | 934 | 5.9 s | 16.4 s | 21.3 ms | 25.2 ms |
| H200 | AWQ 4-bit | long | 16 | 161 | 13.2 s | 22.8 s | 12.8 ms | 18.1 ms |
| H200 | FP8 | agent | 1 | 96 | 64 ms | 124 ms | 10.2 ms | 10.8 ms |
| H200 | FP8 | agent | 16 | 1,180 | 269 ms | 334 ms | 12.3 ms | 16.9 ms |
| H200 | FP8 | agent | 64 | 2,449 | 900 ms | 6.0 s | 17.5 ms | 23.4 ms |
| H200 | FP8 | chat | 64 | 1,753 | 2.1 s | 8.1 s | 16.1 ms | 21.8 ms |
| H200 | FP8 | long | 16 | 352 | 4.9 s | 8.4 s | 13.6 ms | 17.9 ms |
| H200 | BF16 | agent | 1 | 67 | 68 ms | 105 ms | 14.7 ms | 15.5 ms |
| H200 | BF16 | agent | 16 | 859 | 293 ms | 361 ms | 17.3 ms | 21.3 ms |
| H200 | BF16 | agent | 64 | 1,527 | 1.2 s | 8.7 s | 21.9 ms | 26.5 ms |
| H200 | BF16 | chat | 64 | 1,202 | 3.5 s | 11.5 s | 20.3 ms | 27.0 ms |
| H200 | BF16 | long | 16 | 242 | 7.4 s | 12.6 s | 18.6 ms | 23.0 ms |
| A100 | AWQ 4-bit | agent | 1 | 61 | 86 ms | 540 ms | 16.0 ms | 20.4 ms |
| A100 | AWQ 4-bit | agent | 16 | 173 | 15.8 s | 22.4 s | 19.7 ms | 24.3 ms |
| A100 | AWQ 4-bit | agent | 32 | 178 | 38.2 s | 44.2 s | 19.5 ms | 29.3 ms |
| A100 | AWQ 4-bit | chat | 32 | 284 | 23.0 s | 26.1 s | 18.5 ms | 27.2 ms |
| A100 | AWQ 4-bit | long | 8 | 82 | 13.2 s | 20.8 s | 21.4 ms | 30.5 ms |
| A100 | State cache 80 | agent | 1 | 48 | 1.3 s | 1.3 s | 16.0 ms | 21.0 ms |
| A100 | State cache 80 | agent | 16 | 188 | 7.7 s | 21.6 s | 22.8 ms | 30.3 ms |
| A100 | State cache 80 | agent | 32 | 228 | 21.9 s | 35.5 s | 22.6 ms | 34.4 ms |
| A100 | State cache 80 | chat | 32 | 358 | 13.4 s | 21.4 s | 21.1 ms | 31.2 ms |
| A100 | State cache 80 | long | 8 | 76 | 11.8 s | 26.4 s | 20.2 ms | 24.4 ms |
| A100 | MTP | agent | 1 | 74 | 1.4 s | 1.4 s | 6.4 ms | 25.7 ms |
| A100 | MTP | agent | 16 | 123 | 27.5 s | 29.9 s | 7.2 ms | 295.5 ms |
| A100 | MTP | agent | 32 | 128 | 57.3 s | 62.9 s | 7.2 ms | 296.3 ms |
| A100 | MTP | chat | 32 | 236 | 31.1 s | 33.8 s | 7.0 ms | 95.7 ms |
| A100 | MTP | long | 8 | 72 | 20.5 s | 26.1 s | 7.3 ms | 402.0 ms |
| A100 | Prefix cache off | agent | 1 | 47 | 1.3 s | 1.3 s | 16.0 ms | 19.1 ms |
| A100 | Prefix cache off | agent | 16 | 153 | 15.2 s | 20.7 s | 23.9 ms | 31.5 ms |
| A100 | Prefix cache off | agent | 32 | 158 | 24.5 s | 50.8 s | 32.9 ms | 39.8 ms |
| A100 | Prefix cache off | chat | 32 | 425 | 6.1 s | 17.6 s | 27.9 ms | 37.3 ms |
| A100 | Prefix cache off | long | 8 | 78 | 14.3 s | 20.8 s | 21.4 ms | 28.5 ms |

### T3. Server facts per setting

| GPU | setting | weights GB | KV cache tokens | running cap | DeltaNet state slots | server ready s | tool calls right | args right |
|---|---|---|---|---|---|---|---|---|
| H200 | AWQ 4-bit | 18.83 | 894,521 | 64 | default | 65 | 100% | 100% |
| H200 | FP8 | 28.47 | 809,196 | 61 | 309 | 220 | 100% | 100% |
| H200 | BF16 | 51.05 | 615,355 | 46 | 234 | 60 | 100% | 100% |
| A100 | AWQ 4-bit | 18.83 | 134,392 | 9 | 49 | 75 | 100% | 100% |
| A100 | State cache 80 | 18.83 | 61,573 | 16 | 80 | 75 | 100% | 100% |
| A100 | MTP | 18.83 | 90,869 | 3 | 16 | 90 | 100% | 100% |
| A100 | Prefix cache off | 18.83 | 174,325 | 32 | n/a (prefix cache off) | 80 | 100% | 100% |

MTP mean accepted length: 2.98 tokens per verification step (synthetic random-token prompts).

### T4. Stress tests (AWQ baseline, agent workload)

| GPU | test | concurrency | req/s sent | requests | tok/s | TTFT p50 | TTFT p99 | TTFT p99.9 | ITL p99 | failed |
|---|---|---|---|---|---|---|---|---|---|---|
| H200 | burst |  |  | 304 | 1,958 | 18.0 s | 33.8 s | 33.8 s | 28.5 ms | 0 |
| H200 | cold start | 16 |  | 48 | 470 | 2.7 s | 9.1 s | 9.1 s | 16.0 ms | 0 |
| H200 | p99.9 run | 64 |  | 1008 | 2,044 | 1.6 s | 2.8 s | 8.5 s | 29.2 ms | 0 |
| H200 | arrival rate |  | 4 | 64 | 979 | 98 ms | 281 ms | 309 ms | 61.4 ms | 0 |
| H200 | arrival rate |  | 16 | 160 | 1,915 | 3.9 s | 7.8 s | 7.9 s | 172.4 ms | 0 |
| A100 | burst |  |  | 160 | 185 | 99.7 s | 216.2 s | 216.2 s | 25.7 ms | 0 |
| A100 | cold start | 16 |  | 32 | 151 | 18.7 s | 22.9 s | 22.9 s | 26.5 ms | 0 |
| A100 | p99.9 run | 32 |  | 608 | 181 | 38.8 s | 45.7 s | 45.8 s | 27.4 ms | 0 |
| A100 | arrival rate |  | 1 | 32 | 167 | 11.3 s | 19.4 s | 19.6 s | 1,205.4 ms | 0 |
| A100 | arrival rate |  | 4 | 64 | 174 | 37.2 s | 75.7 s | 75.8 s | 132.4 ms | 0 |

### T5. Linear-layer kernels: all weight matrices, one forward pass (ms, speedup vs BF16)

| tokens per pass | A100 BF16 | A100 AWQ | H200 BF16 | H200 AWQ | H200 FP8 |
|---|---|---|---|---|---|
| 1 | 37.8 | 13.6 (2.79×) | 12.9 | 7.2 (1.78×) | 20.3 (0.63×) |
| 16 | 38.4 | 14.1 (2.72×) | 13.1 | 7.6 (1.73×) | 20.1 (0.65×) |
| 64 | 44.9 | 21.3 (2.11×) | 13.6 | 12.7 (1.07×) | 19.4 (0.70×) |
| 256 | 63.0 | 64.7 (0.97×) | 18.0 | 35.6 (0.50×) | 20.5 (0.88×) |
| 1024 | 199.2 | 249.2 (0.80×) | 72.6 | 144.1 (0.50×) | 41.5 (1.75×) |
| 4096 | 759.0 | 993.8 (0.76×) | 283.2 | 577.1 (0.49×) | 163.8 (1.73×) |

Kernels: BF16 = cuBLAS (torch F.linear); AWQ = Marlin W4A16 (SGLang); FP8 = DeepGEMM W8A8 128x128 block (SGLang, the server's path on Hopper).

### T6. FlashInfer batch decode, 16 attention layers (µs)

| GPU | KV format | 16 req × 16K ctx | 64 × 4K | 256 × 1K | vs BF16 at 64 × 4K |
|---|---|---|---|---|---|
| A100 | BF16 | 856 | 1,011 | 872 | 1.00× |
| A100 | FP8 e5m2 | 2,062 | 2,738 | 1,946 | 2.71× |
| A100 | FP8 e4m3 | 10,413 | 10,419 | 10,563 | 10.30× |
| H200 | BF16 | 586 | 593 | 568 | 1.00× |
| H200 | FP8 e5m2 | 800 | 792 | 775 | 1.34× |
| H200 | FP8 e4m3 | 800 | 792 | 775 | 1.34× |

### T7. FlashInfer JIT: first-use compile time per attention variant (s; later calls < 0.1 s)

| variant | A100 | H200 |
|---|---|---|
| Batch decode, BF16, head_dim 256 (Qwen3.5-27B) | 10.4 | 8.1 |
| + logits soft-cap | 10.3 | 7.9 |
| + fused RoPE (ROPE_LLAMA) | 10.3 | 8.0 |
| + sliding window (256) | 10.2 | 8.0 |
| FP8 e5m2 KV cache | 11.8 | 8.4 |
| head_dim 64 | 10.3 | 7.9 |
| Decode on tensor cores (prefill kernel) | 17.8 | 16.6 |
| Batch prefill, causal, paged KV | 0.0 | 69.4 |
| Single-request decode | 8.8 | 6.5 |
| Single-request prefill, causal | 15.9 | 52.4 |
