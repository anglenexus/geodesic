# PLAN: GRPO Post-Training of Qwen3.6-27B with LoRA on one H200 (Miles + Harbor)

Companion docs: [../harbor/README.md](../harbor/README.md) (running Harbor), [../harbor/PLAN.md](../harbor/PLAN.md) (trajectory collection).

## Status (2026-10-23)

| Phase | Status |
|---|---|
| 0. Box setup + baseline (no training) | ⏳ Not started |
| 1. Smoke test: Miles + Harbor + LoRA, small model | ⏳ |
| 2. 27B memory profile + adapter-scope check | ⏳ |
| 3. Pilot GRPO run, 27B | ⏳ |
| 4. Evaluation on held-out tasks | ⏳ |

**Code (2026-10-23):** everything in §7 is written and unit-tested on the Mac (909 tests). Nothing has run on the box yet.

---

## 0. Decisions

| # | Decision | Value |
|---|---|---|
| D1 | Policy model | `Qwen/Qwen3.6-27B` (dense, 26.9B params: 48 linear-attention + 16 full-attention layers, vocab 248,320) |
| D2 | Hardware | **1× H200 (141 GB)**. Training and rollout colocated on the same GPU |
| D3 | Framework | **Miles** (SGLang rollout, Megatron training). Fallback: SkyRL (§9, R9) |
| D4 | Fine-tuning method | **LoRA**, Megatron backend (Miles' FSDP backend has no LoRA). Full fine-tuning doesn't fit (430 GB) |
| D5 | LoRA rank, alpha, learning rate | **r = 64, alpha = 64** (scale α/r = 1). **LR = 1e-5, constant.** If the pilot shows collapse or no movement, step through the 5-value ladder in §3a, one value at a time (no grid). Memory isn't what limits the rank (§1). Revisit rank after the pilot |
| D6 | Adapter scope | **Native LoRA** (`--megatron-to-hf-mode raw`) on `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`: attention in the 16 GQA layers plus the MLPs of all 64 layers, **318.7M parameters** at r = 64 (`configs/common.sh`). The linear-attention projections get no adapters ("native GDN layers carry no adapters"). Decided 2026-10-23; Phase 2 prints the list of modules that received adapters to confirm (§2) |
| D7 | Algorithm | **GRPO**, exactly as written in §3. Every extra (KL, warmup, LR decay, filtering) is **off** unless explicitly enabled |
| D8 | Agent / environment | **Assumed: `terminus-2`** through Harbor, the same agent used for data collection. Rewards come from the task verifier (0/1). Other ways to give the model tools exist; see §2b for the comparison |
| D9 | Token handling | **TITO** through Miles' session server: the trainer uses the token IDs SGLang sampled, never re-tokenized text |
| D10 | Context limit and summarization | **terminus-2 summarization off**, in training and evaluation. A trajectory that reaches the context limit ends **without running the verifier**, with reward 0 (corrected 2026-10-23, §2b). **Training (SFT and RL): 48k.** **Evaluation (validation and test, base and every adapter): 64k.** Decided 2026-10-22. In training, the 48k limit is Miles' `--max-seq-len` and terminus-2's `model_info.max_input_tokens`; SGLang's context is 48k + one turn's output (56k), as in Miles' example, so the turn that crosses 48k can finish (§2b). Why, and when to change it: §2b. Memory at 48k is tight: ~135 GB of 141 in the training phase (§1, corrected 2026-10-23); fallback 40k or 32k |
| D11 | Rollout precision | **bf16**, the same as training. FP8 rollout is off (avoids train/rollout logprob mismatch; Miles validates FP8+LoRA only for specific models) |
| D12 | Train / test split | **Train on the 20 selected TB 2.1 tasks** (`../harbor/state/tb21_selected_tasks.txt`). **Test on 20 other TB 2.1 tasks; validate on 8 more** (§2a; lists in `../harbor/state/tb21_test20_tasks.txt` and `tb21_val8_tasks.txt`). Chosen 2026-10-22 from 39 candidates after a GPT-5.6 Luna run on them. The three sets are disjoint. Fixed before training and never changed after looking at test results. This measures transfer from TB tasks to other TB tasks: no official TB 2.1 score can be reported for a trained adapter (R7) |
| D13 | Thinking mode | **Off** (`chat_template_kwargs.enable_thinking=false`) for the baseline and the first training runs. Decided 2026-10-22. Goal: get one run working end to end first. Turning it on later goes up a ladder of levels with metric gates (§3c) |
| D15 | SFT warm start | **Optional, off by default.** Used only if Phase 0 shows too few training tasks with signal (R4): LoRA SFT on **Qwen's** passing trajectories before GRPO (Qwen3.6-27B's Phase 0 passes; changed from Luna 2026-10-23), with the SFT hyperparameter set in §3d. Decided 2026-10-22 |
| D14 | Sandboxes | Local Docker on the GPU box (**16 CPUs, 200 GB RAM**). Start at **16 concurrent trials**; Phase 0 tests 24. The 20 tasks each reserve 1 CPU (overfull-hbox: 2), and the trainer, SGLang and Ray need ~4 cores, which leaves ~12. Containers are idle while the model generates, so some oversubscription is fine. But timeouts are wall-clock, so CPU contention turns into false failures, especially on CPU-heavy tasks (crack-7z-hash, password-recovery, write-compressor). Raise concurrency only if the timeout rate doesn't rise. Also cap by total task memory (§1), and stagger starts |

---

## 1. Memory budget (1× H200, colocated)

Calculated from the official `config.json` at 32k tokens per sequence, bf16, per-layer activation checkpointing and chunked logits. Phase 2 replaces these with measurements.

Colocated mode alternates two phases on the same GPU. The idle side is moved off the GPU between them (`--colocate`, `--lora-base-cpu-backup`).

| Rollout phase | GB | | Training phase | GB |
|---|---|---|---|---|
| SGLang base weights (bf16) | 54 | | Frozen base (bf16) | 54 |
| LoRA adapter (bf16) | ~1 | | Adapter weights + grads + fp32 master + Adam, r = 64 (all-linear upper bound) | 6.7 |
| CUDA graphs, runtime overhead | ~5 | | Checkpointed layer inputs (64 × 5120 × 2 B × 32k) | 21 |
| **KV cache + linear-attention state** | **~80** | | One recomputed layer + gradient transients | ~12 |
| | | | Logits for the whole pass (32k × 248,320 × 2 B, bf16; `--log-probs-backend fused` adds no fp32 copy) | 16 |
| **Total** | **~140** | | **Total** | **~110** |

- **Rollout capacity:** full-attention KV is 66 KB/token, and linear-attention state is 151 MB per sequence. A full 32k rollout takes about 2.2 GB, so about 36 at once. Most trajectories are shorter (Luna p50: 9 turns).
- **Logits (corrected 2026-10-23):** an earlier version of this table assumed logits computed 2,048 tokens at a time (2 GB). Miles' output layer computes them for every token in the pass at once; its chunk setting only splits the log-prob step after that (`miles/backends/training_utils/loss/hub/logit_processors.py`). So logits cost tokens × 248,320 × 2 B: 16 GB at 32k, 24 GB at 48k. `--log-probs-backend fused` (in `configs/common.sh`) streams them without the default backend's fp32 copies.
- **At 48k (D10, training limit):** checkpointed layer inputs 32 GB, recompute ~18 GB and logits 24 GB, so the training phase is **~135 GB of 141**. Tight; Phase 2 measures it. If it runs out of memory, lower `MAX_LEN` in `configs/common.sh` to 40k or 32k. A full-length rollout takes ~3.3 GB (~24 at once), but SGLang only allocates what each rollout actually uses. At 64k the training phase would be ~160 GB, which doesn't fit.
- **Rollout phase as configured (2026-10-23):** SGLang gets 0.85 of the GPU (`SGLANG_MEM` in `configs/grpo.sh`): 120 GB, of which ~55 GB is weights and adapter and **~65 GB is the KV pool**, not the ~80 GB in the table, which left no headroom for CUDA graphs and trainer leftovers. One trajectory at SGLang's 56k cap takes ~3.9 GB, so 16 concurrent trials all at the cap would need 63 GB. If the pool is short, SGLang queues requests (slower, not out of memory). Phase 2 measures the headroom and can raise the fraction.
- **Rank isn't what limits memory.** r = 256 costs 27 GB of adapter state, which still fits in the training phase (~116 GB). We start at r = 64 for learning reasons (RL gives little information per episode) and to leave headroom for longer contexts.
**Host RAM (the box has 200 GB):**

| Item | GB |
|---|---|
| OS, Docker daemon, Ray/Miles/Megatron processes, SGLang host side | ~20–25 |
| CPU mirror of the frozen base (`--lora-base-cpu-backup`) | 54 |
| Offloaded trainer state during rollout (adapter + Adam, r = 64) | ~7 |
| Pinned transfer buffers | ~5–10 |
| Headroom | ~20 |
| **Left for task containers** | **~85–95** → ~32 containers at 2–3 GB each |

TB 2.1 task memory limits are 2 GB (68 tasks), 4 GB (13) and 8 GB (8). Those are caps; Phase 0 measures actual use from the status page's Docker stats. **CPU (16 cores), not RAM, is the tighter limit** on concurrency (D14).

---

## 2. LoRA on Qwen3.6-27B: what Miles supports

From Miles' LoRA docs (2026-10):

| Path | Flag | Qwen3.6-27B status | Adapter scope |
|---|---|---|---|
| **Native** (raw mode) | `--megatron-to-hf-mode raw` | Supported as a "Qwen3.5/3.6/Next hybrid (GQA layers only)". Colocated, single adapter. Needs a torch_dist base checkpoint | **Full-attention (GQA) layers only.** "Native GDN layers carry no adapters." Unclear whether the MLPs of every layer are included |
| **Bridge** (Megatron-Bridge PEFT) | `--megatron-to-hf-mode bridge` | **Not listed** for 27B (only for 35B-A3B). Megatron-Bridge itself supports Qwen3.5/3.6 dense | All linear layers, excluding MTP and vision via wildcards |

What each scope means at r = 64:

| Scope | Trainable params | Notes |
|---|---|---|
| Attention in the 16 GQA layers only | ~42M | Smallest. Leaves 75% of layers and all MLPs untouched |
| + MLPs in all 64 layers | ~319M | Typical "attention + MLP" scope |
| + linear-attention projections (all-linear) | ~419M | Upper bound used in §1 |

**Chosen (2026-10-23, D6): native, attention + MLPs, 318.7M.** The native path is the one whose linear-attention layers take packed sequences (below), and it adds adapters to the MLPs of every layer. Phase 2 still prints the list of modules that actually received adapters.
- If native means attention-only in 16 layers, try Bridge with `--target-modules all-linear` before accepting the small scope.
- If neither path works, see the fallback in R1.

### Sequence packing in training (decided 2026-10-23)

**Choice: pack several chats into one forward pass, up to 48k tokens** (`--use-dynamic-batch-size --max-tokens-per-gpu 49152`, Megatron's default packed `thd` layout). The alternative was one chat per pass (`--qkv-format bshd --micro-batch-size 1`).

- **Why packing is allowed:** the one-chat-per-pass requirement comes from the *bridge* path: megatron-core's GatedDeltaNet rejects packed sequences (`scripts/run_qwen3_5_35b_a3b_lora.py:10, 23`). The *native* path uses Miles' own linear-attention layers, which take packed sequences (`miles_plugins/models/linear_attn.py` passes `cu_seqlens` through). Miles' native-LoRA example for Qwen3.5-9B trains with packing on (`examples/lora/run_lora_native.py`).
- **Why it's better:** peak memory is unchanged, since no pass holds more than 48k tokens, the same as one full-length chat (§1). Our chats are mostly 3–13k tokens, so short ones share a pass instead of each being padded and run alone. Fewer passes per step.
- **Caveats:**
  - Read from Miles' code and examples at commit 8e99947, not measured. No Miles example runs native LoRA with packing on a *dense* Qwen3.5/3.6 with MLP adapters; the 9B example has attention adapters only.
  - Packing doesn't change the loss in principle: sequences stay separate through `cu_seqlens` and the loss is a token mean (§3b). A bug in either would show up as a mismatch between the trainer's and SGLang's log-probs.
  - **Check in Phase 1** (Qwen3.5-4B smoke test): train with packing, and confirm (1) no error from the linear-attention layers, and (2) the trainer's log-probs match SGLang's (Miles logs the difference; it should stay ~1e-2, as in Miles' example). If either fails, switch to one chat per pass. That is only a flag change in `configs/common.sh`, at the cost of speed, not memory.

---

## 2a. Task split (D12)

### Training set: 20 tasks
The 20 tasks in `../harbor/state/tb21_selected_tasks.txt`, chosen for Luna's 10–90% band plus cost and length limits (see `../harbor/state/tb21_task_notes.md`).

Existing baseline: Qwen3.5-27B (AWQ, thinking on, 32k context) passes 22.4% overall (22/98), and gets nonzero reward on 11 of the 20 tasks. Phase 0 re-measures this with Qwen3.6-27B.

### Test set: 20 tasks (chosen 2026-10-22)
List: `../harbor/state/tb21_test20_tasks.txt`. Chosen from the 39 candidates below using 10 GPT-5.6 Luna (terminus-2) attempts per task, run on the Mac on 2026-10-22 (jobs `tb21_test31_b1`/`_b2` in the harbor ledger; the 8 ⚠ candidates weren't run). Criteria: (1) signal: drop tasks Luna can't do (break-filter-js-from-html 0%, train-fasttext 0%, query-optimize 10%); (2) time: drop long tasks or 60+ min timeouts (fix-ocaml-gc: 67 min median, 115 turns; build-pov-ray: 200 min; distribution-search, portfolio-optimization, llm-inference-batching-scheduler); (3) reliable grading: drop tasks with repeated harness errors (`tmux send-keys` failures, no reward): git-multibranch, reshard-c4-data, merge-diff-arc-agi-task; (4) none of the ⚠ tasks.

Luna's rate is only a proxy for Qwen: on the training 20, Qwen3.5-27B scored 0% on tasks Luna passes 85–91% and 80% on tasks Luna passes 15–42%. So Luna-easy tasks are kept. The Phase 0 baseline with Qwen3.6-27B is the real check: swap out tasks that are 0% or 100% for Qwen **before** the test set is frozen.

All 20 have agent timeouts ≤ 30 min. Luna: 86.9% mean, median trial ≤ 6.1 min, $0.007/trial.

| Task | Luna (10 attempts) | Median trial (min) | Timeout (min) |
|---|---|---|---|
| log-summary-date-ranges | 30% | 1.3 | 15 |
| code-from-image | 50% | 4.9 | 20 |
| fix-git | 60% | 1.4 | 15 |
| prove-plus-comm | 70% | 2.6 | 15 |
| constraints-scheduling | 78% | 1.9 | 20 |
| headless-terminal | 80% | 2.4 | 15 |
| build-pmars | 90% | 3.4 | 15 |
| feal-linear-cryptanalysis | 90% | 3.8 | 30 |
| regex-log | 90% | 2.6 | 15 |
| cobol-modernization | 100% | 2.1 | 15 |
| git-leak-recovery | 100% | 2.1 | 15 |
| hf-model-inference | 100% | 2.8 | 15 |
| modernize-scientific-stack | 100% | 1.4 | 10 |
| multi-source-data-merger | 100% | 1.6 | 15 |
| nginx-request-logging | 100% | 1.8 | 15 |
| openssl-selfsigned-cert | 100% | 1.3 | 15 |
| pypi-server | 100% | 1.4 | 15 |
| pytorch-model-recovery | 100% | 6.1 | 15 |
| sqlite-db-truncate | 100% | 1.6 | 15 |
| vulnerable-secret | 100% | 1.6 | 15 |

### Validation set: 8 tasks (chosen 2026-10-22)
List: `../harbor/state/tb21_val8_tasks.txt`. For checkpoint selection and early stopping during training; the test set is only run for the final report. Six come from the 16 non-selected tasks Luna was already run on (baselines exist); two are test candidates that were dropped only for their 60-min timeouts. Disjoint from training and test.

| Task | Luna (attempts) | Median trial (min) | Timeout (min) |
|---|---|---|---|
| kv-store-grpc | 100% (85) | 1.2 | 15 |
| fix-code-vulnerability | 93% (85) | 1.0 | 15 |
| custom-memory-heap-crash | 100% (85) | 3.1 | 30 |
| pytorch-model-cli | 93% (85) | 4.4 | 15 |
| path-tracing | 20% (10) | 10.2 | 30 |
| schemelike-metacircular-eval | 56% (10) | 13.2 | 40 |
| distribution-search | 90% (10) | 1.9 | 60 |
| portfolio-optimization | 100% (10) | 6.3 | 60 |

Kept out on purpose: **feal-differential-cryptanalysis** (near-duplicate of the test task feal-linear-cryptanalysis) and **path-tracing-reverse** (sibling of path-tracing).

### Test-set candidates: 39 tasks (the pool the 20 above were chosen from)
These are TB 2.1 tasks none of our runs had touched. They're grouped by public **TB 2.0** pass rate for terminus-2 submissions (`../harbor/state/tb21_task_ranking.csv`, column `terminus2_rate`, mostly frontier models), and listed with their TB 2.1 agent timeout. ⚠ marks tasks that failed the oracle on the Mac (`../harbor/state/tb21_task_notes.md`); each needs an oracle check on the H200 box (native x86) in Phase 0.

**Easy (terminus-2 > 90%): 7 tasks.** Cheap regression checks.

| Task | t2 rate | Timeout (min) |
|---|---|---|
| constraints-scheduling | 98% | 20 |
| portfolio-optimization | 98% | 60 |
| git-leak-recovery | 95% | 15 |
| prove-plus-comm | 95% | 15 |
| modernize-scientific-stack | 92% | 10 |
| multi-source-data-merger | 92% | 15 |
| pypi-server | 92% | 15 |

**In band (terminus-2 10–90%): 32 tasks.**

| Task | t2 rate | Timeout (min) | Note |
|---|---|---|---|
| distribution-search | 90% | 60 | |
| fix-git | 90% | 15 | changed in 2.1 (tests, Dockerfile) |
| vulnerable-secret | 90% | 15 | |
| pytorch-model-recovery | 89% | 15 | changed in 2.1 (Dockerfile) |
| headless-terminal | 88% | 15 | |
| git-multibranch | 85% | 15 | |
| nginx-request-logging | 85% | 15 | |
| openssl-selfsigned-cert | 85% | 15 | |
| build-pmars | 82% | 15 | changed in 2.1 (instructions) |
| cobol-modernization | 82% | 15 | |
| merge-diff-arc-agi-task | 82% | 15 | 4 GB |
| log-summary-date-ranges | 80% | 15 | |
| hf-model-inference | 79% | 15 | changed in 2.1 (tests, Dockerfile) |
| regex-log | 78% | 15 | |
| build-pov-ray | 77% | **200** | skip at intermediate checkpoints |
| mcmc-sampling-stan ⚠ | 72% | 30 | 4 CPU / 8 GB. Oracle fails from environment drift: expect to drop |
| code-from-image | 69% | 20 | |
| qemu-startup ⚠ | 62% | 15 | Rosetta-only failure: oracle on x86 |
| reshard-c4-data | 62% | 60 | |
| llm-inference-batching-scheduler | 60% | 30 | |
| sqlite-db-truncate | 54% | 15 | oracle ✓ (Mac) |
| fix-ocaml-gc | 51% | 60 | oracle ✓ (Mac) |
| bn-fit-modify ⚠ | 42% | 60 | Verifier timed out under Rosetta: oracle on x86 |
| break-filter-js-from-html | 40% | 20 | |
| build-cython-ext ⚠ | 40% | 15 | Oracle fails from environment drift: expect to drop |
| query-optimize | 37% | 15 | changed in 2.1 (instructions, verifier timeout 15→30 min) |
| tune-mjcf ⚠ | 35% | 15 | Rosetta-only failure: oracle on x86 |
| compile-compcert ⚠ | 34% | 40 | 2 CPU / 4 GB. Oracle timed out under Rosetta: oracle on x86 |
| qemu-alpine-ssh ⚠ | 30% | 15 | Rosetta-only failure: oracle on x86 |
| protein-assembly ⚠ | 22% | 30 | Non-deterministic oracle: expect to drop |
| feal-linear-cryptanalysis | 20% | 30 | |
| train-fasttext | 10% | 60 | changed in 2.1 (test script, tests). Rate from only 6 submissions |

**Rules for the test and validation sets:**
- A test or validation task that fails the oracle on the H200 box is **dropped before any baseline is run**, and the drop is recorded here (replace it from the candidates above only before the Phase 0 baseline).
- The test set never changes after the Phase 0 baseline.
- **Checkpoints during training:** validation set. **Final checkpoint (and base model):** test set. All 20 test tasks have timeouts ≤ 30 min, so no subset is needed.
- **Not used:** the 14 tasks below 10% for terminus-2 (likely 0% for Qwen before and after training), the 8 ⚠ candidates and the 9 other dropped candidates above (distribution-search and portfolio-optimization moved to validation), and the 10 non-selected Luna tasks not in validation (they stay the R4 pool).

---

## 2b. Agent harness and summarization (D8, D10)

### Assumption: post-training uses terminus-2
The plan assumes the policy acts through **terminus-2**, Harbor's reference agent:
- Each turn the model writes one JSON object in plain text (`analysis`, `plan`, `commands` = keystrokes + wait duration, `task_complete`).
- The keystrokes are sent to a tmux session in the task container, and the screen output comes back as the next user message.
- One model call per turn; no sub-agents, except during summarization.

Why terminus-2:
- **It matches the data we have.** All ledger trajectories (Luna, Qwen) use this format, so Qwen's can serve as SFT data (D15) and Luna's as a comparison.
- **The pipeline already runs it:** ledger, browser and baselines.
- **One plain-text call per turn keeps TITO simple.** No tool-call parser sits between the sampled tokens and the action.

**Other ways to give the model tools** (not planned; listed so the choice is explicit):

| Option | How actions are written | For | Against |
|---|---|---|---|
| **terminus-2** (chosen) | JSON in plain text; keystrokes into tmux | Same format as the Luna data; one call per turn; Harbor's reference agent | Not Qwen's trained tool format. Parser warnings when the JSON is malformed. Keystroke timing (`duration`) is a weak point |
| **Native tool calling** through the chat template (tools such as `bash`, `edit_file`, returned as `<tool_call>` and answered with `<tool_response>`), e.g. Miles' SWE-agent example or a small custom loop | Qwen's own trained format | Likely a higher Qwen baseline: Qwen3.6 was trained on this format | Luna data would need converting, so SFT data no longer matches exactly. SGLang's tool-call parser sits in the loop, and a call re-serialized into history can differ from the sampled tokens (TITO must be checked) |
| **Bash in markdown code blocks** (e.g. Harbor's `mini-swe-agent`) | Plain text, one shell command per turn | Very simple; one call per turn | Different format from the Luna data; no interactive programs (no tmux) |
| **Installed CLI agents** (Claude Code, Codex CLI, Gemini CLI, Aider, OpenHands, ... in `harbor/agents/installed/`) | Each agent's own tool format, often several model calls per turn and sub-agents | Strong products | Built for specific providers' APIs. Token IDs per call are hard to capture, which makes TITO impractical |

Changing the harness later means a new baseline, and a new SFT data format if D15 is used.

### Summarization: what terminus-2 does
From the terminus-2 source (`harbor/agents/terminus_2/terminus_2.py`, harbor 0.23):
- The context limit comes from `model_info.max_input_tokens`.
- **Proactive:** before each turn, if `limit − current prompt tokens < proactive_summarization_threshold` (**default 8000**), terminus-2 runs a 3-call handoff (summary, questions, answers) and **replaces** the chat history with it.
- **Fallback:** if a request still exceeds the server's context, it trims history and summarizes when `enable_summarize=true` (default). When false, it raises `ContextLengthExceededError` and the agent stops. `enable_summarize=false` also turns off the proactive summarization.
- Harbor itself warns that "rollout details will be incomplete if context summarization occurs".

Why it matters for training:
- **It breaks TITO.** After a summary, the prompt no longer extends the earlier token sequence (R6).
- **It adds 3 model calls that aren't part of the task's action sequence.** They would either go untrained or need their own samples.
- **Our past runs summarized early.** `../harbor/state/run_qwen27b.sh` set `max_input_tokens` to 16k, so summarization started at about an 8k-token prompt.

### What to set (suggestion)
1. **Summarization off:** `enable_summarize=false`, which also disables proactive summarization. In GRPO it's set by `HARBOR_TERMINUS_2_ENABLE_SUMMARIZE=false` in `scripts/agent_server.sh`.
2. **The context limit (as built, 2026-10-23):** 48k = Miles' `--max-seq-len` (the session server cuts each training sample there) = terminus-2's `model_info.max_input_tokens` (`AGENT_MAX_INPUT_TOKENS`). SGLang's `--context-length` is 48k + 8,192 (one turn's output cap) = 56k, as in Miles' Harbor example. terminus-2 ignores the `max_seq_len` Miles sends; in practice SGLang ends the trial, by rejecting a request whose prompt plus 8,192 output tokens exceeds 56k.
3. **Hitting the limit ends the trajectory without the verifier** (corrected 2026-10-23). With summarization off, terminus-2 re-raises `ContextLengthExceededError` (`terminus_2.py:1088`). The trial stops before verification, and the agent server returns `exit_status: "AgentError"` with reward 0. Plain `harbor run` takes the same path, so the ledger and evaluation behave the same. An **agent timeout** is different: the verifier still runs and its reward counts (`trial/single_step.py:83`). Both are policy outcomes, not infrastructure exclusions (but see §8 item 6). Log the rates (§6).
4. **Size the limit from data.** Peak prompt tokens per trial, from the ledger's trajectories:

| Model (run) | Trials | p50 | p90 | > 32k |
|---|---|---|---|---|
| GPT-5.6 Luna (no cap) | 2,402 | 12.0k | 35.6k | 13% |
| Qwen3-4B (capped by its own context) | 118 | 6.4k | 21.0k | 0% |
| Qwen3.5-27B, thinking, summarized at ~8k | 92 | 9.1k | 11.1k | 0% |

On the 20 training tasks, Luna exceeded 32k on 16% of trials. It's concentrated in four tasks: adaptive-rejection-sampler 71%, crack-7z-hash 52%, mailman 50%, write-compressor 44%. Many of those trials *passed*: of Luna's passing trials, 14% went over 32k. At 32k, those tasks would partly fail on length alone. Going higher:

| Peak prompt over | All Luna trials (training tasks) | Passing trials |
|---|---|---|
| 32k | 15.6% | 13.7% |
| 48k | 3.1% | 2.6% |
| 64k | 1.0% | 0.6% |

Above 48k, what remains is mostly crack-7z-hash (27% over 48k, 11% over 64k) and mailman (11%, 4%).

**Decision (D10): train at 48k, evaluate at 64k.**
- **Why not 64k in training:** the training phase would be ~160 GB of 141 (an estimate; corrected 2026-10-23 for the logits, §1), so it doesn't fit. Peak memory is set by the single longest trajectory, so running out of memory would hit mid-run. The longest trajectories also set the step time. And 64k gains only ~2% of trials over 48k; for SFT, about 14 of 688 passing Luna trajectories.
- **Why 64k at evaluation is fair:** base and adapters get the same limit, and serving alone at 64k fits easily. Training at 48k only makes very long failures cost more, which doesn't hurt when evaluating with more room. Validation runs use 64k too, so checkpoints are chosen on the same setting as the final test.
- **Summarization stays off at evaluation:** the adapter never saw summarization handoffs in training. When comparing with other TB 2.1 numbers (e.g. runs with terminus-2's default summarization), state the setting.
- **Switch training to 64k only if both:** Phase 0 shows Qwen3.6 exceeding 48k on more than 5% of training-task trials, **and** Phase 2 measures a 64k training step with ≥ 10 GB headroom. Past Qwen runs were shorter than Luna's, so 32k may also be enough: if fewer than 5% of Qwen3.6 trials exceed 32k in Phase 0, training can drop to 32k (cheaper).
5. `max_turns`: leave unlimited. The agent timeout already bounds each trajectory.

### Harbor agent server for GRPO (as built, 2026-10-23)
Miles' Harbor example for local Docker (`examples/swe-agent-harbor-docker`) talks to a separate Harbor agent server (`miles_agent_server.py` on the `harbor-miles-v0.20.0` branch, pinned to commit ac63e51 in `configs/pins.sh`). `scripts/agent_server.sh` starts it on the host, and Miles POSTs each trial to `http://127.0.0.1:11000/run`. What we found in its code, and what we set:
- **terminus-2 parser forced to XML.** The server hardcodes `parser_name: "xml"`, while the SFT data, the ledger baselines and evaluation all use terminus-2's default JSON format. `grpo/harbor_parser.patch` (one line, applied by `setup.sh`) makes it read `HARBOR_TERMINUS_2_PARSER`, and `agent_server.sh` sets `json`. The server also sets `interleaved_thinking: True`, which is empty with thinking off; the Phase 1 TITO check would catch a difference.
- **Timeouts.** The server scales every task timeout by 2 by default. `HARBOR_TIMEOUT_MULTIPLIER=1.0` keeps the `task.toml` values, as in the ledger and Phase 4 evaluation. There's no `--agent-timeout`, so each task's own timeout applies (training tasks: agent 750–1800 s, verifier 360–1800 s). Miles' client-side ceiling `AGENT_TRIAL_TIMEOUT` counts time a trial spends **queued** behind the 16 running ones (Miles sends all 160 at once), so `scripts/run.sh` sets it to 12 h, as a backstop only.
- **Image cleanup off.** Every 30 min the server runs `docker image prune -a`, which would delete the pre-pulled task images between waves. `DOCKER_PRUNE_INTERVAL_SEC=1e9` turns it off. Containers are still removed after each trial (`compose down`).
- **Ports.** Miles' session server uses 30000–30015 (16 workers), and its router uses 31000. The agent server uses 11000. terminus-2 runs inside the agent server process on the host, so it reaches the session server at 127.0.0.1.
- **What comes back.** Every trial returns a reward (0 if the verifier didn't run) and an `exit_status`: `Submitted`, `TimeLimitExceeded`, `SequenceLengthLimitExceeded`, `AgentError` (including the 48k limit), or `Error: …`. Miles marks a sample aborted only if no model call was recorded or collecting its tokens failed (`agentic_tool_call.py:123,134`). So `check_no_aborted` drops a group only in that case. Every other failure trains as reward 0, except the infrastructure failures that `grpo/infra_filter.py` excludes per attempt (§8 item 6).

---

## 3. Algorithm (GRPO, as implemented)

For each training task, sample **G** trajectories from the current policy. Rewards r_i ∈ {0, 1} come from the verifier.

```
advantage   A_i = (r_i − mean(r_1..r_G)) / (std(r_1..r_G) + 1e-6)
ratio       ρ_t = π_θ(token_t | context) / π_old(token_t | context)
loss        L   = − (1 / N_tokens) · Σ_i Σ_{t ∈ assistant tokens of i} min(ρ_t A_i, clip(ρ_t, 1−ε, 1+ε) A_i)
```

All values (learning rate, batch, clip, optimizer, loss averaging) are in **§3d, RL set**.

**Off by default.** Each one needs an explicit decision to enable, and gets logged when enabled:
- KL penalty to the reference model (coefficient 0)
- LR warmup or decay
- Entropy bonus
- Auxiliary cross-entropy (SFT) loss mixed into the RL loss (§3e)
- Gradient-norm clipping (Miles/Megatron default 1.0, so it has to be turned off explicitly; §3d)
- Filtering out groups where every trajectory has the same reward (they produce zero gradient anyway; filtering only saves compute)
- Length penalties
- Importance-sampling correction for rollout/train mismatch. With TITO and bf16 on both sides, this mismatch should be near zero, which we **measure** (§6)

**Before Phase 3:** dump Miles' full resolved config and check that nothing on this list is on by default.

### 3a. Learning rate and LoRA alpha (D5)

**Alpha and LR act as one setting.** The adapter adds (α/r)·B·A to each weight. Adam's step size doesn't depend on the gradient's scale, so the change in effective weights per step is roughly (α/r) × LR. We fix **α = r** (scale 1), quote every LR at scale 1, and tune only the LR. To compare with a recipe that uses another α/r, multiply its LR by its α/r.

**What others used** (web and repo search, 2026-10-22). Nobody has published a LoRA RL learning rate for Qwen3.6-27B. The closest points:

| Source | Model | Setup | LR | α/r | Effective (α/r)·LR |
|---|---|---|---|---|---|
| Miles LoRA recipes (`examples/lora/`: Qwen3-4B, Qwen2.5-0.5B/3B, gpt-oss-20B, Kimi-K2.5; `scripts/run_glm5_1_744b_a40b_lora.py`) | various | GRPO, LoRA | 1e-5, constant | 1 | 1e-5 |
| Miles `examples/lora/run-qwen3-4b-megatron-lora-result.sh` | Qwen3-4B | GRPO, LoRA r = 64, α = 32 | 2e-5 | 0.5 | 1e-5 |
| Miles `scripts/run_qwen3_5_35b_a3b_lora.py` | Qwen3.5/3.6-35B-A3B | GRPO, LoRA r = 16, α = 32 | 1e-5, constant | 2 | 2e-5 |
| [AllenAI TMax-27B](https://huggingface.co/allenai/tmax-27b) | **Qwen3.6-27B** | RL on terminal tasks, **full fine-tuning** (as far as the card says), 8 prompts × 32 samples | 1e-6, constant | – | LoRA usually needs ~10× the full-FT LR (the Kimi recipe's comment says the same) → ~1e-5 |
| LoRA **SFT** on Qwen3.5/3.6-27B ([TUDUM](https://arxiv.org/pdf/2607.01927), [community adapters](https://huggingface.co/hotdogs/qwen3.6-27b-cybersecurity-lora)) | 27B | SFT, not RL | 1e-4 to 2e-4 | 1–2 | Not comparable: SFT tolerates much larger steps than RL |

Everything RL-related points to an effective **1e-5**, so that's the starting value.

**Fallback ladder (α = r):** `2e-6, 5e-6, `**`1e-5`**`, 2e-5, 5e-5`. This isn't a grid, and there's no separate sweep phase. The Phase 3 pilot *is* the first run:
1. Start the pilot at 1e-5. Check at step 10 (about half a day).
2. **Too high** (any of: policy entropy dropping fast, parser-warning rate rising, output length blowing up, training reward falling below the Phase 0 baseline): restart from the base model one rung down (5e-6, then 2e-6).
3. **Too low** (training reward flat **and** the per-step policy change, mean |logprob after update − logprob before| on the batch's tokens, staying near zero): restart one rung up (2e-5, then 5e-5).
4. Otherwise keep going. Expected cost: one run. Worst case: two short restarts in one direction.

The per-step policy change is the earlier signal. Reward from 160 trajectories per step is noisy, but how far the policy moves shows right away whether the step size is in range. No LR sweep on Qwen3.5-4B in Phase 1: the value may not transfer across model sizes. Phase 1 only checks that one update changes the logprobs. If the rank changes later, keep α = r and re-check the LR with the same step-10 test.

### 3b. Loss averaging

Each assistant token t of trajectory i has a loss ℓ_{i,t} (the clipped term in §3). How they combine into one number decides how much each trajectory, and each token, pushes the update.

| Option | Formula | Weight per token | Miles |
|---|---|---|---|
| **Token mean** (chosen) | Σ_i Σ_t ℓ_{i,t} / Σ_i T_i | Every token weighs the same, so a trajectory weighs in proportion to its length | `--calculate-per-token-loss`. All Miles LoRA recipes set it |
| Sample mean | (1/B) Σ_i (Σ_t ℓ_{i,t} / T_i) | 1/(B·T_i): every trajectory weighs the same, so tokens in long trajectories get less weight | **Default** when the flag is absent (`get_sum_of_sample_mean`) |
| Per-trajectory mean over split samples | like sample mean, but the denominator is the whole trajectory's token count even when the trajectory is split into several samples | Same as sample mean, per trajectory | `denominators=rollout_mask_sums`. Only relevant if one trajectory becomes several samples (thinking with history stripped, §3c) |
| Fixed normalizer | Σ_i Σ_t ℓ_{i,t} / C, with C a constant (e.g. batch size × max length) | Constant; no dependence on lengths | Not a flag; would need a small patch |

**Why token mean first:**
- **Long failures get their full penalty.** Many of our failures are long: timeouts and wandering for many turns. Under sample mean, each token of a 30k-token failure counts about 1/30 as much as a token of a 1k-token one. That pushes less against wandering and tends to make trajectories grow longer. Token mean doesn't depend on how tokens are split among trajectories.
- **One sample per trajectory.** With thinking off and TITO, a trajectory's assistant tokens across all turns form one sample, so token mean weighs every generated token equally.
- **It matches the recipes** Miles' LoRA support was validated on.

**Its downside:** long trajectories dominate the gradient. If they're mostly failures, the step mostly pushes probabilities down. Log the share of assistant tokens coming from reward-1 vs reward-0 trajectories each step (§6). If the pilot stalls with nearly all tokens from failures, sample mean is the first alternative to try (an explicit change, logged).

**Check in Phase 1:** with the flag on, the denominator is the total loss-token count over the whole step's batch (all microbatches), not per microbatch.

### 3c. Thinking mode: turning it on gradually (D13)

Start with thinking off. Measured evidence for why (ledger, Qwen3.5-27B AWQ, thinking on, 32k, over an SSH tunnel): ~2,048 output tokens per turn (Luna 512, Qwen3-4B without thinking 433); 63% of trials hit the agent timeout; p50/p90 output 22k/37k tokens per trajectory against a 32k cap. Part of the timeout rate came from AWQ and the tunnel, so the H200 numbers will differ.

**Dials** that make thinking cheaper, roughly ordered by effect:

| Dial | Range | Effect | Cost / caveat |
|---|---|---|---|
| **Per-turn thinking budget** | 0, 512, 1024, 2048, unlimited tokens | Caps thinking per turn; when the cap is hit, SGLang forces `</think>` and the model writes its JSON reply. This is the main dial | SGLang's custom logit processor (`--enable-custom-logit-processor`, `custom_params.thinking_budget`). Its built-in `Qwen3ThinkingBudgetLogitProcessor` hardcodes Qwen3 token IDs; Qwen3.6's vocab (248,320) needs a subclass with its own IDs. There's an open report of the budget not being enforced on Qwen3.6-27B ([sglang#25536](https://github.com/sgl-project/sglang/issues/25536)), so verify first. The forced tokens weren't sampled by the policy: mask them out of the loss |
| **Keep earlier thinking in history** | `preserve_thinking` true / false (Qwen3.6 chat-template option, default false) | **true:** each turn's prompt extends the previous one, so a trajectory stays **one training sequence** (fixes R5), and Qwen says the model was trained to use past thinking. **false:** thinking is stripped, so each turn becomes its own training sample | true: thinking fills the context (budget × turns). false: ~3–5× training compute per step. Prefer **true + a budget** |
| **Context limit** | 32k → 48k → 64k | Room for thinking without hitting the limit | At 64k: ~4.4 GB per rollout → ~18 concurrent (vs ~36); training phase ~160 GB of 141 (checkpointed inputs 43 GB, logits 33 GB): **doesn't fit on one H200** without a memory change (e.g. computing logits in chunks, which Miles doesn't do today) |
| **Summarization threshold** (terminus-2 `max_input_tokens`) | ≥ the context limit | Summarization never fires, so TITO holds (R6); trajectories that hit the limit end with reward 0 | Same as D10 |
| **Agent timeout multiplier** (Harbor) | 1×, 1.5×, 2×, 3× | Fewer false failures from slow decoding | The slowest trial sets the step time, so steps get longer. Use the same multiplier for base and adapters at test time, and report it |
| **Concurrent trials** | 16 → 8 | Faster decoding per trial, so fewer timeouts | Fewer trials in flight; longer steps |
| ~~Reply `max_tokens`~~ | – | Caps the whole reply | Not recommended: truncation cuts the JSON and causes parse errors. Use the thinking budget instead |

**Ladder.** Levels are measured on the **base model without training** (training set, 20 tasks × 4 attempts, a few hours per level), so choosing a level costs no training runs:

| Level | Thinking | Budget per turn | preserve_thinking | Context | Timeout multiplier |
|---|---|---|---|---|---|
| L0 (start) | off | – | – | 32k | 1× |
| L1 | on | 512 | true | 32k | 1× |
| L2 | on | 1024 | true | 48k | 1.5× |
| L3 | on | 2048 | true | 64k | 2× |
| L4 | on | unlimited | true | 64k | 3× |

**Metrics per level, with gates** (compared with L0 on the same tasks):

| Metric | Gate to accept a level |
|---|---|
| Timeout rate | ≤ L0 + 5 points |
| Context-limit hit rate | ≤ 5% |
| Summarizations | 0 |
| Parser-warning rate | ≤ L0 + 5 points |
| Estimated step time (tokens per trajectory × trajectories, from Phase 2 throughput) | ≤ 2× L0 |
| **Pass rate** | the reason to move up |

**Rule:** go up one level at a time, stopping at the first level that fails a gate. Choose the cheapest accepted level whose pass rate is **≥ 10 points above L0**, paired by task (with 80 attempts per level, smaller gaps are within noise). If no level qualifies, stay at L0.

**When:** after the thinking-off pilot (Phase 3) works. A change of level starts a new training run from the base model. An adapter trained without thinking hasn't seen thinking-mode prompts, so continuing it would be a different experiment, and an explicit choice.

### 3d. Hyperparameter sets: RL and SFT

Two separate sets. They are never mixed: an SFT run and a GRPO run each get their own config file (`configs/`). Every value is set explicitly on the command line, so a framework default never decides one silently. "Source" says where each value comes from; "Later sweep" lists the values to try only if the first run says so (§3a procedure), one at a time.

**RL (GRPO) set**

| Setting | Value | Miles flag | Source / reason | Later sweep |
|---|---|---|---|---|
| Learning rate | **1e-5**, constant | `--lr 1e-5 --lr-decay-style constant` | §3a: Miles LoRA recipes, TMax-27B ×10 for LoRA. Miles' default (1e-6) is a full-FT value | 2e-6, 5e-6, 2e-5, 5e-5 (§3a ladder) |
| LoRA r, α, dropout | 64, 64, 0 | `--lora-rank 64 --lora-alpha 64 --lora-dropout 0` | D5. α = r keeps the scale at 1 | r ∈ {32, 128} after the pilot, keeping α = r |
| Tasks per step | **all 20** | `--rollout-batch-size 20` | §3f | 10 |
| Group size G | **8** | `--n-samples-per-prompt 8` | Miles recipes; ~22% Qwen baseline needs several attempts per task to see any success | 16 |
| Trajectories per optimizer step | 160 (= 20 × 8), one update per rollout batch | `--global-batch-size 160` | Fully on-policy: ρ = 1 at the update, so clipping is inactive | – |
| Advantage | (r − group mean) / (group std + 1e-6) | `--advantage-estimator grpo` | §3 | – |
| Clip ε | 0.2 both sides | `--eps-clip 0.2`, no `--eps-clip-high` | Inactive with one update per batch; kept so the loss is defined | – |
| Loss averaging | token mean | `--calculate-per-token-loss` | §3b | sample mean, if long failures dominate |
| KL to reference, entropy bonus, auxiliary SFT loss | 0, 0, none | `--kl-loss-coef 0 --kl-coef 0 --entropy-coef 0`, `--loss-type policy_loss` | Off by default (§3, §3e) | only as an explicit decision |
| Optimizer | Adam, β = (0.9, 0.98), ε = 1e-8, weight decay 0 | `--optimizer adam --adam-beta1 0.9 --adam-beta2 0.98 --weight-decay 0` | β from Miles' LoRA recipes. Weight decay 0: Megatron's default is 0.01, the recipes use 0.1 | – |
| Gradient clipping | **off** | `--clip-grad 0` | Miles/Megatron default to 1.0, which counts as a training extra, so it's off unless enabled. Grad norm is logged. Turn on (1.0) only if norm spikes show up | 1.0 |
| Sampling | temperature 1.0, top-p 1.0, no top-k | `--rollout-temperature 1.0 --rollout-top-p 1.0` | Sampling from the policy as-is keeps the logprobs the trainer computes valid | – |
| Context limit | 48k in training (evaluation 64k), summarization off | Miles `--max-seq-len 49152`; SGLang `--sglang-context-length 57344` (48k + one turn); terminus-2 `AGENT_MAX_INPUT_TOKENS` | D10, §2b. `--rollout-max-context-len` only caps SGLang; `--max-seq-len` is what the agentic path truncates at and forwards to Harbor | 32k or 64k, by the rule in §2b |
| Output per model call | 8,192 tokens | `--rollout-max-response-len 8192`, `AGENT_MAX_OUTPUT_TOKENS` | Harbor's default. Thinking off averages ~430–510 tokens per turn, so it only stops runaway output | – |
| Steps, saves | 50 steps (Phase 3), adapter + optimizer state every 10 | `--num-rollout 50 --save-interval 10` | Phase 3 | – |
| Thinking | off | `--apply-chat-template-kwargs '{"enable_thinking": false}'`, which also reaches the session server; `--tito-model qwen36` | D13 | §3c ladder |

**SFT set (optional warm start, D15)**

Data (changed 2026-10-23): **Qwen's passing** trajectories (reward 1) on the 20 training tasks, in terminus-2's JSON-in-text format, exported from the ledger by `sft/export_trajectories.py`. The real run uses Qwen3.6-27B's Phase 0 passes. The smoke test uses the Qwen3.5-27B passes already in the ledger: 22 trials on 11 tasks, 62 chats (`data/sft_qwen35_27b.jsonl`). Thinking (`reasoning_content`) is dropped, which matches thinking-off. A summarized trial (`linear_history`) becomes one chat per linear piece, with copied hand-off context and parse-error replies kept as context but not trained (`step_loss_mask: 0`). Chats over the context limit are dropped, never truncated.

**Rendering (`sft/rollout.py`, 2026-10-23):** Miles' built-in SFT rollout doesn't reproduce the token stream GRPO trains on. A custom rollout renders each chat with Qwen's chat template (`enable_thinking=False, preserve_thinking=True`, as TITO `qwen36` does). Each assistant turn is laid out as opener (not trained), reply + `<|im_end|>` (trained), then `\n` (not trained). The unit tests check this against the real Qwen3.6 tokenizer.

| Setting | Value | Miles flag | Source / reason | Later sweep |
|---|---|---|---|---|
| Learning rate | **1e-4**, constant | `--lr 1e-4 --lr-decay-style constant` | Published LoRA SFT on Qwen3.5/3.6-27B and Qwen3.5-9B terminal adapters: 1e-4 to 2e-4 at α/r = 1–2, i.e. effective ~2e-4, **peak** of a decaying schedule. A constant LR at about the average of such a schedule is ~1e-4 | 5e-5, 2e-4 |
| LoRA r, α, dropout | 64, 64, 0 | same as RL | Same adapter as RL, so GRPO continues from it directly | – |
| Epochs | **2** | `--num-epoch 2` | Published runs use 2–5. The data is narrow (20 tasks, many similar trajectories), so stay low to limit memorizing | 1, 3 |
| Batch | 16 trajectories per optimizer step, packed into passes of up to 48k tokens (§2) | `--rollout-batch-size 16 --global-batch-size 16` | Published 27B LoRA SFT runs use an effective 16 | 32 |
| Loss | cross-entropy on the trained assistant tokens only, token mean | `--loss-type sft_loss --calculate-per-token-loss` | Same averaging argument as §3b | – |
| Optimizer | Adam, β = (0.9, 0.999), ε = 1e-8, weight decay 0 | `--adam-beta2 0.999 --weight-decay 0` | Megatron's default β; weight decay off as for RL | – |
| Gradient clipping | off | `--clip-grad 0` | Same rule as RL | 1.0 |
| Warmup / decay | off | – | No-default-extras rule | – |
| Max length | 48k, same as RL | – | SFT and RL see the same context | – |

**Choosing among SFT runs:** pass rate on the **validation set** (§2a) through Harbor (decided 2026-10-23: no held-out loss). Never the test set. Cost (at Luna's scale, the upper bound): 688 trajectories × ~15k tokens ≈ 10M tokens per epoch, under ~1.5 h per epoch on the H200. Qwen's passes are fewer. That's cheap next to RL, so one sweep rung costs little.

### 3e. Regularization terms next to the RL loss

None are on. All three sit next to the GRPO loss, and each would add a weight to choose:

| Term | Added to the loss | What it does | Miles flag, default | Plan |
|---|---|---|---|---|
| **Entropy bonus** | − c · mean over tokens of H(π(·\|context)) | Pushes the policy to stay random; slows collapse | `--entropy-coef`, default **0.0** (recipes 0.0) | 0 |
| **KL to the reference model** | β · KL(π_θ ‖ π_ref), per token | Keeps the policy close to the base model. With LoRA, π_ref is the base with the adapter off | `--use-kl-loss` off, `--kl-loss-coef` **0.0**; `--kl-coef` (reward shaping) **0.0** | 0 |
| **Auxiliary cross-entropy (SFT) loss** | λ · (− mean log π_θ(demonstration tokens)) on fixed demonstrations, e.g. Luna passes | Keeps imitating good examples while doing RL | **No flag**: `--loss-type` picks *one* loss (`policy_loss` or `sft_loss`); mixing them needs `custom_loss` | none |

How the RL loss relates to cross-entropy: with one update per batch (ρ = 1), the GRPO gradient is −Σ_i A_i Σ_t ∇log π_θ(token_t). That's cross-entropy on the model's own samples, weighted by advantage: it pushes up passing trajectories (A > 0) and down failing ones (A < 0). So an auxiliary cross-entropy term adds *external* demonstrations to the same objective. If it's ever used, λ is a new hyperparameter with no default. The simpler order is SFT first (D15), then GRPO.

### 3f. Tasks per step: the recipe with only 20 tasks

In Miles, a step is `--rollout-batch-size` prompts (tasks) × `--n-samples-per-prompt` (G) trajectories. `--global-batch-size` is how many trajectories go into each optimizer update. Usual recipes have thousands of prompts and sample a batch of them per step (Miles' recipes use 32–256 prompts × 8; TMax-27B used 8 tasks × 32). Each step sees a random subset, and an "epoch" is many steps.

With 20 tasks, there's nothing to subsample. **Use all 20 every step**: 20 × 8 = 160 trajectories, one optimizer update.
- **For it:** every step trains on every task. Which tasks happen to land in a step adds no noise. Steps compare cleanly with each other: same tasks, so reward changes reflect the policy.
- **Cost:** ~160 trials at 16 concurrent is ~10 waves, ~50 min per step (§5), vs ~40 min for 16 tasks.
- **Tasks with no signal** (0/8 or 8/8) still run and contribute zero gradient. Dropping them is filtering, which is off by default (§3). They're also where change shows up first if the policy improves.
- **Later option: 10 tasks per step**, alternating the two halves. Twice as many updates per hour with half the trajectories each, so noisier. Try it only if the step time with 20 is too slow to make progress.

---

## 4. Phases

### Phase 0: Box setup and baseline (no training)
1. Run `setup.sh` on the H200 box: Docker, Harbor (pinned `harbor-miles-v0.20.0` commit + parser patch), the pinned Miles image and source (`configs/pins.sh`), model weights, task images, torch_dist checkpoints.
2. Serve Qwen3.6-27B with SGLang (bf16, one GPU, 32k context). Run Harbor `terminus-2` against it, reusing `aees_harbor` (scheduler, ledger, browser).
3. **Oracle check** on the 20 test + 8 validation tasks (§2a), natively on x86 (none are ⚠ tasks; all were solved by Luna on the Mac). Drop failures and record them in §2a.
4. **Baseline, training set:** 20 tasks × 8 attempts, thinking off (D13), summarization off (§2b). Also record each trial's peak prompt tokens, which checks the 48k training limit against the rule in §2b (D10). Record pass rate, turns, tokens, parser warnings, and timeouts per task. Tasks where every attempt fails, or every attempt passes, give no gradient (every advantage is 0). Keep them in the training set but report how many of the 20 have signal.
5. **Baseline, test + validation sets:** 20 test + 8 validation tasks × 4 attempts (112 trials), thinking off.
6. **Exit criterion:** the split is frozen, and the baseline numbers are recorded for both sets.

### Phase 1: Smoke test with a small model
Use **Qwen3.5-4B** (same `qwen3_5` architecture, about 7× faster) to check the plumbing before spending 27B time:
1. Get Miles' Harbor example (`examples/swe-agent-harbor-docker`, local Docker mode) running on 2–3 tasks with G = 4, LoRA r = 64, `--colocate`.
2. Check each of these:
   - **TITO:** for a few trajectories, compare the session server's token IDs with the training sample's token IDs, exactly. Use `verify_session_tito_tokenizer.py`.
   - **Loss mask:** decode the masked-in tokens and confirm they're exactly the assistant replies.
   - **Rewards:** Harbor's verifier result lands in `Sample.reward`. Infrastructure failures (the fixed list in §8 item 6) are excluded per attempt; timeouts and the 48k limit count. Check that an injected failure (e.g. a missing task folder) is masked and its group's other attempts still train.
   - **Adapter sync:** after one update, SGLang serves the new adapter (`--check-lora-weight-equal`).
   - **Logprob agreement:** the trainer's recomputed logprobs match SGLang's for the same tokens, within bf16 noise.
3. Run 10 steps and check that reward on the training tasks moves.

### Phase 2: 27B memory profile and adapter scope
1. Convert Qwen3.6-27B to torch_dist. Print which modules get adapters under native and under Bridge (§2), and pick the scope.
2. Measure peak GPU memory in each phase at 8k, 16k and 32k tokens. Compare with §1.
3. Measure one step's timing breakdown (rollout, train, adapter sync, offload/reload).
4. **Exit criterion:** a full 32k-token training step runs without running out of memory, with measured headroom.

### Phase 3: Pilot GRPO run on 27B
1. The RL set in §3d: all 20 training tasks per step, G = 8, r = 64, LR 1e-5.
2. Run until the training reward plateaus or 50 steps, whichever comes first. Save the adapter every 10 steps.
3. **Exit criterion:** training reward rises above the Phase 0 baseline, with no collapse (output length blowing up, parser-warning rate rising, entropy going to zero).

### Phase 4: Evaluation
1. Serve the base model plus each saved adapter (SGLang `--lora-paths`). Run the validation set (§2a) × 4 attempts at intermediate checkpoints, and the test set (§2a) × 4 attempts for the base model and the final (chosen) checkpoint, through the same Harbor pipeline. Ingest the runs into the ledger.
2. Report pass@1 with confidence intervals, per task and in aggregate, for base vs adapter checkpoints, plus turns and tokens per task. Compare on the same tasks (paired) so differences between tasks cancel. Also report **training-set** pass rates, kept separate from the test numbers: they show whether GRPO fit the training tasks, not whether the model improved in general.
3. Write a short report (numbers, issues, decisions needed) before deciding on longer runs, a different rank, or more tasks.

---

## 5. Throughput estimate (Phase 2 replaces this)

| Part | Estimate | Basis |
|---|---|---|
| Rollout | ~50 min per step, often longer | 160 trials (20 tasks × 8) at 16 concurrent (10 waves) × ~5 min mean trial (measured with Luna). ~35 min at 24 concurrent. **The slowest trial sets the step time** when the loop is synchronous, so tasks with long timeouts stretch it |
| Training | ~12 min per step | 160 × ~12k tokens ≈ 1.9M tokens × 6 × 27B FLOPs ÷ ~400 TFLOPS effective |
| Offload, reload, adapter sync | ~1–3 min per step | Moving 54 GB of base weights between CPU and GPU in each direction |
| **Total** | **~65–100 min per step → ~15–22 steps per day** | |

Most of the time is Docker trials, not GPU work. Ways to improve this, once the synchronous loop works:
- Run the sandboxes on an EC2 CPU box (32+ at once) and keep the H200 alternating: no code change.
- Fewer timeouts (less thinking): timed-out trials hold 25–80% of trial-minutes in the collected runs.
- Overlap rollout and training on the one H200: §5b measures whether that is feasible before any code is written.

---

## 5b. Overlap prototype: serving and training on one H200 at the same time (2026-10-24)

**Question.** Can SGLang serve the agents' rollouts while the LoRA trainer runs on the same GPU, so a step
takes max(rollout, training) instead of rollout + training? Miles can't do this today: native LoRA needs
`--colocate`, and Miles' async trainer refuses `--colocate`. So the prototype measures the GPU itself
first, with no Miles changes.

**Calculated beforehand** (the prototype replaces these):
- **Decode is HBM-bound.** Every decode step re-reads all weights: 54 GB in bf16 means ≥11 ms per step,
  ~70 tokens/s per stream, and only 2–12% of the tensor cores at 6–32 sequences.
- **Training is compute-bound.** 48k-token micro-batches run at ~50–60% tensor use and ~30–50% of HBM.
- **Their needs complement each other, but together they want more bandwidth than the GPU has.** Expect
  decode ~1.3–1.8× slower and training ~1.2–1.5× slower while they overlap.
- **Memory capacity is the real limit.** bf16 serving (~89 GB) plus training as configured (~135 GB) is
  224 GB. What fits (~129 GB): FP8 weights and FP8 KV cache for serving (~48 GB), plus checkpointed layer
  inputs offloaded to host RAM (−32 GB) and logits computed in chunks (−22 GB) for training (~81 GB).
- **How the GPU is shared:**
  - Without MPS, the two processes time-slice: no real overlap, and decode waits behind training kernels.
  - With MPS, they share one context and their kernels run at the same time.
  - CUDA streams don't help across processes.
  - MIG can't split 141 GB into 135 + 48.
- **Staleness.** Overlap means each batch was sampled by the policy one update old (lag 1). The
  correction would be a per-token weight min(π_trainer / π_sampler, C), which also covers the FP8/bf16
  difference. It is a training extra (D7), so it is off unless decided. Lag 2 adds nothing while
  rollout > training.

**What runs** (`proto/bench.sh`, in the pinned Miles image):
- **Serving:** the real Qwen3.6-27B in SGLang (bf16, and the official FP8 checkpoint), with a random r = 64
  adapter.
- **Agent load:** `proto/agent_load.py` replays turn sizes and environment waits from 907 real trials
  (`proto/load_profile.json`: Luna, Qwen3-4B and Qwen3.5-27B, token counts and durations only).
- **Training load:** `proto/train_proxy.py` has the model's exact layer shapes, flash attention and Miles'
  per-micro-batch pattern (old log-probs, forward, recompute, backward, Adam), and a ballast tensor for the
  rest of the 54 GB base. It does not run the delta-rule kernel (~2% of FLOPs).
- **Real trials:** an EC2 m7i.8xlarge runs 8 real terminus-2 trials the whole time, through an SSH tunnel
  and a logging pass-through, so the share of trial time spent waiting on the model is measured, not
  assumed.

| Phase | Server | Load | Measures |
|---|---|---|---|
| pcie | — | 4 GB copies | host↔GPU GB/s (offload budget) |
| train_alone, train_alone_offload | — | trainer, 48k micro-batches | tokens/s, TFLOPS, peak memory, cost of offloading |
| serve_bf16 | bf16, 0.85 | 16 agents | production serving baseline |
| serve_fp8_32, serve_fp8 | FP8, 0.36 | 32, then 16 agents | the serving config that fits beside the trainer |
| co_timeslice | FP8, no MPS | 16 agents + trainer (offload) | both slowdowns, time-sliced |
| serve_fp8_mps | FP8, MPS | 16 agents | MPS baseline |
| co_mps, co_mps70 | FP8, MPS | 16 agents + trainer at 100% / 70% of SMs | both slowdowns under MPS; adapter hot-load ×5 |

GPU counters (DCGM SM, tensor and DRAM activity, when profiling is allowed; nvidia-smi otherwise) are
sampled every second.

**Pass criteria** (`proto/report.py`), per co-run phase:
- no out-of-memory at 48k, with ≥ 8 GB headroom
- decode ≤ 1.5× slower (median per token), and p99 stream stall ≤ 10 s
- training ≤ 1.5× slower
- the projected step time, with sandboxes on EC2, is ≥ 15% shorter than alternating

**Cost:**
- H200: ~25 min of setup, then ~60 min of phases. About $9 at $6.56/h.
- EC2: ~$3.
- If it passes, building the overlap is still 1–2 weeks of Miles changes (a remote SGLang, adapter pushes,
  chunked logits, activation offload) plus the staleness decision.

**How to run** (laptop, from `aees/post_training`):
1. `bash proto/ec2_up.sh`: launches the EC2 box in us-east-2 with the laptop's AWS CLI profile, SSH from
   this IP only, and a self-terminate after 4 h. Installs Harbor and pulls the 8 task images (~15–20 min).
2. Create the Nebius H200 (Ubuntu 22.04 image with an NVIDIA driver ≥ 580), then
   `bash proto/kickoff.sh <h200-ip> [user]`:
   - copies this folder to the H200
   - lets EC2 forward to the H200's port 30000 and nothing else
   - starts `setup.sh` (steps 1–4, 6, 7; bf16 + FP8 weights) and `proto/bench.sh` in tmux `proto` on the H200
   - starts `proto/ec2_trials.sh` in tmux `proto` on EC2
3. Watch: `ssh <h200> tail -f proto_<RUN>.log`. Done when `/data/proto/<RUN>/bench.log` says "done".
4. `bash proto/fetch.sh` (results to `data/proto/<RUN>/`, then `report.md` and `dashboard.html`), then
   `bash proto/ec2_down.sh`, then delete the H200. `fetch.sh` can also be run mid-run for a partial dashboard.

`dashboard.html` (one file, opens offline) shows, per phase grouped as inference only / training only /
together: projected minutes per GRPO step, alternating vs overlapped, as bars per rollout scenario; decode
ms/token and training s/micro-batch alone vs together; peak GPU memory; the checks; one phase over time
(ms/token per call, micro-batches, SM/tensor/HBM activity, memory, stalls, requests in flight, tok/s); and
the real EC2 trials' latency over the whole run.

To re-run phases: `PHASES="co_mps co_mps70" bash proto/bench.sh` on the H200. Knobs: `SERVE_MEM_CO`,
`CO_MIN`, `MPS_CAP`, `AGENTS`.

---

## 6. Logging (per step)

- **Reward:** mean, per-task pass rate, and the fraction of groups where every trajectory got the same reward (these give no gradient). Share of assistant tokens from reward-1 vs reward-0 trajectories (§3b).
- **Lengths:** turns, assistant tokens, total tokens; the share of trajectories that hit the context limit (`ContextLengthExceededError`).
- **Behavior:** `terminus-2` parser-warning rate, `task_complete` rate, timeout rate, infrastructure-exclusion rate.
- **Optimization:** loss, gradient norm, adapter weight norm, mean |trainer logprob − SGLang logprob| on sampled tokens (should be ~0 under TITO), policy entropy on assistant tokens, per-step policy change (mean |logprob after update − logprob before|, §3a).
- **Time:** rollout, train, sync and offload durations; GPU memory peak in each phase.
- Every training trajectory also goes into the `aees_harbor` ledger, tagged with run ID and step, so it can be browsed in the existing trajectory browser.

**Training tracker (built 2026-10-24): `monitor/server.py` + `monitor/index.html`.**
- **How it works.** It runs on the box with the host's python3 (3.10 or later), uses the standard library only, and binds to 127.0.0.1:8770. Every 30 s it re-reads what the run already writes, with no wandb:
  - Miles' event log `$SAVE_DIR/events/*.jsonl` (all `rollout/*`, `train/*`, `perf/*` metrics per step);
  - each trial's `result.json`, assigned to a step by finish time;
  - checkpoint folders;
  - eval trials under `$DATA/trials/eval/<run>/iter_N/{val,test}/` (the layout the Phase 4 eval script writes);
  - `nvidia-smi`, `/proc/meminfo`, `df`, `docker ps`.
- **What it shows:**
  - status (running / stalled / finished), step k/N (= epochs), elapsed time, step time, ETA;
  - success rates: train per step, val/test per checkpoint, with Phase 0 baselines from an optional `monitor/baseline.json`;
  - how attempts ended (passed / failed / timeout / 48k / error / excluded infra);
  - loss, gradient norm, entropy, trainer vs SGLang log-prob difference, KL, clip fraction;
  - tasks with a learning signal;
  - output tokens per turn, turns, parse-error rate;
  - time per step (rollout / training / slowest trial), trials per hour;
  - host memory;
  - a picker for any logged metric;
  - a per-task heatmap, a checkpoint table (best by validation) and recent trials.
- **Access from the laptop:** `scripts/tunnel.sh user@box`, then open http://localhost:8770. "Download snapshot" saves `status.json`, which the page can open offline. `scripts/pull_run.sh` copies a run's inputs (a few MB) so the tracker also runs locally.
- **Phase 1:** check the Miles metric names against the candidate lists at the top of the page's script. The page also lists every key it finds.

**How to run the tracker**

1. **On the box**, start it next to the run, in its own terminal or tmux window. It only reads files, so it can be started before, during or after training, and restarted at any time; host history is kept in `$DATA/monitor/<run>/host.jsonl`.
   ```bash
   cd aees/post_training
   RUN=grpo_pilot1 python3 monitor/server.py        # RUN = the last part of SAVE_DIR (= agent_server.sh's RUN)
   # options: --data /data (or DATA=...), --num-rollout 50 (or NUM_ROLLOUT=...), --port 8770
   ```
   It prints `tracking grpo_pilot1: http://127.0.0.1:8770` and listens on the box's loopback only.
2. **On the laptop**, open the tunnel and keep it running. Extra ssh options go after the host:
   ```bash
   bash scripts/tunnel.sh ubuntu@<box> [-i ~/.ssh/key.pem]   # PORT=8770 by default
   ```
   Then open **http://localhost:8770**. The page refreshes every 30 s. The header turns red ("stalled?") if nothing new has been logged for 2× the median step time (at least 20 min), and the data age turns red if the tracker itself stops refreshing.
3. **Offline:**
   - **Snapshot:** click "Download snapshot" on the page (it saves `<run>_status.json`). Later, open `monitor/index.html` straight from disk and load the file with "Open snapshot".
   - **Full copy:** `bash scripts/pull_run.sh ubuntu@<box> grpo_pilot1` copies the run's tracker inputs into `runs/`, laid out like the box's `/data`. Then on the laptop:
     ```bash
     python3 monitor/server.py --run grpo_pilot1 --data runs    # Python 3.10+; open http://localhost:8770
     ```
     Add `WITH_TRAJECTORIES=1` to also copy transcripts (MBs per trial).
4. **Phase 0 baselines (optional):** after Phase 0, write `monitor/baseline.json`:
   ```json
   {"train": 0.22, "val": 0.30, "test": 0.25, "tasks": {"mailman": 0.375, "crack-7z-hash": 0.0}}
   ```
   - `train`, `val` and `test` are overall pass rates, drawn as dashed lines and as the first row of the checkpoint table.
   - `tasks` holds per-task pass rates on the training tasks, shown as the heatmap's "base" column (passes out of 8).
5. **Evals:** the Phase 4 eval script writes checkpoint N's trials to `$DATA/trials/eval/<run>/iter_<N>/{val,test}/<trial>/`. N is Miles' 0-based rollout id, the same as the checkpoint's `iter_<N>` folder. The tracker picks them up on its next refresh.
6. **Try it without a run:** `tests/test_monitor.py` has a fake-run generator (`make_run`). A 20-task × 8-attempt, 25-step fake run was checked in headless Chrome on 2026-10-24.

---

## 7. Code layout (as built, 2026-10-23)

```
post_training/
├── PLAN.md
├── setup.sh                     # box setup: Docker, pinned Harbor (+ patch) and Miles, weights, task images, torch_dist
├── configs/
│   ├── pins.sh                  # Miles image + commit, Harbor commit (one place)
│   ├── common.sh                # model, checkpoint, LoRA, packing/memory, colocate flags (SFT and GRPO)
│   ├── sft.sh                   # SFT set (§3d)
│   └── grpo.sh                  # RL set (§3d), SGLang, Harbor wiring, optional SFT warm start (INIT_ADAPTER)
├── scripts/
│   ├── run.sh                   # run.sh sft|grpo: docker run the Miles image, start Ray, submit the job
│   ├── agent_server.sh          # Harbor agent server on the host (GRPO)
│   ├── tunnel.sh                # laptop: SSH tunnel to the tracker
│   ├── pull_run.sh              # laptop: copy a run's tracker inputs for offline viewing
│   └── backup.sh                # upload a run's adapters to HF or S3
├── sft/
│   ├── export_trajectories.py   # ledger -> chat jsonl (passing trials)
│   └── rollout.py               # Miles SFT rollout: renders chats as GRPO's token stream
├── grpo/
│   ├── split_tasks.py           # task lists -> grpo_train.jsonl / grpo_val.jsonl
│   ├── harbor_parser.patch      # agent server: terminus-2 parser from an env var; exception type in each reply
│   └── infra_filter.py          # excludes infra failures per attempt from GRPO groups (§8 item 6)
├── monitor/
│   ├── server.py                # training tracker on the box: reads the run's files, serves the page (§6)
│   └── index.html               # the page: charts and tables, live or from a saved snapshot
├── proto/                       # overlap prototype (§5b): serve + train on one H200, with EC2 sandboxes
│   ├── bench.sh                 # H200: all phases in the Miles container; MPS daemon on the host
│   ├── train_proxy.py           # training load with the model's shapes; shapes.py: layer shapes, FLOPs
│   ├── agent_load.py            # agents replaying load_profile.json (make_profile.py builds it); adapter hot-load
│   ├── adapter.py               # random r = 64 adapter named after the checkpoint's modules
│   ├── report.py                # slowdowns, projected step times, pass/fail -> report.md, dashboard.html
│   ├── dashboard.html           # results page; report.py embeds report.json into a copy per run
│   ├── ec2_up.sh, ec2_prepare.sh, ec2_trials.sh, ec2_down.sh, llm_log_proxy.py   # EC2 box + real trials
│   └── kickoff.sh, fetch.sh     # laptop: start everything; pull results and write the report
├── tests/                       # unit tests (the loss-mask check lives here)
└── data/                        # generated jsonl (git-ignored; copy to the box or regenerate)
```

Evaluation (Phase 4) reuses `../harbor` (scheduler, ledger, browser); no code here yet.

Miles is used as an installed dependency. Our code is configs, thin wrappers, and checks, with no fork until something forces one. As in `../harbor/PLAN.md` §8: one module at a time, reviewed before the next.

---

## 8. Open questions for Miles (to answer in Phases 1–2)

1. Exactly which `qwen3_5` modules get adapters under native LoRA: attention in the GQA layers only, or MLPs too?
2. Does Bridge LoRA run for the 27B dense model even though only 35B-A3B is listed?
3. ~~Does the Harbor connector drive `terminus-2`?~~ Answered from the code (2026-10-23): yes, through the agent server, with fixed agent kwargs plus env-var knobs. The parser needed a patch (§2b, agent server).
4. How does the session server handle a chat template that rewrites history (Qwen strips earlier `<think>` blocks)? Does that break the token prefix and produce one sample per tree leaf (v2)?
5. ~~Default loss averaging~~ (answered from the code: per-sample mean unless `--calculate-per-token-loss`; default LR 1e-6; `--eps-clip-high` unset by default). Still open: which other extras (KL, filtering, importance-sampling correction) are on by default.
6. **Infrastructure failures and timeouts in GRPO groups (decided 2026-10-23).**
   - **Rule: exclude an attempt only when the policy cannot have caused its failure. Everything else keeps its reward:** the verifier's result, or 0 when no verifier ran.
   - **Excluded (infra), from a fixed list:**
     - the container never started (`EnvironmentStartTimeoutError`) or agent setup failed (`AgentSetupTimeoutError`);
     - the agent server answered `Error: …`, `TaskNotFound`, `InvalidInstanceId`, `InvalidSessionServerId`, `Flushed` or `ImportError`;
     - the agent server couldn't be reached (Miles' agent function returns nothing; today that trains as reward 0);
     - no model call was recorded (Miles drops the whole group itself, `apply_aborted_filter`).
   - **Kept as failures:**
     - agent timeouts (the verifier still runs, so work finished in time counts);
     - the 48k limit (reward 0);
     - verifier timeouts and crashes;
     - mid-episode errors.
   - **Exclude per attempt, not per group.** The excluded attempt gets loss mask 0 (`Sample.remove_sample`). The group's mean and std are computed over the remaining attempts, reusing Miles' own GRPO normalization (`grpo/infra_filter.py` via `--custom-reward-post-process-path`). If fewer than 4 of 8 remain, the whole group is masked (the std is too noisy). If more than 10% of a step is excluded, the run stops: that's an outage, not noise.
   - **Why per attempt:** Miles' built-in path drops the whole group and refills it with 8 more trials. At a 2% infra rate that throws out ~15% of groups (1 − 0.98⁸), mostly good attempts, and adds ~1.5 waves per step. Applied to timeouts, it's unworkable: 60% of groups at Luna's 10.7% timeout rate, 99.9% at Qwen3.5-27B's 63%, where the refills never end.
   - **Why timeouts stay failures:**
     - The policy causes them (long commands, loops, long replies), and the reward should push those down. In a group with 2 passes and 6 timeouts, scoring the timeouts 0 teaches "finish in time"; masking them leaves no comparison.
     - Masking them would let the policy escape a low reward by timing out.
     - Masking would drop the hardest tasks from training while evaluation still counts their timeouts.
     - **The fix for a high timeout rate is less thinking and shorter output in rollouts**, not hiding the zeros. The one exception is false timeouts from CPU overload, handled by concurrency (D14): compare each task's timeout rate with Phase 0 at the same concurrency.
   - **Mid-episode "infra" errors are mostly the policy's doing.** On the ledger's no-reward trials, checked 2026-10-23:
     - The tmux send-keys failures (22) mostly follow the model sending `set -e` / `set -eu` into the interactive shell, which exits on the next failing command and takes the tmux session with it.
     - The "Command timed out after 120 seconds" errors (16) are `docker compose exec` calls collecting terminal output from a container that stopped responding. They're spread over 11 tasks; the cause is either the policy's own processes or host load.
     - Both score 0. Their rate is logged; a jump in one step points to the host.
   - **Evidence (web, 2026-10-23):**
     - Miles fork PR celve/miles#7: discards pre-start and unreachable-service failures; timeouts, step limits and `max_seq_len` "stay scored 0" so a policy can't escape a low reward.
     - iCoder-27B: a sentinel reward for unjudgeable attempts, removed from group statistics, advantage 0, normalized over the valid ones.
     - ClawGym II: sandbox failures excluded; wall-clock timeouts terminated, not excluded.
     - SWE-Master: masking timeouts after an SFT warm start led to "continuous degradation in reward and eventual training collapse".
     - rStar2-Agent and DAPO: filtering over-long outputs gave no benefit.
     - DeepSWE masked timeouts successfully, but trained from scratch on easier data.
   - **Needs:** `grpo/harbor_parser.patch` also returns the trial's exception type (in `agent_metrics`). Today's `exit_status` reports both the 48k limit and tmux failures as `AgentError`, and a container-start timeout as `TimeLimitExceeded`.

---

## 9. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Native LoRA on 27B covers only attention in the 16 GQA layers (~42M params), which may be too small to learn much | Try Bridge with `all-linear` (Phase 2). Otherwise: accept the small scope and measure; or run QLoRA/Bridge from a patched spec; or move to Qwen3.5-35B-A3B, which Bridge LoRA lists |
| R2 | Colocated phase switches run out of memory, or offload is slow | Phase 2 profiling. Reduce the context to 16k, or lower SGLang's `mem-fraction-static` |
| R3 | Docker throughput sets the pace (slow steps, long tails); with 16 CPUs, contention causes false timeouts | 16–24 concurrent trials (D14), capped agent timeouts, async rollout later, a separate sandbox host if needed |
| R4 | Too few of the 20 training tasks give signal for Qwen3.6-27B (Qwen3.5-27B: 11 of 20) | Phase 0 measures this. If too few, add training tasks from outside the test and validation sets (e.g. the 10 non-selected Luna tasks not in validation: chess-best-move, circuit-fibsqrt, extract-elf, feal-differential-cryptanalysis, make-mips-interpreter, model-extraction-relu-logits, path-tracing-reverse, regex-chess, rstan-to-pystan, winning-avg-corewars), never from the test or validation set |
| R5 | The chat template rewrites history (thinking blocks), breaking TITO prefixes | Thinking off for now (D13). When it's turned on, use `preserve_thinking=true` (§3c) so the prompt stays a prefix extension |
| R6 | `terminus-2` summarization rewrites history mid-trajectory, so token IDs no longer form one sequence | Summarization off (`HARBOR_TERMINUS_2_ENABLE_SUMMARIZE=false` in `scripts/agent_server.sh`, which also disables proactive summarization); trajectories end at the context limit (§2b, D10) |
| R7 | Contamination: training on TB 2.1 then reporting TB 2.1 scores | D12. Never report official TB 2.1 numbers for an adapter trained on TB 2.1 tasks |
| R8 | Reward hacking (e.g. `pip install --upgrade <package-under-test>`) | The verifier usually catches it. Review a sample of passing trajectories each pilot run in the browser |
| R9 | Miles' Harbor connector is still experimental | Fallback: SkyRL's Harbor integration (more used, but re-tokenizes, so TITO would need patching), or a thin custom loop on Harbor's `collect_rollout_details` |
| R10 | Host RAM (200 GB) runs out: CPU base copy + offloaded trainer state + containers | Memory-aware concurrency cap (D14). Drop `--lora-base-cpu-backup` (frees 54 GB; the base reloads from NVMe, ~10–20 s per phase switch). Or move sandboxes to a separate Docker host or a cloud provider |
