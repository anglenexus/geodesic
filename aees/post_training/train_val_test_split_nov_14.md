# Train / test / validation split, 2026-11-14 (final)

Decided 2026-11-14 by the user and **frozen**: no task moves between sets without the user's approval, recorded in the change log below. Replaces `state/tb21_selected_tasks.txt` (old train), `tb21_test20_tasks.txt` (old test), `tb21_val8_tasks.txt` (old validation) and `tb21_train20_l1114.txt`. The old sets were not matched: Qwen3.6-27B (Tmax settings) passed 46% of the old train set and 91% of the old test set.

**Task lists** (`aees/harbor/state/`): `tb21_split1114_train.txt` (**20**), `tb21_split1114_test.txt` (**20**), `tb21_split1114_val.txt` (**8**). Machine-readable record with seed and change log: `state/split_nov14.json`.

## Success rates per set

Mean of per-task success rates (each task weighted equally), SD across tasks in brackets. Qwen is known for every task; Luna has not been run on 3 test tasks (see below).

| Set | Tasks | Qwen3.6-27B (Tmax settings) | GPT-5.6 Luna (measured tasks) | Tasks at Qwen 0/5 or 5/5 | Mean agent limit (min) | Types: software / systems / security / science-math / data / ML |
|---|---|---|---|---|---|---|
| **Train** | 20 | **55.0%** (42) | **66.2%** (28), 20 of 20 tasks | 12 | 19 | 6 / 3 / 3 / 4 / 3 / 1 |
| **Test** | 20 | **53.0%** (34) | **62.8%** (34), 17 of 20 tasks | 8 | 21 | 6 / 4 / 2 / 2 / 4 / 2 |
| **Validation** | 8 | **55.0%** (36) | **66.4%** (36), 8 of 8 tasks | 3 | 22 | 3 / 1 / 0 / 1 / 2 / 1 |

**Balance:** Qwen means differ by at most 2.0 points (55.0 / 53.0 / 55.0). Luna means: 66.2 / 62.8 (17 measured test tasks) / 66.4. The test set's Luna mean is provisional until mcmc-sampling-stan, build-cython-ext and tune-mjcf are run with Luna (not done: no API runs without approval). Luna is not run on test or validation for training; its role here is only the balance check.

**Sources.** Qwen3.6-27B: full Tmax-aligned TB 2.1 run `tb21full_1112_1454b` (thinking on, temperature 0.7 / top_p 0.95, 65,536 context, 64 turns, 120 s per command, each task's own limits), 5 attempts per task (`../post_training/serving_speed_on_TB2.md` §6). GPT-5.6 Luna: Luna trials in the ledger at the API-default reasoning (measured to match `medium`), 9-509 attempts per task.

## Train set (20 tasks)

| Task | Type (TB category) | Qwen: passed / 5 | Qwen | Luna: passed / graded | Luna | Agent limit (min) |
|---|---|---|---|---|---|---|
| cobol-modernization | software (software-engineering) | 5 / 5 | 100% | 10 / 10 | 100% | 15 |
| multi-source-data-merger | data (data-processing) | 5 / 5 | 100% | 10 / 10 | 100% | 15 |
| regex-log | data (data-processing) | 5 / 5 | 100% | 9 / 10 | 90% | 15 |
| polyglot-c-py | software (software-engineering) | 5 / 5 | 100% | 280 / 338 | 83% | 15 |
| constraints-scheduling | systems (personal-assistant) | 5 / 5 | 100% | 7 / 9 | 78% | 20 |
| code-from-image | software (software-engineering) | 5 / 5 | 100% | 5 / 10 | 50% | 20 |
| log-summary-date-ranges | data (data-processing) | 5 / 5 | 100% | 3 / 10 | 30% | 15 |
| vulnerable-secret | security (security) | 4 / 5 | 80% | 10 / 10 | 100% | 15 |
| crack-7z-hash | security (security) | 4 / 5 | 80% | 251 / 383 | 66% | 30 |
| sqlite-with-gcov | systems (system-administration) | 4 / 5 | 80% | 185 / 479 | 39% | 15 |
| custom-memory-heap-crash | software (debugging) | 3 / 5 | 60% | 85 / 85 | 100% | 30 |
| count-dataset-tokens | ML (model-training) | 2 / 5 | 40% | 68 / 387 | 18% | 15 |
| dna-insert | science/math (scientific-computing) | 1 / 5 | 20% | 171 / 337 | 51% | 30 |
| largest-eigenval | science/math (mathematics) | 1 / 5 | 20% | 199 / 434 | 46% | 15 |
| sanitize-git-repo | security (security) | 1 / 5 | 20% | 187 / 509 | 37% | 15 |
| feal-differential-cryptanalysis | science/math (mathematics) | 0 / 5 | 0% | 78 / 85 | 92% | 30 |
| feal-linear-cryptanalysis | science/math (mathematics) | 0 / 5 | 0% | 9 / 10 | 90% | 30 |
| polyglot-rust-c | software (software-engineering) | 0 / 5 | 0% | 373 / 434 | 86% | 15 |
| db-wal-recovery | systems (file-operations) | 0 / 5 | 0% | 230 / 435 | 53% | 15 |
| write-compressor | software (software-engineering) | 0 / 5 | 0% | 74 / 401 | 18% | 15 |

## Test set (20 tasks)

| Task | Type (TB category) | Qwen: passed / 5 | Qwen | Luna: passed / graded | Luna | Agent limit (min) |
|---|---|---|---|---|---|---|
| hf-model-inference | data (data-science) | 5 / 5 | 100% | 10 / 10 | 100% | 15 |
| modernize-scientific-stack | science/math (scientific-computing) | 5 / 5 | 100% | 10 / 10 | 100% | 10 |
| pytorch-model-recovery | ML (model-training) | 5 / 5 | 100% | 10 / 10 | 100% | 15 |
| headless-terminal | software (software-engineering) | 5 / 5 | 100% | 8 / 10 | 80% | 15 |
| financial-document-processor | data (data-processing) | 4 / 5 | 80% | 274 / 407 | 67% | 20 |
| mcmc-sampling-stan | data (data-science) | 4 / 5 | 80% | not measured | – | 30 |
| sqlite-db-truncate | software (debugging) | 3 / 5 | 60% | 10 / 10 | 100% | 15 |
| git-multibranch | systems (system-administration) | 3 / 5 | 60% | 5 / 7 | 71% | 15 |
| large-scale-text-editing | systems (file-operations) | 3 / 5 | 60% | 273 / 405 | 67% | 20 |
| mailman | systems (system-administration) | 3 / 5 | 60% | 261 / 393 | 66% | 30 |
| build-cython-ext | software (debugging) | 3 / 5 | 60% | not measured | – | 15 |
| password-recovery | security (security) | 2 / 5 | 40% | 363 / 414 | 88% | 15 |
| llm-inference-batching-scheduler | ML (machine-learning) | 2 / 5 | 40% | 6 / 10 | 60% | 30 |
| extract-elf | systems (file-operations) | 2 / 5 | 40% | 2 / 85 | 2% | 15 |
| break-filter-js-from-html | security (security) | 2 / 5 | 40% | 0 / 10 | 0% | 20 |
| tune-mjcf | science/math (scientific-computing) | 2 / 5 | 40% | not measured | – | 15 |
| rstan-to-pystan | data (data-science) | 0 / 5 | 0% | 8 / 10 | 80% | 30 |
| schemelike-metacircular-eval | software (software-engineering) | 0 / 5 | 0% | 5 / 9 | 56% | 40 |
| path-tracing | software (software-engineering) | 0 / 5 | 0% | 2 / 10 | 20% | 30 |
| path-tracing-reverse | software (software-engineering) | 0 / 5 | 0% | 1 / 10 | 10% | 30 |

## Validation set (8 tasks)

| Task | Type (TB category) | Qwen: passed / 5 | Qwen | Luna: passed / graded | Luna | Agent limit (min) |
|---|---|---|---|---|---|---|
| pytorch-model-cli | ML (model-training) | 5 / 5 | 100% | 79 / 85 | 93% | 15 |
| build-pmars | software (software-engineering) | 5 / 5 | 100% | 9 / 10 | 90% | 15 |
| nginx-request-logging | systems (system-administration) | 4 / 5 | 80% | 10 / 10 | 100% | 15 |
| fix-ocaml-gc | software (software-engineering) | 3 / 5 | 60% | 10 / 10 | 100% | 60 |
| adaptive-rejection-sampler | science/math (scientific-computing) | 3 / 5 | 60% | 292 / 336 | 87% | 15 |
| cancel-async-tasks | software (software-engineering) | 1 / 5 | 20% | 88 / 389 | 23% | 15 |
| query-optimize | data (data-science) | 1 / 5 | 20% | 1 / 10 | 10% | 15 |
| mteb-retrieve | data (data-science) | 0 / 5 | 0% | 126 / 434 | 29% | 30 |

## How the split was made

- **Pool and draw:** the 66 TB 2.1 tasks with success rates for both models; 200,000 random 20 / 8 / 20 assignments (seed 20261114); kept the one with the closest mean per-task success rates for both models and the most even mix of 6 task types (TB `category` merged: software, systems, security, science/math, data, ML).
- **Near-duplicate tasks stay in one split** (no train/test leakage): feal-differential + feal-linear (train), polyglot-c-py + polyglot-rust-c (train), path-tracing + path-tracing-reverse (test). References: Kapoor & Narayanan 2023 (*Patterns*), Allamanis 2019 (Onward!), Lee et al. 2022 (ACL), scikit-learn GroupShuffleSplit.

## Change log (all approved by the user, 2026-11-14)

- **train:** out train-fasttext, chess-best-move, build-pov-ray; in dna-insert, sanitize-git-repo, write-compressor. Luna cannot produce training successes (train-fasttext 0%, chess-best-move 5%) or 30% harness no-reward (build-pov-ray); replacements chosen from the left-out pool to keep the sets matched.
- **test:** out git-leak-recovery, kv-store-grpc, pypi-server, make-mips-interpreter, model-extraction-relu-logits; in financial-document-processor, break-filter-js-from-html, mcmc-sampling-stan, build-cython-ext, tune-mjcf. user-chosen swap-ins (Qwen 1-4/5); removed three Qwen 5/5 and two Qwen 0/5 + Luna ~0% tasks; balanced on Qwen only (Luna not measured for mcmc-sampling-stan, build-cython-ext, tune-mjcf).

**Left out (21):** build-pov-ray, chess-best-move, circuit-fibsqrt, distribution-search, fix-code-vulnerability, fix-git, git-leak-recovery, kv-store-grpc, make-mips-interpreter, merge-diff-arc-agi-task, model-extraction-relu-logits, openssl-selfsigned-cert, overfull-hbox, portfolio-optimization, prove-plus-comm, pypi-server, regex-chess, reshard-c4-data, sparql-university, train-fasttext, winning-avg-corewars

## Uses

- **Train:** Luna data collection for the Q-function reward model; SFT/RL rollouts. Only train tasks.
- **Validation:** checkpoint selection during training.
- **Test:** base model and final checkpoint only.

## Training data: GPT-5.6 Luna, high reasoning (run h1115, 2026-11-15)

For the reward-learning (Q-function) and post-training agents. Collected on the **20 training tasks above only**.

**Run.** terminus-2 with the Tmax settings (64 turns, 120 s per command, 65,536-token context with stop-and-grade,
16,384 max tokens, each task's own time limits), `openai/gpt-5.6-luna` with **reasoning effort high**, called through
OpenAI's Responses API with **each step's reasoning sent back** (encrypted) on every later call, so Luna sees its own
earlier reasoning. Batch 1: 50 attempts per task (1,000 trials, 80.6% passed). Batch 2: attempts per task set from
the batch-1 pass rate for **300 successes per task** (~7,700 trials). Settings: `aees/harbor/state/h1115_agent_kwargs.json`;
Harbor build: harbor-miles + `post_training/grpo/{harbor_parser,command_timeout,responses_reasoning}.patch`.

**Where the data lives.**
- Ledger `aees/harbor/state/ledger.sqlite`: its own partition (scenario: model `openai/gpt-5.6-luna`, agent kwargs with
  `reasoning_effort: high` and `llm_kwargs.responses_carry_reasoning: true`). Rows point to trial folders; the ledger
  holds no text.
- Trial folders on the laptop: `~/code/jobs_remote/aees-l<k>/tb21_h1115_i<k>_<batch>_<task>/<task>__<time>__<id>/`
  (batch 1 also `tb21_h1115_i<k>_b1/`). After the EC2 boxes are terminated this mirror is the **only copy**: back it up.
- Per trial, `agent/trajectory.json`, per agent step:
  - `message`: the visible reply (terminus-2 JSON: analysis, plan, commands, task_complete)
  - `reasoning_content`: **OpenAI's summary of that step's reasoning** (`reasoning.summary: detailed`)
  - `metrics.extra.reasoning_tokens`: reasoning tokens billed for the step
  - `metrics.extra.reasoning_summary_parts`, `metrics.extra.reasoning_content_kind` = `"provider summary (detailed)"`
  - `metrics.extra.reasoning_encrypted`: the encrypted reasoning that was sent back to Luna. Only OpenAI can read it;
    kept so the exact model input can be reconstructed, not as training text.
- `result.json` → `agent_result.metadata.stopped_at_context_limit: true` marks trials stopped at 65,536 tokens; they are
  graded as they stand (not infra errors).

**What the summaries are.** OpenAI's first-person paraphrases of the hidden reasoning, written by a summarizer, **not
the raw reasoning** (OpenAI never returns it). Present on **~53% of steps** (95% of steps with ≥ 2,000 reasoning tokens,
~45% under 500) and on 98.6% of trajectories at least once. A step without one has `reasoning_content: null`. In samples
they match the step's reply (e.g. 3,540 reasoning tokens on write-compressor → 13 summary parts on the coding format).
No earlier Luna run (default or medium reasoning) stored any reasoning text.

**Clean replies before training (required).** In this run 15.3% of replies carry text the agent ignored: stray
fragments after the JSON (often non-ASCII, e.g. `}ుత్వ`; 0.7% of replies at medium reasoning), echoed prompt text, a
second full JSON reply, or prose before the JSON. terminus-2 executed only the first complete JSON object.
`aees/harbor/aees_harbor/reply_clean.py`: `clean_trajectory(traj)` returns a copy where each agent `message` is exactly
that object (an exact substring, nothing re-serialized) and records what it removed in the step's
`extra.reply_cleaning`; everything else (reasoning_content, metrics, observations) is unchanged. Replies the agent
could not parse (0.4%) are left as they are (`unparsed`). Raw files are never modified. Tests:
`aees/harbor/tests/unit/test_reply_clean.py`, including a check against terminus-2's own parser on every mirrored
trajectory (same object, same commands).

**Pipeline gap.** The reward-model export in `~/code/aees/reward` (and `post_training/sft/rollout.py`, which renders
thinking-off) does not read `reasoning_content` yet, and must apply `clean_trajectory` first.
