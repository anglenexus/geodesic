# Train / test / validation split, 2026-11-14 (final)

Decided 2026-11-14 by the user and **frozen**: no task moves between sets from here on without the user's approval, recorded in the change log below. Replaces `state/tb21_selected_tasks.txt` (old train), `tb21_test20_tasks.txt` (old test), `tb21_val8_tasks.txt` (old validation) and `tb21_train20_l1114.txt`. The old sets were not matched: Qwen3.6-27B (Tmax settings) passed 46% of the old train set and 91% of the old test set.

**Task lists** (`aees/harbor/state/`): `tb21_split1114_train.txt` (**20**), `tb21_split1114_test.txt` (**20**), `tb21_split1114_val.txt` (**8**). Machine-readable record with seed and change log: `state/split_nov14.json`.

## Success rates per set (matched)

Mean of per-task success rates (each task weighted equally), SD across tasks in brackets.

| Set | Tasks | GPT-5.6 Luna | Qwen3.6-27B (Tmax settings) | Mean agent limit (min) | Task types: software / systems / security / science-math / data / ML |
|---|---|---|---|---|---|
| **Train** | 20 | **66.2%** (28) | **55.0%** (42) | 19 | 6 / 3 / 3 / 4 / 3 / 1 |
| **Test** | 20 | **65.1%** (37) | **53.0%** (41) | 21 | 9 / 4 / 1 / 2 / 2 / 2 |
| **Validation** | 8 | **66.4%** (36) | **55.0%** (36) | 22 | 3 / 1 / 0 / 1 / 2 / 1 |

Largest difference between the three sets: **1.3 points for Luna, 2.0 points for Qwen.**

**Sources of the rates.** Qwen3.6-27B: the full Tmax-aligned TB 2.1 run `tb21full_1112_1454b` (thinking on, temperature 0.7 / top_p 0.95, 65,536 context, 64 turns, 120 s per command, each task's own limits), 5 attempts per task (`../post_training/serving_speed_on_TB2.md` §6). GPT-5.6 Luna: all Luna trials in the ledger at the API-default reasoning (measured to match `medium`), 9-509 attempts per task.

## Train set (20 tasks)

| Task | Type (TB category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
|---|---|---|---|---|---|---|
| cobol-modernization | software (software-engineering) | 10 / 10 | 100% | 5 / 5 | 100% | 15 |
| multi-source-data-merger | data (data-processing) | 10 / 10 | 100% | 5 / 5 | 100% | 15 |
| regex-log | data (data-processing) | 9 / 10 | 90% | 5 / 5 | 100% | 15 |
| polyglot-c-py | software (software-engineering) | 280 / 338 | 83% | 5 / 5 | 100% | 15 |
| vulnerable-secret | security (security) | 10 / 10 | 100% | 4 / 5 | 80% | 15 |
| constraints-scheduling | systems (personal-assistant) | 7 / 9 | 78% | 5 / 5 | 100% | 20 |
| custom-memory-heap-crash | software (debugging) | 85 / 85 | 100% | 3 / 5 | 60% | 30 |
| code-from-image | software (software-engineering) | 5 / 10 | 50% | 5 / 5 | 100% | 20 |
| crack-7z-hash | security (security) | 251 / 383 | 66% | 4 / 5 | 80% | 30 |
| log-summary-date-ranges | data (data-processing) | 3 / 10 | 30% | 5 / 5 | 100% | 15 |
| sqlite-with-gcov | systems (system-administration) | 185 / 479 | 39% | 4 / 5 | 80% | 15 |
| feal-differential-cryptanalysis | science/math (mathematics) | 78 / 85 | 92% | 0 / 5 | 0% | 30 |
| feal-linear-cryptanalysis | science/math (mathematics) | 9 / 10 | 90% | 0 / 5 | 0% | 30 |
| polyglot-rust-c | software (software-engineering) | 373 / 434 | 86% | 0 / 5 | 0% | 15 |
| dna-insert | science/math (scientific-computing) | 171 / 337 | 51% | 1 / 5 | 20% | 30 |
| largest-eigenval | science/math (mathematics) | 199 / 434 | 46% | 1 / 5 | 20% | 15 |
| count-dataset-tokens | ML (model-training) | 68 / 387 | 18% | 2 / 5 | 40% | 15 |
| sanitize-git-repo | security (security) | 187 / 509 | 37% | 1 / 5 | 20% | 15 |
| db-wal-recovery | systems (file-operations) | 230 / 435 | 53% | 0 / 5 | 0% | 15 |
| write-compressor | software (software-engineering) | 74 / 401 | 18% | 0 / 5 | 0% | 15 |

## Test set (20 tasks)

| Task | Type (TB category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
|---|---|---|---|---|---|---|
| git-leak-recovery | software (software-engineering) | 9 / 9 | 100% | 5 / 5 | 100% | 15 |
| hf-model-inference | data (data-science) | 10 / 10 | 100% | 5 / 5 | 100% | 15 |
| kv-store-grpc | software (software-engineering) | 85 / 85 | 100% | 5 / 5 | 100% | 15 |
| modernize-scientific-stack | science/math (scientific-computing) | 10 / 10 | 100% | 5 / 5 | 100% | 10 |
| pypi-server | software (software-engineering) | 10 / 10 | 100% | 5 / 5 | 100% | 15 |
| pytorch-model-recovery | ML (model-training) | 10 / 10 | 100% | 5 / 5 | 100% | 15 |
| headless-terminal | software (software-engineering) | 8 / 10 | 80% | 5 / 5 | 100% | 15 |
| sqlite-db-truncate | software (debugging) | 10 / 10 | 100% | 3 / 5 | 60% | 15 |
| git-multibranch | systems (system-administration) | 5 / 7 | 71% | 3 / 5 | 60% | 15 |
| password-recovery | security (security) | 363 / 414 | 88% | 2 / 5 | 40% | 15 |
| large-scale-text-editing | systems (file-operations) | 273 / 405 | 67% | 3 / 5 | 60% | 20 |
| mailman | systems (system-administration) | 261 / 393 | 66% | 3 / 5 | 60% | 30 |
| llm-inference-batching-scheduler | ML (machine-learning) | 6 / 10 | 60% | 2 / 5 | 40% | 30 |
| rstan-to-pystan | data (data-science) | 8 / 10 | 80% | 0 / 5 | 0% | 30 |
| schemelike-metacircular-eval | software (software-engineering) | 5 / 9 | 56% | 0 / 5 | 0% | 40 |
| extract-elf | systems (file-operations) | 2 / 85 | 2% | 2 / 5 | 40% | 15 |
| path-tracing | software (software-engineering) | 2 / 10 | 20% | 0 / 5 | 0% | 30 |
| path-tracing-reverse | software (software-engineering) | 1 / 10 | 10% | 0 / 5 | 0% | 30 |
| model-extraction-relu-logits | science/math (mathematics) | 1 / 85 | 1% | 0 / 5 | 0% | 15 |
| make-mips-interpreter | software (software-engineering) | 0 / 10 | 0% | 0 / 5 | 0% | 30 |

## Validation set (8 tasks)

| Task | Type (TB category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
|---|---|---|---|---|---|---|
| pytorch-model-cli | ML (model-training) | 79 / 85 | 93% | 5 / 5 | 100% | 15 |
| build-pmars | software (software-engineering) | 9 / 10 | 90% | 5 / 5 | 100% | 15 |
| nginx-request-logging | systems (system-administration) | 10 / 10 | 100% | 4 / 5 | 80% | 15 |
| fix-ocaml-gc | software (software-engineering) | 10 / 10 | 100% | 3 / 5 | 60% | 60 |
| adaptive-rejection-sampler | science/math (scientific-computing) | 292 / 336 | 87% | 3 / 5 | 60% | 15 |
| cancel-async-tasks | software (software-engineering) | 88 / 389 | 23% | 1 / 5 | 20% | 15 |
| query-optimize | data (data-science) | 1 / 10 | 10% | 1 / 5 | 20% | 15 |
| mteb-retrieve | data (data-science) | 126 / 434 | 29% | 0 / 5 | 0% | 30 |

## How the split was made

- **Pool:** the 66 TB 2.1 tasks with success rates for both models (all 89 minus 22 without Luna data and `configure-git-webserver`, which Tmax excludes).
- **Random draw:** 200,000 random 20 / 8 / 20 assignments (seed 20261114); kept the one with the closest mean per-task success rates for both models and the most even mix of 6 task types (TB `category` merged: software, systems, security, science/math, data, ML).
- **Near-duplicate tasks stay in one split** (no train/test leakage): feal-differential + feal-linear (train), polyglot-c-py + polyglot-rust-c (train), path-tracing + path-tracing-reverse (test). References: Kapoor & Narayanan 2023 (*Patterns*), Allamanis 2019 (Onward!), Lee et al. 2022 (ACL), scikit-learn GroupShuffleSplit.
- **Change after the draw (2026-11-14, user-approved):** in train, train-fasttext (Luna 0%), chess-best-move (Luna 5%) and build-pov-ray (30% harness no-reward) were replaced by dna-insert, sanitize-git-repo and write-compressor, from the left-out tasks, keeping the sets matched; Luna must be able to produce training successes.
- **Test and validation kept as drawn.** About half their tasks are at Qwen 0/5 or 5/5 (as in train; TB 2.1 has many such tasks), which limits how much change they can show; no swap-in candidates were clearly better without longer time limits.

**Left out (18):** break-filter-js-from-html, build-pov-ray, chess-best-move, circuit-fibsqrt, distribution-search, financial-document-processor, fix-code-vulnerability, fix-git, merge-diff-arc-agi-task, openssl-selfsigned-cert, overfull-hbox, portfolio-optimization, prove-plus-comm, regex-chess, reshard-c4-data, sparql-university, train-fasttext, winning-avg-corewars

## Caveats

- Qwen rates: 5 attempts per task (steps of 20 points). Luna rates: 9-509 attempts per task. Set means are what is matched; per-task rates are noisy.
- Luna rates are at the API-default reasoning without Tmax's turn/context limits; the Tmax-aligned medium run (`l1114`, paused) gave similar rates where it overlapped.

## Uses

- **Train:** Luna data collection for the Q-function reward model; SFT/RL rollouts. Only train tasks.
- **Validation:** checkpoint selection during training.
- **Test:** base model and final checkpoint only.
