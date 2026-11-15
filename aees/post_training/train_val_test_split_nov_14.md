# Train / validation / test split, 2026-11-14

Decided 2026-11-14 (user). Replaces the earlier lists (`state/tb21_selected_tasks.txt`, `tb21_test20_tasks.txt`, `tb21_val8_tasks.txt`, and the 20-task list `tb21_train20_l1114.txt` used by the paused run `l1114`). The earlier sets were not matched: Qwen3.6-27B (Tmax settings) passed **91%** of the old test set but **46%** of the old training set, and GPT-5.6 Luna passed 87% vs 54-57%.

**Task lists** (in `aees/harbor/state/`): `tb21_split1114_train.txt` (20), `tb21_split1114_val.txt` (8), `tb21_split1114_test.txt` (20). Machine-readable record: `state/split_nov14.json`.

## How the split was made

- **Pool: 66 TB 2.1 tasks** that have success rates for both models:
  - **Qwen3.6-27B:** the full Tmax-aligned TB 2.1 run `tb21full_1112_1454b` (thinking on, temperature 0.7 / top_p 0.95, 65,536 context, 64 turns, 120 s per command, each task's own limits), 5 attempts per task; per-task table in `../post_training/serving_speed_on_TB2.md` §6.
  - **GPT-5.6 Luna:** every Luna trial in the ledger at the API's default reasoning (measured to match `medium`), without the Tmax turn/context limits. 9 to 509 attempts per task (10 for most tasks outside the old training set).
  - Not in the pool: the 22 TB 2.1 tasks with no Luna data (e.g. qemu-*, tune-mjcf, bn-fit-modify, sam-cell-seg), and `configure-git-webserver` (Tmax excludes it).
- **Random draw with a balance check:** 200,000 random 20 / 8 / 20 assignments of the pool (seed 20261114); kept the one whose sets have the closest mean per-task success rates for **both** models and the most even mix of task types. 18 pool tasks are left out at random.
- **Task types:** each task's `category` in its `task.toml`, merged into 6 types: software (software-engineering, debugging, games), systems (system-administration, file-operations, personal-assistant), security, science/math (scientific-computing, mathematics, optimization), data (data-science, data-processing, data-querying), ML (model-training, machine-learning, video-processing).
- **Near-duplicate tasks stay in one split** (decision 2026-11-14, option 1), so the test set measures generalization, not recall of a near-identical training task: feal-differential + feal-linear (train), polyglot-c-py + polyglot-rust-c (train), path-tracing + path-tracing-reverse (test). Rationale: train/test leakage through near-duplicates inflates test scores (Kapoor & Narayanan 2023, *Patterns*; Allamanis 2019, Onward!; Lee et al. 2022, ACL; scikit-learn GroupShuffleSplit).

## The success rates are matched

Mean of per-task success rates (each task weighted equally), with the spread across tasks (SD), and pooled trial-level rates.

| Set | Tasks | Luna mean (SD) | Qwen mean (SD) | Luna pooled | Qwen pooled | Mean agent limit (min) |
|---|---|---|---|---|---|---|
| **train** | 20 | **64.7%** (32) | **53.0%** (44) | 56.2% | 53.0% | 30 |
| **val** | 8 | **66.4%** (36) | **55.0%** (36) | 47.9% | 55.0% | 22 |
| **test** | 20 | **65.1%** (37) | **53.0%** (41) | 67.4% | 53.0% | 21 |
| left out | 18 | 63.4% (33) | 63.3% (43) | 48.8% | 63.3% | 31 |

Largest difference between the three sets' means: **1.7 points for Luna, 2.0 points for Qwen.** (Pooled rates differ more because the number of Luna attempts per task varies; the per-task mean is the balanced quantity.)

**Task types per set:**

| Set | software | systems | security | science/math | data | ML |
|---|---|---|---|---|---|---|
| train | 7 | 3 | 2 | 3 | 3 | 2 |
| val | 3 | 1 | 0 | 1 | 2 | 1 |
| test | 9 | 4 | 1 | 2 | 2 | 2 |

## Train set (20 tasks)

| Task | Type (category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
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
| build-pov-ray | software (software-engineering) | 5 / 7 | 71% | 0 / 5 | 0% | 200 |
| largest-eigenval | science/math (mathematics) | 199 / 434 | 46% | 1 / 5 | 20% | 15 |
| count-dataset-tokens | ML (model-training) | 68 / 387 | 18% | 2 / 5 | 40% | 15 |
| db-wal-recovery | systems (file-operations) | 230 / 435 | 53% | 0 / 5 | 0% | 15 |
| chess-best-move | software (games) | 4 / 85 | 5% | 0 / 5 | 0% | 15 |
| train-fasttext | ML (model-training) | 0 / 10 | 0% | 0 / 5 | 0% | 60 |

## Val set (8 tasks)

| Task | Type (category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
|---|---|---|---|---|---|---|
| pytorch-model-cli | ML (model-training) | 79 / 85 | 93% | 5 / 5 | 100% | 15 |
| build-pmars | software (software-engineering) | 9 / 10 | 90% | 5 / 5 | 100% | 15 |
| nginx-request-logging | systems (system-administration) | 10 / 10 | 100% | 4 / 5 | 80% | 15 |
| fix-ocaml-gc | software (software-engineering) | 10 / 10 | 100% | 3 / 5 | 60% | 60 |
| adaptive-rejection-sampler | science/math (scientific-computing) | 292 / 336 | 87% | 3 / 5 | 60% | 15 |
| cancel-async-tasks | software (software-engineering) | 88 / 389 | 23% | 1 / 5 | 20% | 15 |
| query-optimize | data (data-science) | 1 / 10 | 10% | 1 / 5 | 20% | 15 |
| mteb-retrieve | data (data-science) | 126 / 434 | 29% | 0 / 5 | 0% | 30 |

## Test set (20 tasks)

| Task | Type (category) | Luna: passed / graded | Luna | Qwen: passed / 5 | Qwen | Agent limit (min) |
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

## Left out (18 tasks, at random)

break-filter-js-from-html, circuit-fibsqrt, distribution-search, dna-insert, financial-document-processor, fix-code-vulnerability, fix-git, merge-diff-arc-agi-task, openssl-selfsigned-cert, overfull-hbox, portfolio-optimization, prove-plus-comm, regex-chess, reshard-c4-data, sanitize-git-repo, sparql-university, winning-avg-corewars, write-compressor

## Caveats

- Qwen rates come from 5 attempts per task (steps of 20 points); Luna rates from 9-509 attempts. Per-task rates are noisy; the set means are what is matched.
- Luna rates are at the API-default reasoning without Tmax's turn/context limits; the paused medium run (`l1114`, Tmax limits) gave similar rates on the tasks it covered (57.0% vs 54.4% on the old training set).
- 18 of the old 20 training tasks moved: the old training set was chosen for Luna's 10-90% band, which made it harder for Luna than the rest of the benchmark.

## Next

- Collect the Q-function training data on the new training set (GPT-5.6 Luna, reasoning medium, Tmax-aligned settings); the paused run `l1114` is kept as its own partition and not extended.
- `../post_training/PLAN.md` §2a and the post-training task lists should be updated to these sets before the Phase 0 baseline.
