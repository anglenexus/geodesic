# TB 2.1 task notes (Phase 0 oracle, Mac + Rosetta, 2026-10-11)

Oracle run: `~/code/jobs/tb21_oracle_phase0` (25 tasks, 20 passed).

## Removed from the selected 20
| Task | Reason | Status |
|---|---|---|
| mcmc-sampling-stan | Oracle fails on any platform: the solution installs `rstan` from CRAN at runtime; newer `RcppParallel` needs `cmake`, which the image lacks (environment drift) | Dropped |
| protein-assembly | Oracle fails: `dnachisel` stochastic optimizer raised `NoSolutionError` (non-deterministic reference solution) | Dropped |
| qemu-startup | Rosetta: `Unimplemented syscall number 282` (QEMU can't run under emulation) | **Deferred: re-test oracle on EC2 (x86)** |
| tune-mjcf | Rosetta: `Fatal Python error: Illegal instruction` (MuJoCo CPU instructions) | **Deferred: re-test oracle on EC2 (x86)** |
| qemu-alpine-ssh (reserve) | Same QEMU/Rosetta failure as qemu-startup | **Deferred: re-test on EC2** |

## Replacements (all passed the oracle on the Mac)
mcmc-sampling-stan → write-compressor, protein-assembly → sanitize-git-repo,
qemu-startup → model-extraction-relu-logits, tune-mjcf → chess-best-move.

## Reserves
The strict stability filter yields exactly 25 tasks, so no strict-filter reserves remain.
If Phase 2 pushes tasks out of the 10–90% band, draw from a relaxed filter (to be computed and oracle-tested first).

## Relaxed-filter reserves: oracle on Mac (2026-10-12, `~/code/jobs/tb21_oracle_reserves`)
| Task | Oracle | Notes |
|---|---|---|
| sqlite-db-truncate | pass | usable reserve #1 |
| fix-ocaml-gc | pass | usable reserve #2 (agent timeout 3600 s) |
| rstan-to-pystan | pass | usable reserve #3 (data-science) |
| schemelike-metacircular-eval | pass | usable reserve #4 |
| bn-fit-modify | verifier timeout (3600 s) | Correct reference solution still timed out, likely slow under Rosetta. **Deferred: re-test on EC2** |
| build-cython-ext | fail | Environment drift: solution installs latest packages; repo's own tests fail. Dropped |

## Decisions
- 2026-10-13: **keep crack-7z-hash**. TB 2.1 raised its memory (2→4 GB) and agent timeout (900→1800 s) vs 2.0, so its 2.0-based pass rate (terminus-2 0.55) likely understates 2.1. The pilot measures it; swap only if it falls outside 10–90%.

## Final 20 (2026-10-19)
`state/tb21_selected_tasks.txt` now holds the 20 tasks chosen by the rule in
`state/tb21_candidates_round2_results.md`: Luna pass rate in 10–90%, cost ≤ 80th percentile, length ≤ 90th
percentile. That's the 12 original in-band tasks plus 8 round-2 replacements: polyglot-c-py, largest-eigenval,
overfull-hbox, db-wal-recovery, dna-insert, mteb-retrieve, adaptive-rejection-sampler, polyglot-rust-c.
- Removed (out of band after 85 attempts): model-extraction-relu-logits, extract-elf, chess-best-move (too hard);
  feal-differential-cryptanalysis, fix-code-vulnerability, pytorch-model-cli, custom-memory-heap-crash, kv-store-grpc (too easy).
- Kept: write-compressor (12%, $0.066), even though 85% of its trials end in an agent timeout (user decision).
- Round-2 oracle failures on the Mac: compile-compcert (40-min oracle timeout under Rosetta), mteb-leaderboard.
