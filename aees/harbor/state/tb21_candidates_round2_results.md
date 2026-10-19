# Round-2 candidates: 10-attempt results (2026-10-18, Mac, terminus-2 + gpt-5.6-luna)

Job `~/code/jobs/tb21_cand_b1`; 160 trials ingested (held). Stopped by the $15 kill switch near the end
(final recorded cost $17.09; see "Kill-switch overshoot").

| Task | Pass / graded | Rate | $ / trajectory | Min / trajectory | Verdict |
|---|---|---|---|---|---|
| adaptive-rejection-sampler | 9/10 | 90% | 0.028 | 7.3 | in band (upper edge) |
| polyglot-rust-c | 9/10 | 90% | 0.015 | 3.6 | in band (upper edge) |
| polyglot-c-py | 8/10 | 80% | 0.006 | 2.4 | in band |
| largest-eigenval | 6/10 | 60% | 0.013 | 3.9 | in band |
| overfull-hbox | 6/10 | 60% | 0.015 | 5.3 | in band |
| db-wal-recovery | 5/10 | 50% | 0.006 | 1.6 | in band |
| dna-insert | 4/10 | 40% | 0.013 | 3.5 | in band |
| mteb-retrieve | 2/10 | 20% | 0.007 | 5.5 | in band |
| winning-avg-corewars | 9/10 | 90% | 0.103 | 25.2 | in band, expensive |
| rstan-to-pystan | 8/10 | 80% | 0.051 | 25.5 | in band, expensive |
| circuit-fibsqrt | 5/9 | 56% | 0.441 | 39.9 | in band, very expensive |
| schemelike-metacircular-eval | 5/9 | 56% | 0.097 | 19.6 | in band, expensive |
| path-tracing | 2/10 | 20% | 0.096 | 16.5 | in band, expensive |
| path-tracing-reverse | 1/10 | 10% | 0.137 | 9.6 | in band (lower edge), expensive |
| make-mips-interpreter | 0/10 | 0% | 0.156 | 22.2 | too hard |
| regex-chess | 0/8 | 0% | 0.526 | 31.1 | too hard, very expensive |

## Proposed replacements for the 8 out-of-band tasks
Proposed in `state/tb21_selected_tasks_v2.txt`, not yet applied.
- **In band, ≤ $0.03 per trajectory:** polyglot-c-py, largest-eigenval, overfull-hbox, db-wal-recovery, dna-insert, mteb-retrieve, adaptive-rejection-sampler, polyglot-rust-c.
- adaptive-rejection-sampler and polyglot-rust-c sit at the 90% edge with only 10 attempts each.
- **Backups:** rstan-to-pystan (80%, $0.05), then schemelike-metacircular-eval (56%, $0.10).

## Kill-switch overshoot
The watchdog fired at $15.24 (13:26:46) and sent SIGINT, but harbor kept running in-flight trials through
the 300 s grace period. It got SIGTERM at 13:31:46, and the final recorded cost was $17.09, $1.85 over the
trigger point instead of the designed ~$0.4. Long, expensive in-flight trials (regex-chess, circuit-fibsqrt)
finished during the grace period.
Fix (backlog): shorter grace period (e.g. 60 s) and terminate harbor's whole process group.

## Selection rule for the final 20 (2026-10-19)
Pool: 36 tasks with measured Luna results (original 20 at ~85 attempts, round-2 candidates at 10).
A task is selected if all of these hold:
1. Raw pass rate is in **[10%, 90%]**.
2. Mean **cost per trajectory ≤ 80th percentile** of the pool ($0.066). At the 70th percentile ($0.036) only 17 tasks qualify.
3. **Length ≤ 90th percentile**: median ≤ 48 agent steps and ≤ 25.2 min. This excludes extremely long trajectories.

Exactly 20 tasks pass. They're the list in `state/tb21_selected_tasks_v2.txt`. Mean cost of the 20 is about $0.018 per
trajectory, so roughly $185 per 10,000.
