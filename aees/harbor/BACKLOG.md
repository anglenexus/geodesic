# Backlog: AEES trajectory collection on Harbor

Ordered roughly by when they're needed. See [PLAN.md](PLAN.md) for the design.

## Next
- [x] **Ingest pilot batch into the ledger** (`aees_harbor/ingest.py`, 2026-10-13): batch 1 → `state/ledger.sqlite`, 40 trials held.
- [x] **Trajectory browser** (`python -m aees_harbor.browse`, 2026-10-13): ledger / task / trajectory pages; tool calls derived from raw terminus JSON at view time.
- [ ] **`facts.py`**: extract raw per-trial facts with no judgement: reward, exception type and message head, HTTP status / API error code if present, test-report presence and counts, phase timings, host platform. Tests against `tests/fixtures/trials/`.
- [ ] **Error catalog report**: group pilot trials by (exception type, HTTP status, reward present, test report present) with example lines; review and `resolve` held indices.

## After the first batches
- [ ] Classification rules from the observed catalog (structured signals first: HTTP status, exception class, test-report contents). PLAN §5a.
- [ ] [todo] LLM-judge fallback (GPT-5.6) with an "uncategorized → human review" bucket.
- [ ] `atif_check.py`: trajectory completeness checks (PLAN §6.2).
- [x] Scheduler (`aees_harbor/schedule.py`, 2026-10-13): back-to-back batches, API/disk/Docker preflight, STOP file, deadline, aggregate-cost kill switch (tested), auto-ingest.
- [x] Run-status page (`aees_harbor/status.py`, port 8766).
- [ ] Runner that leases indices from the ledger *before* launching (today indices are assigned at ingest time, in start-time order).
- [ ] Quality suite and size/disk report (PLAN §6.3–6.4).
- [ ] Custom native tool-calling agent (boilerplate) and comparison against terminus-2.

## Deferred / environment
- [ ] EC2: re-test oracle for qemu-startup, qemu-alpine-ssh, tune-mjcf, bn-fit-modify (Rosetta-related failures on Mac).
- [x] EC2 setup scripts + runbook (`ec2/`, untested on EC2): Docker address pools for 100 concurrent trials, image pre-pull, oracle check, 10k schedule.
- [x] Multi-instance support: `scripts/make_ec2_plan.py` (disjoint task shares) + `aees_harbor.merge` (all-or-nothing ledger merge), agent brief `ec2/AGENT_BRIEF.md`.
- [ ] Run EC2 (2 instances x 5,000) and merge the ledgers on the Mac.
- [ ] Swap the 8 tasks outside the 10–90% band after the overnight run (85 attempts each): model-extraction-relu-logits, extract-elf, chess-best-move (too hard); feal-differential-cryptanalysis, fix-code-vulnerability, pytorch-model-cli, custom-memory-heap-crash, kv-store-grpc (too easy).
- [x] `.gitignore` for local env, ledger and run logs.
- [ ] Investigate `RuntimeError: Command timed out after 120 seconds` (harness command timeout, 4.5% of trials at 20-way on the Mac; likely CPU contention).

## Decisions log
- 2026-10-12: no automatic failure classifier until real error data is collected (PLAN §5a).
- 2026-10-13: keep crack-7z-hash (2.1 gave it more resources; the pilot measures its real pass rate).
