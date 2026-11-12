# Backlog: AEES trajectory collection on Harbor

Ordered roughly by when they're needed. See [PLAN.md](PLAN.md) for the design.

## Standing requirement: rollout statistics for every rollout (user request, 2026-11-12)
Every rollout (data collection, baselines, evals, and each GRPO rollout step) must keep producing the statistics the
run tracker shows today, and stay browsable. In post-training this becomes a **click-into** from the main training
dashboard (`../post_training/monitor`, port 8770): click a step, eval or baseline → its rollout page.
- **Per rollout:** done / total, running / queued; pass / fail / no reward; raw pass rate; agent timeouts; exception
  breakdown (type + first message line); throughput and projected finish; API cost or GPU time; per-task outcome grid
  in start order; slowest trials; turns and tokens per trial (incl. reasoning tokens where the model reports them).
- **Every trial** links to its trajectory in the trajectory browser (port 8765), via the ledger (tagged with run + step).
- [ ] Reuse `aees_harbor.status.job_status` (and the job card in `browser/status.html`) on a rollout step's trial dirs,
  so the post-training tracker can link to `/rollout/<run>/<step>` with the same view.
- [ ] Record reasoning tokens and the `reasoning_effort` setting per trial (the GPT-5.6 Luna runs sent none, so the
  model's default applied; see PLAN decisions log).

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
- [x] EC2 runs merged on the Mac (2026-10-24: i3 resume, 2,618 trials; 2026-10-29: 4 x m7i.8xlarge, 2,000 trials).
- [ ] Fix the recurring tmux "send-keys" harness error (sqlite-with-gcov, mailman, configure-git-webserver).

- [x] `.gitignore` for local env, ledger and run logs.
- [x] Kill-switch overshoot fixed (2026-10-19): harbor runs in its own process group; cancel signals the whole group, 60 s grace, then SIGTERM/SIGKILL. Tested with a fake harbor that ignores SIGINT.
- [x] Task swap applied (2026-10-19): `state/tb21_selected_tasks.txt` = final 20 (see task notes).
- [ ] Investigate `RuntimeError: Command timed out after 120 seconds` (harness command timeout, 4.5% of trials at 20-way on the Mac; likely CPU contention).

## Decisions log
- 2026-10-12: no automatic failure classifier until real error data is collected (PLAN §5a).
- 2026-10-13: keep crack-7z-hash (2.1 gave it more resources; the pilot measures its real pass rate).
