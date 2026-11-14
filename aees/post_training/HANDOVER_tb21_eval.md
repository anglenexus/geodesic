# Handover: full Terminal-Bench 2.1 baseline of Qwen3.6-27B with Tmax settings

Written 2026-11-12, 3:40 PM PT, while the run was in progress. The design, its decisions and Phase 1's results are in
`align_tmax_plan.md`. This file covers only what you need to finish the run. Code: PR #32 (branch `aees-tmax-phase1`,
worktree `~/code/geodesic-phase1`).

## Goal

Measure the base model's pass rate on Terminal-Bench 2.1 with Tmax's evaluation settings, comparable to Tmax's
**43.6 ± 1.9** for Qwen3.6-27B (paper Table 3). This is the baseline the post-training experiments will be judged
against.

## What is running

| | |
|---|---|
| Run | `tb21full_1112_1454b` (the `b` run; see "False start" below) |
| Started | 3:25 PM PT, 2026-11-12. Deadline 330 min (the driver is stopped at ~8:55 PM PT at the latest) |
| Trials | 88 tasks × 5 attempts = 440, 40 at a time (20 per box). Extendable to 8 attempts (712 trials) |
| Tasks | `../harbor/state/tb21_tmax88_tasks.txt`: all 89 TB 2.1 tasks except `configure-git-webserver`, which Tmax also excludes (paper §H) |
| GPU host | Nebius 8×H100, `computeinstance-e00dg1ardvv05yr08a`, `willzou1211@89.169.99.22`. 4 SGLang engines × TP2 on ports 30001-30004 |
| Sandbox boxes | 2 × m7a.8xlarge on demand, `3.129.9.10` and `3.137.157.11` (ssh `-i ~/.ssh/aees-ec2.pem ubuntu@<ip>`). Each holds all 89 task images on a 600 GB disk, and they shut themselves down at 07:01 UTC on 2026-11-13 (11:01 PM PT) |
| Dashboard | http://localhost:8771 through `PORT=8771 bash scripts/tunnel.sh willzou1211@89.169.99.22`. Plain-text version: `curl -s localhost:8771/summary.txt` |
| Spend | $28 at 3:35 PM PT; it runs at ~$41.7/h. Expected total ~$150-200 |

The settings are exactly Tmax's evaluation settings:
- thinking on, with `preserve_thinking`;
- temperature 0.7, top_p 0.95;
- 65,536-token context, both served and evaluated;
- 16,384 max tokens per call;
- 64 steps;
- 120 s per command;
- each task's own agent and verifier time limits, multiplier 1.0, no overrides.

When a trial reaches the context limit, the agent stops and the verifier grades it (`HARBOR_TERMINUS_2_STOP_AT_CONTEXT=1`).

`$DATA/monitor/<run>/settings.json` on the host records the settings.

I checked every task individually:
- the task list, the dataset, the time limits, the reference and Tmax's records all agree;
- the boxes' task files are byte-identical to the checked copy;
- the agent-server environment on both boxes is correct.

The per-task table is in the session scratchpad; regenerate it if you need it.

## Processes on the laptop (keep them running)

| Process | What it does |
|---|---|
| `scripts/budget_guard.py --hard 250 --soft 230` | Log: `state/budget_guard.log` (ledger reset for this run; Phase 1's ledger is archived as `state/budget_ledger.phase1_1112_0027.json`). At $230 it stops the driver over ssh, and the finish step then backs up and tears down. At $250 it deletes every Nebius instance and EC2 box. |
| `scripts/teardown_watch.sh tb21full_1112_1454b computeinstance-e00dg1ardvv05yr08a` | Deletes the host once S3 has `BACKUP_OK`. Log: `state/teardown_watch.log`. |
| tunnel to port 8771 | The dashboard. |

## Processes on the host (tmux)

| Session | What it runs |
|---|---|
| `rollout` | `grpo/rollout_only.py`; log `/data/monitor/<run>/rollout.log`; rows in `/data/trials/<run>/rollouts.jsonl` |
| `links` | Tunnels, trial sync and the dispatcher (`curl -s localhost:11000/status`) |
| `p1tracker` | The dashboard server |
| `deadline` | Stops the driver at 330 min |
| `finish` | When the driver ends: backs up to `s3://aees-grpo-backups-611728564241/tb21full_1112_1454b/`, then terminates the boxes and deletes the host |

The host has AWS credentials (copied by the user's request).

## Decided: 5 attempts (user, 5:40 PM PT on 2026-11-12)

At the decision point (372/440 done, 44.6%, $115 spent), the user chose to stay at 5 attempts. Do not create the
`extend` file. The notes below are kept for reference only.

The user chose 5, with the option of extending to 8. To extend, run this before the first 5 attempts have finished:

```
ssh willzou1211@89.169.99.22 touch /data/trials/tb21full_1112_1454b/extend
```

The driver checks every 30 s and then queues attempts 6-8 into the same run. Created after the 5th attempts finish, the file is ignored.
- **Cost:** about +30-45 min and +$25-30 (cap $250 still fits).
- **Benefit:** a ~20% smaller standard error, and per-task rates in steps of 12.5 points.

The best time to ask is when the dispatcher's `waiting` count nears 0, which should be around 5:30 PM PT. A watcher in the
previous session alerts at `waiting ≤ 30`. If you replace it, poll `curl -s localhost:11000/status` on the host.

Without an answer, leave it at 5; the user asked for cost to be kept down.

## What to report at the end

Report times in PT. Give diagnostics, not go/no-go judgements.
1. **Pass rate over the 88 tasks**, with its standard error, against Tmax's 43.6 ± 1.9. Exclude infra errors the
   way `rollout_only.summary` does (via `grpo/infra_filter.infra_cause`), and say how many were excluded.
2. **Pass rate per task**, against the per-task red markers.
   - Those markers come from Tmax's released TB 2.0 runs on Daytona, so their overall pass rate is 39.6%, not 43.6%.
   - 15 tasks changed in TB 2.1 (column `changed_in_2_1` of `../harbor/state/tb21_task_ranking.csv`); treat their markers as approximate.
3. **Timeout rate, context-limit stops (now graded) and 120 s interrupts**, against the reference values in `monitor/tb21_reference.json`.
4. **Steps and agent minutes per trajectory, tok/s, TTFT and ITL.**
5. **Cost, from `state/budget_guard.log`.**
6. **Phase 1 cross-check.** On the 20 training tasks, compare this run's pass rate with Phase 1 (`phase1_1112_0027b`: 46.2%, 22 context-limit stops left ungraded).

Then add a results section to `align_tmax_plan.md`. Add a row to its decision log for any change you make. Commit to PR #32,
staging explicit paths only. Never use `git commit -a`: the main checkout has other agents' uncommitted work.

## Verifying teardown

Teardown is only correct after a verified backup:
1. S3 holds `phase1.tgz` (or the archive `finish.sh` names), and its size has been checked.
2. Then the boxes and the host are gone. Check:
   - `aws ec2 describe-instances --filters Name=tag:Project,Values=aees Name=instance-state-name,Values=running`;
   - `~/.nebius/bin/nebius --profile aees compute instance list` (project `project-e00qzjr2pr00d8qnh3d9xa`).
3. Afterwards:
   - stop the budget guard, the teardown watcher and the tunnel;
   - rename `state/nebius.env` to `nebius.env.deleted-<stamp>`;
   - copy the backup to the laptop, so the dashboard can be served from it as for Phase 1:
     `DATA=<extracted> RUN=<run> python3 monitor/phase1.py --port 8771 --reference monitor/tb21_reference.json`.

If `finish` fails, back up by hand before deleting anything.

## Known issues and history

- **False start (run `tb21full_1112_1454`, no `b`).** `phase1.sh` skipped starting the engines: `status | grep -q` under
  `pipefail` hid the failure. That is fixed in PR #32. The 40 trials dispatched in that window failed on their first
  model call. Their folders are in `aborted_tb21full_1112_1454` on the host and boxes; leave them out of everything.
  The host's SGLang logs are in `/data/monitor/tb21full_1112_1454/`, because the engines were started under that name.
- **`state/run.env` on the laptop** still says `RUN=tb21full_1112_1454`. Only the budget guard's soft stop reads it
  (for the host IP), which is fine.
- **Early pass rates are biased upward**, because short tasks finish first. Longest tasks are dispatched first, so the
  200-min `build-pov-ray` and the 120-min `sam-cell-seg` set the end of the run.
- Our harness is Terminus-2 with Tmax's limits patched in. Tmax used its own agent. That is a known, accepted difference
  (see `align_tmax_plan.md`).

## User preferences that apply

- Never use preemptible GPUs.
- Propose before making multi-step changes, and don't add gates the user didn't ask for.
- Ask before spending more money (raising the cap, reruns).
- Keep the dashboard and the trackers current.
- Every rollout keeps these dashboard statistics. In post-training, they become a click-into from the main training dashboard.
