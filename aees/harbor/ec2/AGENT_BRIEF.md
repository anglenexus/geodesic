# Agent brief: run the TB 2.1 trajectory collection on this EC2 instance

You are a Claude Code agent running on one EC2 instance. Your job is to run **this instance's
share** of a 10,000-trajectory collection (Terminal-Bench 2.1, `terminus-2` + `openai/gpt-5.6-luna`),
monitor it until it finishes, and hand back the results. All the code you need is in this repo
under `aees/harbor/`. The human runbook is `ec2/EC2.md`; this brief is the exact procedure.

## Your inputs (the human gives you these when starting you)

| Input | Values | Meaning |
|---|---|---|
| `INSTANCE_ID` | `1` or `2` | Which share of the tasks this instance runs |
| `INSTANCES` | `2` (default), or `1` if this is the only instance | Total number of instances |

Tasks are split round-robin, so instance 1 and instance 2 never run the same task. Their ledgers
then merge without conflicts (§9). With `INSTANCES=1`, this instance runs all 20 tasks.

**Before starting you, the human has done:**
1. `~/.config/harbor/.env` contains `OPENAI_API_KEY=sk-...`
2. `docker login` (credentials in `~/.docker/config.json`)
3. Copied the Mac's ledger to `~/geodesic/aees/harbor/state/ledger.sqlite` (the **seed**: 36 scenarios, 1,860 trajectories as of 2026-10-18)

If any of these is missing, **stop and ask the human** (see §2). Don't work around them.

## Hard rules
- **Never** print, log or commit the API key or the contents of `~/.config/harbor/.env`.
- **Never** `git commit` or `git push`. Results go back by copying files (§8), not git.
- **Never** delete or overwrite `state/ledger.sqlite`, any `~/jobs/*` folder, or a plan file that a
  scheduler is using.
- **Never** raise concurrency above 100, raise `budget_usd`, or add batches. Lowering concurrency is
  allowed only by the procedure in §6.
- **Never** kill a running `harbor run` except in an emergency (§7). To stop, use the STOP file.
- **Never** start a second scheduler while one is running (`pgrep -fl aees_harbor.schedule`).
- Only run tasks in this instance's share. The oracle check in §3 covers all tasks; that's fine,
  because it has no API cost.
- If something here doesn't match what you see (a command fails, a file is missing), stop and report
  rather than improvising changes to the code. Fixing an obvious one-line bug that blocks a step is
  allowed, but report exactly what you changed.

## 1. Check the machine
```bash
uname -m                                 # must be x86_64
nproc; free -g | awk '/Mem/{print $2}'   # expect >= 64 cores, >= 192 GB
df -h ~ | tail -1                        # expect >= 200 GB free
```
If it's not x86_64, stop: the task images are amd64-only. If cores, memory or disk are lower,
continue but report it; §6 handles lowering concurrency.

## 2. Setup (skip any step that's already done)
```bash
[ -d ~/geodesic/.git ] || git clone https://github.com/anglenexus/geodesic.git ~/geodesic
cd ~/geodesic && git fetch origin && git checkout aees-harbor-collection   # or main, once the PR is merged
bash ~/geodesic/aees/harbor/ec2/setup_ec2.sh
```
`setup_ec2.sh` adds the user to the `docker` group, which only takes effect at the next login. If
`docker info` then fails with "permission denied", run every Docker-using command in this brief
through `sg docker -c '<command>'`, including the `tmux` launches in §5. A tmux server started that
way passes the group on to its sessions.

Then check the human's prerequisites:
```bash
cd ~/geodesic/aees/harbor
grep -q '^OPENAI_API_KEY=sk-' ~/.config/harbor/.env && echo "key present" || echo "KEY MISSING"
grep -q '"auths"' ~/.docker/config.json 2>/dev/null && echo "docker login present" || echo "DOCKER LOGIN MISSING"
uv run python -c "
from aees_harbor.ledger import Ledger
l = Ledger('state/ledger.sqlite'); s = l.scenarios()
print('seed ledger:', len(s), 'scenarios,', sum(len(l.trial_facts(x['key'])) for x in s), 'trials')"
```
Expect `key present`, `docker login present`, and `seed ledger: 36 scenarios, 1860 trials` (or more,
if the human ran more batches on the Mac since). If the seed is missing, **stop and ask**. Starting without it would restart run indices at 1 for tasks the
Mac already ran, and the ledgers could then never be merged.

## 3. Prepare tasks (~15–40 min)
```bash
cd ~/geodesic/aees/harbor && CONCURRENCY=100 bash ec2/prepare_tasks.sh
```
This checks Docker can create 120 networks, pre-pulls the images, runs the **oracle** on the 20
selected and 4 deferred tasks, and checks the API key.
- Every task in **your share** must score **1.0** on the oracle. If any doesn't, stop and report:
  the task's environment is broken on this host.
- Write down the 4 deferred tasks' results (qemu-startup, qemu-alpine-ssh, tune-mjcf,
  bn-fit-modify) for your final report. Don't add them to the plan.

## 4. Make this instance's plan
```bash
cd ~/geodesic/aees/harbor
CAP=$(( $(nproc) * 3 / 2 )); [ "$CAP" -gt 100 ] && CAP=100     # ~1.5 trials per vCPU, at most 100
python3 scripts/make_ec2_plan.py --instance $INSTANCE_ID --instances $INSTANCES --max-concurrency $CAP
```
The cap scales concurrency to the machine: 100 on 64+ vCPUs, 48 on a 32-vCPU m7i.8xlarge. On the Mac, 10 emulated
cores were saturated at 15–20 trials; native x86 needs less per trial, and the ramp-up check (§6) can lower it
further.
This writes `state/schedule_ec2.json` and `state/tasks_i$INSTANCE_ID.txt`.
- With 2 instances: 10 tasks, 5,000 trajectories, $125 aggregate budget. Batches:
  `i<ID>_ec2_b1_ramp` (250 trials at min(60, cap) concurrent), then 4 batches at min(100, cap).
- With 1 instance: 20 tasks, 10,000 trajectories, $250.

**Dry run on a copy.** A dry run marks batches as done in the file it's given, so never point it
at the real plan:
```bash
cp state/schedule_ec2.json /tmp/dryrun_plan.json
uv run python -m aees_harbor.schedule --plan /tmp/dryrun_plan.json --dry-run
```
Expect "launching … finished" for each batch and "all batches processed".

## 5. Launch
```bash
cd ~/geodesic/aees/harbor
tmux new -d -s sched  'uv run python -m aees_harbor.schedule --plan state/schedule_ec2.json'
tmux new -d -s status 'uv run python -m aees_harbor.status --jobs-root ~/jobs --ledger state/ledger.sqlite --schedule state/schedule_ec2.json'
tmux new -d -s browse 'uv run python -m aees_harbor.browse --ledger state/ledger.sqlite --jobs-root ~/jobs'
sleep 20; tail -3 state/schedule_ec2.log; curl -s localhost:8766/api/status | head -c 300; echo
```
The scheduler log should show `launching i<ID>_ec2_b1_ramp`. The human can watch from a laptop with
`ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 ubuntu@<host>`, then open
http://127.0.0.1:8766.

## 6. Ramp-up check (during the first batch, before the 100-concurrent batches start)
The scheduler starts batch 2 as soon as the ramp batch ends, so **check while the ramp batch is
still running**: once about 60% of it is done, or after 30 minutes, whichever comes first. Read
`http://127.0.0.1:8766/api/status`, and use `docker stats --no-stream` for CPU.

| Signal | OK | Action if not OK |
|---|---|---|
| Docker CPU | < 85% of all cores | Lower concurrency (below) |
| Trials ending with no reward | < 3% | Inspect `exception_info` in `result.json`, then lower concurrency or stop |
| `RuntimeError: Command timed out after 120 seconds` | < 1% | Lower concurrency (this is host contention) |
| Agent timeouts (`AgentTimeoutError`) | ≤ ~10% (Mac baseline 6–8%) | Lower concurrency |
| Cost per trial | ≤ $0.03 (Mac: ~$0.018) | Stop and report |
| Throughput | Record it | Report it. The 2-instance sizing assumed ~1,050 trials/hour at 100 concurrent |

**To lower concurrency:**
1. `touch state/STOP`, so no new batch starts.
2. Wait for the ramp batch to finish and the scheduler to exit (`tail state/schedule_ec2.log`
   shows "stopping before …").
3. Edit `state/schedule_ec2.json`: set `concurrency` (e.g. 80 or 60) for every batch whose
   `status` is `stopped` or `skipped`, and change those statuses back to `planned`.
4. `rm state/STOP` and relaunch the `sched` tmux session.

Report what you saw and what you changed.

## 7. Monitor until done (every 30 minutes)
Use `/loop` or a cron-style recurring prompt. Each check:
1. `pgrep -fl aees_harbor.schedule`, the running harbor job name, and `tail -3 state/schedule_ec2.log`.
2. `curl -s localhost:8766/api/status`: per-batch done/running/queued, raw pass rate, no-reward
   count and its exception types, agent timeouts, throughput, ETA, `aggregate_cost_usd` against
   `budget_usd`, and Docker CPU and memory. Also `df -h ~`.
3. Restart the `status` or `browse` tmux sessions if their ports don't respond.
4. Flag:
   - the scheduler dead while batches are still `planned`
   - no trial finished in more than 20 minutes while trials are running
   - more than 3% of trials with no reward, or any new exception type
   - cost per trial above $0.03
   - disk below 40 GB free

   **If the scheduler died between batches**, relaunch the `sched` session; finished batches are
   skipped. **If it died during a batch**, first change that batch's `status` from `running` to
   `external` in the plan, so the scheduler waits for the running harbor job and ingests it instead
   of launching it again.
5. Append one line to `state/ec2_checks.md`: time | batch | done/total | raw pass rate | aggregate
   cost | notes.

**Emergency stop** (runaway cost, disk about to fill, host unstable):
```bash
pkill -f aees_harbor.schedule; pkill -INT -f "harbor run"; sleep 60; docker ps -q | xargs -r docker kill
```
The budget kill switch already cancels a batch automatically at `budget_usd`: it signals harbor's whole
process group, with a 60 s grace period before SIGTERM/SIGKILL.

## 8. When everything is done: package the results
The scheduler log ends with `all batches processed`, or `stopping before …` if the STOP file,
budget, deadline or a preflight check stopped it. Then:
```bash
cd ~/geodesic/aees/harbor
uv run python -c "
import sqlite3; src = sqlite3.connect('state/ledger.sqlite'); dst = sqlite3.connect('state/ledger_i$INSTANCE_ID.sqlite')
src.backup(dst); dst.close(); print('ledger snapshot written')"
du -sh ~/jobs/i${INSTANCE_ID}_*
```
**Final report.** Put it in `state/ec2_report_i$INSTANCE_ID.md` and reply with the same content:
- each batch: trials, raw pass rate, cost, start and end time, concurrency
- per task in your share: attempts so far (seed + this instance) and raw pass rate, flagging
  any outside 10–90%
- total trajectories, aggregate cost, measured throughput (trials/hour) at each concurrency
- the error catalog: counts of trials with no reward, grouped by exception type, with one example
  message each
- the oracle results for the 4 deferred tasks
- anything unexpected, and any change you made

## 9. Merging (on the Mac, after both instances finish; for the human)
```bash
# copy back from each instance N
rsync -a ubuntu@<host-N>:~/jobs/ ~/code/jobs/
scp ubuntu@<host-N>:~/geodesic/aees/harbor/state/ledger_iN.sqlite ~/code/geodesic/aees/harbor/state/

# merge both into the Mac ledger (all-or-nothing; refuses any conflicting row)
cd ~/code/geodesic/aees/harbor
cp state/ledger.sqlite state/ledger_before_merge.sqlite
uv run python -m aees_harbor.merge --into state/ledger.sqlite state/ledger_i1.sqlite state/ledger_i2.sqlite \
    --rewrite /home/ubuntu/jobs=$HOME/code/jobs
```
Both instances started from the same seed and ran disjoint tasks, so the merge adds each
instance's new rows and recognizes the seed rows as identical. **Don't review (`resolve`) seed
trials on the Mac until the merge is done.** Reviewing changes those rows from `held` to pass/fail,
so they'd no longer match the instances' copies of the seed, and the merge would (correctly)
refuse. The `--rewrite` option points the
trial paths at where the jobs were copied, so the trajectory browser on the Mac can open them.
