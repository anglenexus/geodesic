# EC2 runbook: ~100 concurrent Harbor trials

How to run the Terminal-Bench 2.1 trajectory collection on one x86 EC2 instance at ~100 concurrent
trials, using the same scheduler, ledger, kill switch and web tools that ran overnight on the Mac.

> **Status:** the scripts in this folder have **not yet been run on EC2**. Everything they call
> was exercised on the Mac (Harbor 0.23.0, scheduler, kill switch, ingest, browser, status page),
> but the Linux/EC2-specific parts (Docker install, daemon config, 100-way concurrency) are new.
> Start with the ramp-up batch (`-n 60`) and check the status page before going to 100.

## 1. Instance

| Item | Recommendation | Why |
|---|---|---|
| Architecture | **x86_64** (Intel/AMD) | Task images are amd64-only. Native x86 also unlocks the 4 tasks that fail under Rosetta |
| Size | **64–96 vCPU, ≥ 256 GB RAM** (e.g. m7i.16xlarge / m7i.24xlarge / c7i.24xlarge) | On the Mac, 15–20 emulated trials saturated 10 cores; native x86 needs far less per trial, but CPU is the first limit to watch |
| OS | Ubuntu 24.04 LTS | Scripts use `apt` |
| Disk | **250 GB gp3** | ~40 GB of task images + ~1 MB per trial (10k ≈ 10 GB) + headroom |
| Network | Security group: inbound SSH only | Web tools bind to 127.0.0.1; use an SSH tunnel |

Cost is roughly a few dollars an hour on demand; check current AWS pricing for your region.
API cost is separate: ~$0.019 per trajectory measured on the pilot, so ~$190 per 10,000.

## 2. One-time setup

```bash
ssh ubuntu@<ec2-host>
git clone https://github.com/anglenexus/geodesic.git ~/geodesic
bash ~/geodesic/aees/harbor/ec2/setup_ec2.sh
exit                                   # log out/in so the docker group applies
ssh ubuntu@<ec2-host>
nano ~/.config/harbor/.env             # OPENAI_API_KEY=sk-...
docker login                           # Docker Hub; avoids anonymous pull limits (429)
bash ~/geodesic/aees/harbor/ec2/prepare_tasks.sh
```

`setup_ec2.sh` installs Docker, uv and Harbor 0.23.0, sets up the `aees_harbor` environment and
runs the unit tests. It also applies two settings that matter at 100 concurrent trials:

* **Docker address pools.** Harbor creates **one Docker network per trial**, and Docker's default
  pools hand out /16 subnets, so they run out after ~30 networks. The script sets
  `default-address-pools` to /24 subnets from `10.200.0.0/14` (1,024 networks). If your VPC
  already uses that range, change the base in `/etc/docker/daemon.json`.
* **File and inotify limits** for many containers, plus Docker log rotation.

`prepare_tasks.sh` then:
1. checks Docker can create 120 networks
2. pre-pulls the 24 task images
3. runs the **oracle** on the 20 selected tasks and the 4 deferred tasks (no API cost)
4. checks the OpenAI key

All 20 selected tasks should score 1.0. Record the deferred tasks' x86 results in
`state/tb21_task_notes.md`.

## 3. Optional: continue the Mac's ledger

The Mac ledger (`state/ledger.sqlite`, ~2,000 trajectories after the overnight run) is not in git.
Copying it to EC2 keeps run indices continuous for the same scenarios (same task, model, agent,
settings, Harbor version), so EC2 runs continue at the next index instead of restarting at 1:

```bash
# from the Mac
scp ~/code/geodesic/aees/harbor/state/ledger.sqlite ubuntu@<ec2-host>:~/geodesic/aees/harbor/state/
```

The trial directories it points to stay on the Mac. The browser on EC2 lists those trials but can
only open trajectories whose files are on EC2. To browse everything in one place, rsync
`~/code/jobs/tb21_*` too, or merge ledgers later.

## 4. Run the collection

The plan `ec2/schedule_ec2_10k.json` runs **10,000 trajectories** (20 tasks × 500):

| Batch | Trials | Concurrency |
|---|---|---|
| ec2_b1_ramp | 500 | 60 |
| ec2_b2, ec2_b3, ec2_b4 | 2,500 each | 100 |
| ec2_b5 | 2,000 | 100 |

The aggregate cost kill switch is set to **$250** (expected ~$190), and there's no deadline.

```bash
cd ~/geodesic/aees/harbor
cp ec2/schedule_ec2_10k.json state/schedule_ec2.json      # the scheduler updates this copy in place
tmux new -d -s sched 'uv run python -m aees_harbor.schedule --plan state/schedule_ec2.json'
tmux new -d -s status 'uv run python -m aees_harbor.status --jobs-root ~/jobs --ledger state/ledger.sqlite --schedule state/schedule_ec2.json'
tmux new -d -s browse 'uv run python -m aees_harbor.browse --ledger state/ledger.sqlite --jobs-root ~/jobs'
```

The scheduler runs batches back to back. Before each batch it checks the STOP file, the budget,
free disk, Docker, and the API key. During a batch it re-checks the aggregate cost every 60 s and
cancels the batch at the cap. After each batch it ingests the trials into the ledger, held for
review.

**Watch from your laptop** through an SSH tunnel:
```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 ubuntu@<ec2-host>
# status:       http://127.0.0.1:8766
# trajectories: http://127.0.0.1:8765
```

**During the ramp-up batch,** check on the status page before the 100-way batches start:
* Docker CPU stays below ~85% of the cores
* agent timeouts stay at the Mac baseline (~6–7%)
* the harness error `RuntimeError: Command timed out after 120 seconds` stays rare. It appeared in
  4.5% of trials at 20-way on the saturated Mac and is the clearest sign of host contention.

If any of these look wrong, stop after the ramp batch (below), edit `concurrency` in
`state/schedule_ec2.json` while the scheduler is stopped, and restart it.

**Stop:**
* graceful, after the current batch: `touch ~/geodesic/aees/harbor/state/STOP`
* immediate: `pkill -f aees_harbor.schedule; pkill -INT -f "harbor run"`, then
  `docker ps -q | xargs -r docker kill`

**Restart after a crash:** run the same `tmux new ... aees_harbor.schedule` command. Finished
batches are skipped. A batch left in `running` state should be changed to `external` in the plan
file first, so the scheduler waits for it and ingests it rather than launching it again.

## 5. Getting the data back

```bash
# from the Mac
rsync -a ubuntu@<ec2-host>:~/jobs/ ~/code/jobs/
scp ubuntu@<ec2-host>:~/geodesic/aees/harbor/state/ledger.sqlite ~/code/geodesic/aees/harbor/state/ledger_ec2.sqlite
```

Every trial in the ledger is **held** (unreviewed). Classification rules come from the error
catalog of these runs (PLAN.md §5a), not before.

## 6. Known limits at this scale
* The status page re-reads every trial's `result.json` on each refresh (every 10 s). That's fine
  for thousands of trials; if it gets slow at tens of thousands, raise the refresh interval or cache.
* `docker stats` with 100 containers takes a few seconds; the status page samples it every 15 s in
  the background.
* No automatic classification: infra vs. model failures are reviewed from the error catalog.
