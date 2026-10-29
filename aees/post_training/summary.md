# Summary: 10-hour distributed GRPO run (2026-10-26 to 27)

GRPO on Qwen3.6-27B with LoRA (r=64) over the 20 TB2.1 training tasks (`harbor/state/tb21_selected_tasks.txt`).
The agent is terminus-2 in Harbor sandboxes, and Miles does the training. This file covers what we did, how to
replicate it, and how to continue from the last checkpoint. The detailed records are:

- `dist_setup.md`: machine setup;
- `integration_tests.md`: the tests and their log;
- `integration_issues.md`: every failure, in full;
- `PLAN.md`: the reasons for each decision.

## What we did

1. **Layout.** One machine trains and serves; three machines run the sandboxes.

   ```
   H200 (Nebius, 1 GPU)
     Miles container: Megatron trainer + SGLang + session servers on 127.0.0.1:30000-30015
     grpo/dispatch.py on 127.0.0.1:11000    (Miles posts every trial here)
     monitor/server.py on 127.0.0.1:8770    (tracker)
     scripts/links.sh, one ssh per box:
       -L 1100{1,2,3} -> box:11000              (dispatcher -> agent server)
       -R 30000-30015                           (agent on the box -> session server)
       rrsync pulls the trial folders back
   3 x EC2 m7a.8xlarge (us-east-2, on-demand)
     Docker + Harbor agent server, 20 pre-pulled task images
   ```

   - Training and serving take turns on the GPU (colocate); they don't overlap.
   - The dispatcher starts the longest expected tasks first (`grpo/task_minutes.json`, from the ledger) on the
     least-loaded box.

2. **Before renting machines.**
   - Laptop unit tests, 1256 in total, including the dispatcher, the scripts with stubbed `ssh`/`aws`/`docker`, and
     the configs.
   - One-command tooling: `kickoff.sh`, `ec2_sandboxes.sh`, `ec2_box.sh`, `h200_setup.sh`, `links.sh`,
     `preflight.sh`, `finish.sh`.

3. **Setup.**
   - H200: 11 min. The image was ready at 6.9 min; the torch_dist conversion took 4.3 min more.
   - EC2 boxes: 3.5 min.

4. **Integration tests.**
   - T0 (infrastructure): passed 11/11.
   - T2 (serving): 96 concurrent trials failed. 64 passed: prefix-cache hit 0.66, TTFT p50 9 s / p90 12 s,
     ITL 39–43 ms, 585 output tok/s.
   - T3 (2-step mini GRPO, end to end): passed.

5. **Training recipe.**

   | Setting | Value |
   |---|---|
   | Samples per step | 20 tasks × 8 = 160 trials |
   | Concurrency | 64 (21 per box on 3 boxes; from now on 32 per box on 2, `ec2_sizing.md`) |
   | Gradient steps per step | 1, global batch 160 |
   | Learning rate | 1e-5 |
   | Checkpoint | every step |
   | Verifier cap | 600 s |
   | Agent limits | the tasks' own |
   | Run limit | 10 h |

6. **Teardown.** Automatic, after verified backups:
   - `finish.sh` waited for the trainer to stop and backed everything up to S3;
   - it checked every file's size, found `BACKUP_OK` (32 files, 23.3 GB), terminated the boxes, and deleted the
     H200 at 14:31 UTC.

## Results

| Step | Reward (pass rate) | pg_loss | grad_norm | Rollout | Train | Run / checkpoint |
|---|---|---|---|---|---|---|
| 1 | 0.237 | 0.019 | 0.022 | 61 min | 20 min | `grpo10h_1026_1924/iter_0000000` |
| 2 | 0.275 | −0.004 | 0.026 | 49 min | 20 min | `grpo10h_r2/iter_0000000` |
| 3 | 0.281 | 0.019 | 0.035 | 46 min | 19 min | `grpo10h_r2/iter_0000001` |
| 4 | 0.231 | 0.022 | 0.022 | 51 min | 19 min | `grpo10h_r2/iter_0000002` |
| 5 | 0.125 | −0.003 | 0.021 | 75 min | 21 min | `grpo10h_r2/iter_0000003` |
| 6 | 0.306 | 0.010 | 0.025 | 78 min | 18 min | `grpo10h_r2/iter_0000004` ← **latest** |

- **Training was stable.** The gap between trainer and SGLang log-probs stayed around 0.009, KL to the rollout
  policy around 4e-4, and the clip fraction at 0.
- **Reward is noisy.** It moves with the batch: 7–13 of the 20 tasks give a learning signal per step, and the rest
  score the same in all 8 samples (mostly all 0). Six steps don't show a trend.
- **Base vs trained is not measured yet.** The planned eval job does that (validation 8×4, test 20×4).
- **Time per step was 66–100 min** (train ~20 min including the log-prob pass), and the trials set it, not the GPU:
  - the rollout is 70–80% of a step;
  - the longest trial runs to about 41 min (1800 s agent limit plus 600 s verifier cap).
- **The 10-hour limit gave 6 trained steps.** It started a 7th rollout at 13:55 and stopped it before training.
- **Lost time:** about 70 min to the step 2 OOM and its restarts (see Issues).

## Where everything is

Everything is in **`s3://aees-grpo-backups-611728564241/`** (region us-east-2).

| Prefix | Contents |
|---|---|
| `grpo10h_r2/iter_000000{0..4}/adapter/` | Steps 2–6. `adapter_megatron_rank0.pt` (638 MB, Miles-native LoRA), `training_state_rank0.pt` (3.8 GB, Adam), `adapter_config.json` |
| `grpo10h_r2/trials/grpo10h_r2.tar.gz` | Every trial (trajectories, rewards) and `dispatch.jsonl` |
| `grpo10h_r2/monitor/` | Box, dispatcher and sync logs, `train_run.log`, `host.jsonl`, `preflight.jsonl`, `backup_status` |
| `grpo10h_r2/monitor/dashboard/` | `index.html` and `grpo10h_r2_status.json`: the final tracker state |
| `grpo10h_r2/{grpo.sh,common.sh,pins.sh,train.log}` | Exact launch config and log |
| `grpo10h_1026_1924/` | Step 1: adapter, trials, logs, including the two failed restart logs |
| `t3_1026_1924/` | The T3 run |

Reopening the dashboard needs no machine:

```
aws s3 cp --recursive s3://aees-grpo-backups-611728564241/grpo10h_r2/monitor/dashboard/ dash/
cp dash/grpo10h_r2_status.json dash/status.json
cd dash && python3 -m http.server 8770
```

Then open http://127.0.0.1:8770. Or open `index.html` and choose "Open snapshot" with the JSON.

## Replicate or continue: instructions for an agent

Work in `aees/post_training` on the laptop. Run multi-box shell loops under `bash -c`, because zsh doesn't
word-split. Ask the user for anything marked **(user)**.

### 0. Prerequisites

- **(user)** An H200 VM with NVIDIA driver ≥ 580 (Ubuntu 24.04 image), ≥ 1 TB disk, its public IP and ssh user.
  - Driver 550 needs a 580 install and a console Stop/Start (not a reboot); see `dist_setup.md`, H200 item 1.
- Laptop AWS CLI default profile: IAM user `agent_aees`, key pair `aees-ec2` (`~/.ssh/aees-ec2.pem`).
  - Its policy allows EC2 and S3 on the bucket.
  - Spot is denied by an org policy, so the scripts fall back to on-demand (~$1.85/h per box).
  - **(user)** Checks AWS Budgets headroom: the account is auto-stopped at its spend limit.
- `uv run --python 3.12 pytest tests -q` must pass before kick-off.

### 0b. Create the H200 (Nebius CLI on the laptop)

```
bash scripts/nebius_up.sh        # us-central1 H200, driver-580 image, 1.5 TB; prints the IP and writes state/nebius.env
```

It needs the one-time `nebius profile create aees ...` shown in the script header (service account, key file,
tenant, project).

### 1. Kick-off: about 11 min to "ready to train"

```
mkdir -p state/old && mv state/run.env state/boxes.env state/old/ 2>/dev/null   # else the old run name and dead boxes are reused
bash scripts/kickoff.sh <h200-ip> <user> > state/kickoff.log 2>&1
```

- It moves a previous run's `state/run.env` and `state/boxes.env` aside, creates `state/run.env` (`RUN_T3`, `RUN`, `H200`, `H200_USER`), launches 3 boxes (`state/boxes.env`), copies
  the code, and runs `h200_setup.sh` in tmux `setup`.
- It then lets the H200 into the boxes, writes `~/aees_box_ids` there, starts the agent servers, and starts
  `links` and `tracker`.
- Wait for `SETUP_DONE` in `~/setup.log` on the H200.

### 2. Integration tests: not re-run (passed 2026-10-27; kept for reference)

Run on the H200 in `~/geodesic/aees/post_training`, with values from `state/run.env`:

```
RUN=$RUN_T3 bash scripts/preflight.sh t0           # laptop meanwhile: bash scripts/ec2_sandboxes.sh probe <h200-ip>
RUN=$RUN_T3 T2_AGENTS=64 bash scripts/preflight.sh t2   # after SETUP_DONE; optional on a known setup
RUN=$RUN_T3 bash scripts/preflight.sh t3
```

- Results show on the tracker's Preflight card.
- Don't train on a red test; log failures in `integration_issues.md`.

### 3. AWS and Nebius credentials on the H200, needed for the automatic teardown

- **(user)** `aws configure` on the H200, with the `agent_aees` key, region us-east-2.
- **(user)** `nebius profile create` on the H200:
  - endpoint `api.nebius.cloud`, service-account authorization;
  - service account **ID** `serviceaccount-u00m8xacn35xbs65hz` (the ID, not the name), in the editors group;
  - its public-key ID and private key (`~/code/testing/private.pem` on the laptop, scp it over);
  - the tenant and project IDs (previous: `tenant-e00m27dyn90tsdsry0`, `project-u02n20nqig001ntg5jdh7b`).
- Get the instance ID with `nebius compute instance list`.
- Give the H200 the box IDs:

  ```
  bash -c 'source state/boxes.env; echo $BOX_IDS' | ssh <user>@<ip> 'cat > ~/aees_box_ids'
  ```

### 4. Starting point: fresh, or the latest checkpoint

**To continue,** download step 6's adapter on the H200. Pick a new run name; the training section below uses
`NEW=grpo_cont_<MMDD>`.

```
aws s3 cp --recursive s3://aees-grpo-backups-611728564241/grpo10h_r2/iter_0000004/adapter/ \
  /data/ckpt/grpo10h_r2/iter_0000004/adapter/
```

- Continue with **`INIT_ADAPTER`**: step 6's LoRA weights and a fresh Adam. This is the path that has worked on
  the GPU.
- Miles counts rollouts from 0 again, so new checkpoints start at `iter_0000000` in the new run folder. Step 7
  overall is the new run's step 1.
- **Don't use `RESUME_FROM` yet.** It restores the adapter and Adam, but Miles' `RolloutExecutor.load` then looks
  for rollout state under `--load` (the base torch_dist folder) and asserts. It needs a fix first; see Issues.
- The cost of a fresh Adam is one step of moment history.

**To replicate from scratch,** leave out `INIT_ADAPTER`.

### 5. Training: the exact launch used

On the laptop, restart the agent servers in normal mode:

```
bash -c 'bash scripts/ec2_sandboxes.sh serve <RUN>'
```

This applies `HARBOR_VERIFIER_TIMEOUT_SEC=600` and 32 slots per server.

Then on the H200, in `~/geodesic/aees/post_training`. One command does all of it:

```
RUN=<RUN> STEPS=4 INIT_ADAPTER=/root/ckpt/grpo10h_r2/iter_0000004/adapter \
  NEBIUS_INSTANCE_ID=<computeinstance-...> DEADLINE_H=8 bash scripts/launch.sh
```

What `launch.sh` runs, for reference. If the code changed after kick-off, rsync the whole tree from the laptop and
compare checksums first; a partial copy cost a restart.

```
R=<RUN>; B=s3://aees-grpo-backups-611728564241; mkdir -p /data/monitor/$R
sudo docker ps -q | xargs -r sudo docker rm -f
RUN=$R SLOTS=32 bash scripts/links.sh start                       # 2 boxes x 32 = the tested 64
tmux new -d -s tracker "RUN=$R NUM_ROLLOUT=8 python3 monitor/server.py"
tmux new -d -s train "SAVE_DIR=/root/ckpt/$R NUM_ROLLOUT=20 SAVE_INTERVAL=1 DATA=/data \
  INIT_ADAPTER=/root/ckpt/grpo10h_r2/iter_0000004/adapter \
  bash scripts/run.sh grpo > /data/monitor/$R/train_run.log 2>&1"            # leave out INIT_ADAPTER for a fresh run
tmux new -d -s deadline "sleep 36000; tmux kill-session -t train; sudo docker ps -q | xargs -r sudo docker rm -f; \
  echo STOPPED_AT_10H >> /data/monitor/$R/train_run.log"                    # 10-hour limit
tmux new -d -s hourly "while tmux has-session -t train 2>/dev/null; do sleep 3600; sudo chown -R \$USER /data/ckpt/$R \
  && WITH_OPTIMIZER=1 bash scripts/backup.sh /data/ckpt/$R $B 2>&1 | tail -1 >> /data/monitor/hourly_backup.log; done"
tmux new -d -s finish "RUN=$R DEST=$B WAIT_FOR=train NEBIUS_INSTANCE_ID=<computeinstance-...> \
  bash scripts/finish.sh > /data/monitor/finish.log 2>&1"                   # backup -> check -> terminate boxes -> delete H200
```

- `NUM_ROLLOUT=20` is a ceiling; the deadline stops the run.
- Check that the adapter loaded: `train_run.log` should show `Loaded 512 adapter tensors` and a non-zero
  `max|lora_B|`.
- `PYTORCH_CUDA_ALLOC_CONF` is already set in `scripts/run.sh`, which prevents the step 2 OOM.
- Tracker on the laptop: `bash scripts/tunnel.sh <user>@<ip>`, then http://localhost:8770.
- To continue the dashboard of an earlier run on a new machine, copy its final snapshot
  (S3 `<run>/monitor/dashboard/<run>_status.json`) to the H200 and pass `PRIOR_SNAPSHOT=<path>` to `launch.sh`;
  its trained steps come first. `PRIOR_RUN=<run>` does the same from a run folder on the same disk.
- kickoff.sh opens the dashboard on the laptop (background tunnel, http://127.0.0.1:8770/).
- **If something crashes, don't restart in a hurry.** `finish` would back up and delete the H200 once `train` ends.
  To restart, `tmux kill-session -t finish` first, then re-arm it with the new run name.

## Issues and fixes

### Found on the laptop, before renting (no GPU time lost)

| Problem | Fix |
|---|---|
| The dispatcher dropped connections at 12+ trials at once (listen backlog 5; Miles opens 160) | `request_queue_size = 1024`, tested with 160 at once |
| T3 at 2 samples per task: the infra filter masks groups under 4 valid trials and would stop the run | T3 uses 2 tasks × 4 samples; a test checks it |
| Docker `daemon.json` (address pools) not applied on EC2 | `setup.sh` step 3 restarts Docker |

### Setup and tests

| Problem | Fix |
|---|---|
| EC2 spot denied by an org service control policy | Automatic on-demand fallback |
| The agent server refuses to start without a Docker Hub login | Anonymous `~/.docker/config.json` entry; images are pre-pulled, so nothing is pulled |
| `serve` reported success when the servers had failed | Waits on each box's exit code |
| The agent server writes its request log under `/data` (no permission) | `DASHBOARD_LOG_PATH` under the run's monitor folder |
| tmux sessions couldn't reach Docker (started before the docker group was added) | `tmux kill-server`, restart sessions after setup |
| Root-owned folders from `docker -v` (`/data/proto`, adapters with mode 600) | `h200_setup.sh` creates the folders first; `sudo` for copies and removal |
| T0's stub on port 30000 passed T2's health check | T2 stops the stub first |
| **96 concurrent trials are too many for SGLang** (BF16, LoRA, 56k context: ~40 running requests, 5 DeltaNet state slots each; cache hit 0.36, TTFT p90 39 s) | Concurrency 64 (cache hit 0.66, TTFT p90 12 s) |
| **Miles handed agents the VM's private address** for its session servers | `--session-server-ip 0.0.0.0 --session-server-external-host 127.0.0.1` |
| **Miles' ports answered from the internet** | iptables: drop new inbound TCP on the public interface except 22 |
| An old agent server survived its tmux session, so the restart hung | `fuser -k` on port 11000 |

### Training

| Problem | Fix |
|---|---|
| **Straggler:** mailman's verifier hung to its 30-min limit, holding step 1 ~12 min after the other 159 trials | `HARBOR_VERIFIER_TIMEOUT_SEC=600`: no passing trial in the ledger needed > 4.5 min. The 1800 s agent limits still set a ~41-min floor (changing them is opt-in) |
| Dashboard showed no steps: Miles logs `rollout_id` at the top level of each event | Tracker reads it |
| **Step 2 CUDA OOM** in the log-prob pass: a 22.7 GiB allocation with 44 GiB reserved but fragmented | `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`, as in Miles' own Harbor LoRA recipe (`expandable_segments` breaks torch_memory_saver). No OOM in the next 5 steps |
| **Restart 1:** `RESUME_FROM` ignored because only `run.sh` was copied, not the configs | Rsync the whole tree and compare checksums before any restart |
| **Restart 2:** `RESUME_FROM` restored the adapter and Adam, then `RolloutExecutor.load` asserted (it reads rollout state from `--load`) | Continued as run `grpo10h_r2` with `INIT_ADAPTER` (weights only). **Open:** fix the rollout-state path for `RESUME_FROM` |
| The dashboard showed the aborted attempts as extra rollouts | Tracker `PRIOR_RUN`: earlier run's trained steps first, aborted trials left out |
| The teardown could delete the H200 after a crash | Pause `finish` before any restart, re-arm after |

## Open items

- **Eval job** (separate session): base vs trained on validation (8 tasks × 4) and test (20 × 4), from the S3
  adapters. The adapters are Miles-native `.pt` files. Serving them needs a conversion to HF/PEFT format, or Miles'
  own eval path (`--eval-prompt-data` / `--debug-rollout-only`).
- **SFT** on the collected passes (in the trial archives): deferred.
- **`RESUME_FROM`:** fix the rollout-side restore, so runs can continue with Adam and step count.
- **Shorter steps:** the 1800 s agent limits on 4 tasks set the floor; capping them for training would be opt-in.
