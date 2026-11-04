# Integration issues

Every failure seen while setting up or testing the distributed run (`dist_setup.md`, `integration_tests.md`): symptom,
cause, fix, and time lost. **Fix budget: 60 min of H200 time in total.** If a test is still red after that, stop,
write the state here, and decide with the user. Don't start the 10-hour run on a red test.

Once a fix is confirmed, move what works into `dist_setup.md`.

## Read first: open reliability issues of the Harbor EC2 sandboxes (details in the dated entries below)

1. **Cancelled trials leak their containers** on the boxes (agent server kills the worker before `compose down`).
   Worked around: `scripts/ec2_sandboxes.sh reap <run> 120` from the laptop for the whole run. Proper fix is in
   harbor-miles (register the compose project before the worker starts). Entry: 2026-11-03, containers.
2. **terminus-2 installs tmux + asciinema from the internet in every trial** (p50 25 s, limit 120 s); over the limit
   the trial aborts with no model call. Fixed: `setup.sh` step 8 runs `scripts/bake_tools.py` (tools baked into the
   task images; setup 1.7 s). Check a new box's `prepare` log says "0 failed". Entry: 2026-11-03, agent setup.
3. **One aborted trial makes Miles re-run a whole batch (20 tasks x 8 = 160 trials) to replace one group of 8**, and
   the replacement is a different task. Miles requires `--over-sampling-batch-size >= --rollout-batch-size`, so a
   flag can't fix it. Fixed: `grpo/resubmit_group.patch` + `--resubmit-dropped-groups 3` reruns the dropped task
   only. Entry: 2026-11-03, refill.
4. **The boxes shut themselves down `TTL_MIN` after launch (default 900 = 15 h), mid-run if the run outlives them**;
   every running trial then aborts and Miles stops after the reruns. Fixed: `launch.sh` runs `scripts/box_check.sh`
   after the tunnels are up and starts nothing unless every box answers through its tunnel and shuts down at least
   (steps left x STEP_MIN + MARGIN_MIN) minutes later. Before a resume, also compare the boxes' launch time + TTL
   with the projected finish. Entry: 2026-11-03, box shutdown.

## Format

```
### <date> <test or step>: <one-line symptom>
- Seen: exact error / log line
- Cause: (verified, or "suspected")
- Fix: what changed (commit, flag, command)
- Time lost: N min    Status: fixed / open / worked around
```

## Known from the 2026-10-25 prototype (already fixed; kept so they're recognized fast)

### 2026-10-25 H200 driver: 550 on the Nebius image, 580 needed
- Seen: `driver 550.163.01 < 580`; after installing 580 and a warm reboot, dmesg `WPR2 already up`; after a PCI
  reset, CUDA error 802 and Fabric State "In Progress".
- Fix: install 580 and get a **console Stop/Start** from the user (`dist_setup.md`, H200 item 1).
- Time lost: ~45 min. Status: fixed; avoid by asking for a 580 image.

### 2026-10-25 EC2 rsync: remote parent folder missing
- Seen: `rsync: mkdir "/home/ubuntu/geodesic/aees/harbor" failed`.
- Fix: `mkdir -p geodesic/aees proto` before rsync (proto/ec2_up.sh, a75b971).
- Time lost: 2 min. Status: fixed.

### 2026-10-25 `ModuleNotFoundError: proto.train_proxy` in the Miles image
- Cause: the image's pip package `proto` (proto-plus) shadowed our folder.
- Fix: `proto/__init__.py` (06bc99c).
- Time lost: 5 min. Status: fixed.

### 2026-10-25 DCGM container exits 255
- Cause: the host already runs nv-hostengine.
- Fix: use the host `dcgmi`.
- Status: worked around; the bench.sh change isn't committed yet.

### 2026-10-25 own ssh session killed
- Cause: `pkill -f "dcgmi dmon"` matched the ssh command line that ran it.
- Fix: never pkill patterns that appear in your own command.
- Status: rule in `dist_setup.md`.

## This run

Found on the laptop by the L tests and by review, before any machine was rented (no H200 time lost):

### 2026-10-26 L: dispatcher dropped connections with 12+ trials at once
- Seen: `URLError: [Errno 22] Invalid argument`, `Broken pipe`; 11 of 12 replies.
- Cause: `ThreadingHTTPServer`'s listen backlog defaults to 5, and Miles opens all 160 trial connections at once.
- Fix: `grpo/dispatch.py` `Server.request_queue_size = 1024`; test with 160 at once.
- Status: fixed.

### 2026-10-26 L: T3 at 2 tasks × 2 samples would have stopped the run
- Cause: `grpo/infra_filter.py` masks groups with fewer than MIN_VALID = 4 valid attempts. More than 10% of a step
  masked raises "outage?".
- Fix: T3 runs 2 × 4; a test checks N_SAMPLES ≥ MIN_VALID.
- Status: fixed.

### 2026-10-26 L: Docker config not applied on EC2
- Cause: `setup.sh` step 3 wrote `daemon.json` without restarting Docker. Only step 4 (container toolkit, H200
  only) restarted it, so the address pools (needed beyond ~30 trial networks) would be missing on the boxes.
- Fix: step 3 restarts Docker.
- Status: fixed.

### 2026-10-26 L: probe rows written as one line
- Cause: `$(printf ...)` strips the trailing newline.
- Fix: newline appended outside the substitution; test parses each row.
- Status: fixed.

### 2026-10-27 setup: EC2 spot denied by an organization policy
- Seen: `UnauthorizedOperation ... ec2:RunInstances ... with an explicit deny in a service control policy`.
- Fix: none needed. `ec2_sandboxes.sh` fell back to on-demand as designed (3 × m7a.8xlarge, ~$1.85/h each). The
  "launched ... spot" line was misleading; it now says on-demand.
- Time lost: 0. Status: worked around (cost ~$5.6/h for 3 boxes instead of ~$1.8/h).

### 2026-10-27 R/serve: agent server refuses to start without a Docker Hub login
- Seen: `ERROR: Docker is not logged into Docker Hub (no 'https://index.docker.io/v1/' entry in ~/.docker/config.json)`
  (harbor-miles `_check_docker_login`; the only skip is HARBOR_ENV_TYPE=daytona). Found by the rehearsal.
- Cause: a guard against anonymous pull limits. Trials never pull: the images are pre-pulled, the prebuilt compose has no
  pull policy, and `down --rmi local` keeps pulled images.
- Fix: `ec2_box.sh prepare` writes an anonymous entry `{"auths": {"https://index.docker.io/v1/": {}}}`.
- Time lost: ~5 min. Status: fixed.

### 2026-10-27 serve: `ec2_sandboxes.sh serve` said "up" when the servers had failed
- Fix: wait on each box's exit code. Status: fixed.

### 2026-10-27 serve: agent server writes its request log under /data
- Seen: `PermissionError: '/data'` (RequestLogger default path).
- Fix: `agent_server.sh` sets `DASHBOARD_LOG_PATH=$DATA/monitor/$RUN/agent_requests.jsonl`. Status: fixed.

### 2026-10-27 laptop: zsh doesn't word-split `$BOX_IPS`, and `$ip:` is a modifier
- Fix: run multi-box loops under `bash -c`. Status: rule in dist_setup.md.

### 2026-10-27 T0/T2/T3: tmux sessions can't reach Docker
- Seen: `permission denied while trying to connect to the docker API`.
- Cause: the tmux server started before setup.sh added the user to the docker group; new sessions inherit its groups.
- Fix: `tmux kill-server`, then restart links and the tracker from a fresh login. h200_setup.sh should run before any
  long-lived tmux. Time lost: ~10 min. Status: fixed.

### 2026-10-27 T2: /data/proto owned by root
- Cause: `docker run -v $DATA/proto:...` creates a missing host folder as root.
- Fix: h200_setup.sh creates `proto`, `ckpt` and `trials` first. Status: fixed.

### 2026-10-27 T2: T0's stub on port 30000 passed T2's health check
- Fix: T2 stops the stub and checks that its container is alive before polling. Status: fixed.

### 2026-10-27 T2: 96 trials exceed SGLang's capacity (BF16, LoRA, 56k context, mem 0.85)
- Seen: `max_running_requests=39` (state slots bind), `max_total_num_tokens=520391`. Under 96 agents: prefix-cache hit
  0.355, TTFT p90 39 s, 42 ms/token p50, headroom 16 GB.
- Cause: as predicted in integration_tests.md: ~5 DeltaNet state slots (~150 MB each, fp32) per running request.
- Next: T2 at 64 agents to pick the concurrency. `--mamba-ssm-dtype bfloat16` would double the slots but changes the
  sampler's numerics vs the trainer (opt-in only).
- Status: open.

### 2026-10-27 T3: agents were handed the VM's private address
- Seen: `Session servers ready at ['10.96.0.14:30000', ...]`, which the boxes can't reach.
- Fix: `--session-server-ip 0.0.0.0 --session-server-external-host 127.0.0.1` (configs/grpo.sh).
- Time lost: ~8 min (one restart). Status: fixed.

### 2026-10-27 security: Miles' ports reachable from the internet
- Seen: from the laptop, `204.12.188.130:30000` and `:31000` answered (HTTP 404).
- Fix: `iptables -I INPUT -i eth0 -p tcp ! --dport 22 -m conntrack --ctstate NEW -j DROP`, now in h200_setup.sh.
  Verified: blocked from outside; local access, ssh and tunnels fine. Status: fixed.

### 2026-10-27 T3: "no infra errors" check printed 0 twice
- Cause: `grep -c ... || echo 0` (grep exits 1 on zero matches). Fix: count via `cat | grep -c`. Status: fixed.

### 2026-10-27 T2 rerun: adapter copy incomplete
- Cause: the adapter written by the container is root-owned with mode 600, so `cp` as the user failed silently.
- Fix: reuse the original T2 folder. Status: fixed.

### 2026-10-27 training step 1: one trial held the rollout ~12 min after all others
- Seen: 159/160 trials done by minute 48. `mailman` ran to minute ~60: agent stopped at its 1800 s limit (05:02:35),
  then the verifier wrote output at 05:03:24 and hung until its own 1800 s limit (`[verifier] timeout_sec` in
  task.toml).
- Cause: agent and verifier limits add up; the worst case per trial is ~30 + 30 min plus setup.
- Data (ledger, 6,394 trials on the 20 training tasks): verifier p99 3.5 min; **no passing trial needed over
  4.5 min**; mailman's verifier p99 = 30.0 min (hangs on failures), crack-7z-hash max 15, large-scale-text-editing
  max 10.7.
- Fix: training trials get `HARBOR_VERIFIER_TIMEOUT_SEC=600` (ec2_box.sh serve; `VERIFIER_TIMEOUT_SEC` overrides).
  It can't turn a recorded pass into a fail. Applied during step 1's training phase (no trials in flight); verified
  in each server's environment. Worst case per trial is now ~41 min.
- Left: the 1800 s agent limits (mailman, crack-7z-hash, dna-insert, large-scale-text-editing) still set the floor.
  A shorter agent cap for training would change what is learned (opt-in).
- Status: fixed for the verifier.

### 2026-10-27 training step 2: CUDA out of memory in the old-log-prob pass
- Seen (06:41 UTC): `torch.OutOfMemoryError: Tried to allocate 22.73 GiB ... 10.21 GiB is free ... 79.43 GiB is
  allocated by PyTorch, and 44.03 GiB is reserved by PyTorch but unallocated`, in `_compute_log_prob` →
  `forward_only` (the bf16 logits of one ~48k-token micro-batch). Step 1 had passed with the same settings.
- Cause: allocator fragmentation from variable-length agent micro-batches. Miles' own Harbor LoRA recipe
  (examples/swe-agent-harbor-docker/run_glm52_lora_tb2_daytona.py) documents the same failure and sets
  `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512` (expandable_segments breaks
  torch_memory_saver under colocate).
- Fix: that setting in scripts/run.sh's Ray runtime env. Resumed at 06:43:51 with
  `RESUME_FROM=/root/ckpt/grpo10h_1026_1924/iter_0000000/adapter` (first GPU use of RESUME_FROM). The teardown job
  was paused at 06:42 (it would have deleted the H200 after the crash) and re-armed after the restart.
- Time lost: step 2's 43-min rollout + ~7 min restart. Status: fix applied, being verified.

### 2026-10-27 resume after the OOM: two failed restarts before a clean one
- Attempt 2 (06:43): `RESUME_FROM` was ignored. Only scripts/run.sh had been copied to the H200, not the updated
  configs/common.sh and configs/grpo.sh, so it trained from scratch (`lora_adapter_path None`,
  `max|lora_B| = 0`). Caught 15 min in, before it could overwrite step 1's checkpoint. Fix: rsync the whole tree
  and compare checksums before any (re)start.
- Attempt 3 (07:00): with the configs, Miles restored the adapter and Adam ("Restored optimizer state from LoRA
  checkpoint"). Then `RolloutExecutor.load` asserted: it looks for rollout-side state under `--load` (the base
  torch_dist folder), but saves it under `--save`. So for native LoRA, RESUME_FROM with a step count can't
  restore the rollout side as configured.
- Attempt 4 (07:08, run `grpo10h_r2`): `INIT_ADAPTER` (step 1's adapter weights, fresh Adam). Miles starts at
  rollout 0 and loads no rollout state (placement_group.py loads it only when start_rollout_id > 0). Verified:
  `Loaded 512 adapter tensors`, served weights `max|lora_B| = 1.001e-05` (step 1's), rollout running.
- Cost: step 2's rollout run ~1.5 times, ~70 min in all. One Adam step of moment history lost (negligible).
- Dashboard: aborted attempts appeared as extra rollouts. The tracker now takes `PRIOR_RUN`: the earlier run's
  trained steps first, this run's steps after them, aborted trials left out.
- RESUME_FROM status: adapter + optimizer restore works; the rollout-side restore needs the state under --load.
  Until fixed, continue runs with INIT_ADAPTER.

## Continuation run (2026-10-27, H200 in us-central1)

Setup took 11.2 min, and T0 passed 11/11. Integration tests aren't re-run from here on (user decision): the fixes
below are in the scripts, and a new failure becomes a new entry.

### 2026-10-27 H200 creation: no CLI path, wrong project, cloud-init rejected
- Seen: the laptop had no `nebius` CLI. The profile's default project (us-north1) has no H200 platform.
  `--cloud-init-user-data: YAML is invalid: line 9`.
- Cause: H200s exist only in the us-central1 project. `~/.ssh/id_rsa.pub` holds two keys, and the second line
  broke the YAML list.
- Fix: `scripts/nebius_up.sh`: the us-central1 project and subnet, image family `ubuntu24.04-cuda13.0` (driver 580),
  the first key only. It writes `state/nebius.env`. Status: fixed.

### 2026-10-27 kickoff: tmux sessions without the docker group (again)
- Cause: kickoff's `tmux new -s setup` started the tmux server before setup step 2 added the user to the docker
  group, so every later session (links, tracker, tests, train) inherited the old groups.
- Fix: `h200_setup.sh` has `PHASE=pre` (folders, firewall, steps 1-4, ~35 s). kickoff runs it over plain ssh
  before any tmux, then `PHASE=main` in tmux. Status: fixed.

### 2026-10-27 kickoff reused the previous run's state
- Cause: `state/run.env` (old run names) and `state/boxes.env` (terminated boxes) were still there.
- Fix: kickoff moves them to `state/old_<stamp>/` when the H200 IP differs or none of the boxes is running
  (test added). Status: fixed.

### 2026-10-27 probe: rows sent to `ubuntu@<h200>`
- Fix: `ec2_sandboxes.sh probe` takes `H200_USER` from `state/run.env`. Status: fixed.

### 2026-10-27 teardown prerequisites missing on a fresh H200
- Seen: the Nebius image ships `nebius` but not `aws`. `~/aees_box_ids` was written by hand.
- Fix: `h200_setup.sh` installs the AWS CLI (snap), and kickoff writes `~/aees_box_ids`. `aws configure` and
  `nebius profile create` stay manual (user credentials); `launch.sh` warns when either is missing.
  Status: fixed.

### 2026-10-27 T3: two checks failed on its 15-min cap
- Seen: "2 steps trained and saved: 15 min; iter_0000000", "trials synced: 11".
- Cause: a cold SGLang JIT cache on the new disk (4 min start vs ~1.5 min). Step 2's rollout had finished, served
  by the updated weights; the sync caught up to 16/16 a minute later. Not a fault.
- Status: no change (T3 isn't re-run).

### 2026-10-27 noise, not errors
- `ConnectionRefusedError ... :20001` tracebacks and `post-warmup freeze_gc failed`: SGLang's own warm-up poll
  before uvicorn listens. Harmless.
- `tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr...'`: macOS metadata. kickoff now uses
  `COPYFILE_DISABLE=1 tar --no-xattrs`.

### 2026-10-27 manual training launch
- The training launch was a hand-typed chain of six tmux sessions; one partial copy had already cost a restart.
- Fix: `scripts/launch.sh` (H200) starts links, tracker, train, deadline, hourly and finish. It refuses to start
  over a running train and checks that the adapter exists. kickoff now serves the agent servers in normal mode.

## Auxiliary-reward run auxr_1103_0033 (2026-11-03, H200 + 2 x m7a.8xlarge, 32 trials each)

### 2026-11-03 auxr_1103_0033: cancelled trials leave their containers running
- Seen: after steps 4 and 7 (each sent a second batch and cancelled the rest), 64 and 62 trial containers on the two
  boxes from long-cancelled trials, next to the 32 live ones; one cancelled `password-recovery` kept a core busy for an hour.
- Cause (verified in harbor-miles on the boxes): on `/flush` the agent server cancels the `/run` task and kills its
  worker process (`agent_server/trial_runner.py` ~461-467) before the worker's `compose down`. The compose project is
  only registered from the worker's reply, so `/flush_all` (`agent_server/teardown.py` `_kill_trial_containers`) can't
  find it either. Vanilla ran the same code and also cancelled batches (steps 5, 6).
- Fix: `scripts/ec2_sandboxes.sh reap <run> [secs]` runs `compose down --volumes` for every trial project created
  before Miles' latest `rollout K:` log line (all of that rollout's trials have ended or been cancelled by then); run
  every 120 s from the laptop during the run (`state/reap.log`). First pass removed 65 + 63 projects; live trials untouched.
  Proper fix (harbor-miles): register the compose project before the worker starts, or tear down on cancel.
  A stop + resume also leaves the stopped attempt's trials running (61 containers after the step-9 resume, which
  wrote no 'rollout K:' line), so the cutoff is the later of that line and Miles' last start line.
- Time lost: none (CPU only).    Status: worked around

### 2026-11-03 auxr_1103_0033: agent setup over 120 s (tmux install) aborts a trial, its group, and costs a refill
- Seen: step 4 `sanitize-git-repo` "RuntimeError: Command timed out after 120 seconds" (agent setup 120.4 s);
  step 7 `dna-insert` "Failed to start tmux session. Error: None" (agent setup 128 s). No model call, so Miles marks the
  trial ABORTED and `check_no_aborted` drops the whole group, then sends a second 160-trial batch to replace it.
- Cause: terminus-2 installs tmux (and asciinema) into every task container with apt-get from the internet, with a
  120 s limit (`tmux_session.py` `_TOOL_INSTALL_TIMEOUT_SEC`). Over 1,321 trials: p50 25 s, p90 30 s, p99 83 s,
  2 over 120 s. Slow setups come in bursts: 13:26-13:31 (4 trials at 106-120 s with only 2-14 setups starting at
  once, so the mirror or network was slow) and the 64-at-once start waves (17:07 one at 128 s; 17:48 all ~80 s).
  mteb-retrieve's image already has the tools (1.6 s). The infra filter (`grpo/infra_filter.py`) would mask just the
  attempt, but Miles' group filter runs first and these are RuntimeErrors, not AgentSetupTimeoutError.
- Cost: each drop's refill added ~30 min to the rollout (step 4: 81 min, step 7: 80 min, vs ~50).
- Fix (next run; the live run keeps its images): `scripts/bake_tools.py` rebuilds each task image under the same name
  with tmux + asciinema (one layer, original USER kept); `setup.sh` step 8 runs it after the pre-pull, so every
  `ec2_box.sh prepare` gets it. terminus-2 then skips its install (`tmux -V` and `asciinema --version` pass).
- Tested: rehearsal (`TASKS=<the 20> bash scripts/rehearse.sh`, fresh box): 19 baked, mteb-retrieve already had them,
  0 failed; one real trial per task, agent setup 1.7-1.8 s for all 20 (was p50 25 s, max 128 s), no AgentError.
  15 of 20 ended in VerifierTimeoutError: the rehearsal's t3 mode caps the verifier at 60 s and these tasks'
  verifiers take 49-131 s in the live run on the original images, so that's the cap, not the bake.
- Time lost: none.    Status: fixed for the next run (box setup); the 120 s limit itself is unchanged
- auxr_1103_0033: applied to the live boxes between steps 9 and 10 (21:23-21:25 UTC, after step 9's rollout closed,
  before step 10's trials): each box 19 baked, 1 had the tools, 0 failed; all 40 images verified (`tmux -V`,
  `asciinema --version`). Steps 1-9 ran on the original images, step 10 on the baked ones.

### 2026-11-03 auxr_1103_0033: one aborted trial re-runs a whole batch of 160 to replace one group of 8
- Seen: steps 4, 7 and 9 (and vanilla steps 5, 6): one aborted trial -> its group of 8 dropped -> 160 more trials
  sent (dispatcher: 64 running + ~96 queued); the first clean group to finish is kept, the rest cancelled.
- Cause (Miles `sglang_rollout.generate_rollout_async`): the `check_no_aborted` dynamic filter drops the group and
  the loop refills with `data_source(over_sampling_batch_size)`; that defaults to `rollout_batch_size` (20 prompts)
  and `arguments.py` asserts `over_sampling_batch_size >= rollout_batch_size`. The data source hands out the next
  prompts, not the dropped one, so the kept group is whichever task finishes first: step 4 trained two
  db-wal-recovery groups and no sanitize-git-repo, step 7 two large-scale-text-editing and no dna-insert, step 9
  replaced a 6/8 sqlite-with-gcov group with a second write-compressor group at 0/8 (step 9: 38/160 instead of
  ~44). "First to finish" favors tasks whose trials all hit a fixed time limit together, which the dispatcher also
  starts first (longest first): write-compressor's 8 all timed out at ~17 min and completed before any other group.
- Cost: ~30 min per drop (rollout 80 min vs ~50), 152 wasted trials, the cancelled trials' leaked containers (above),
  and a task-mix shift in that step's update.
- Not chosen: removing `check_no_aborted` and masking just the aborted attempt in `grpo/infra_filter.py`
  (judged hacky).
- Fix: `grpo/resubmit_group.patch` (applied by `setup.sh` step 6 after aux_reward.patch) adds
  `--resubmit-dropped-groups N` to Miles' `inference_rollout_train.generate_rollout_async` (the loop our
  `generate.RolloutFn` runs; `sglang_rollout.py` has a copy of the loop that we don't use): a dropped group is rerun
  with a copy of its own prompt as submitted, as many groups as were dropped, each up to N times (then the rollout
  stops with an error); no new prompts are drawn, so every rollout trains on all 20 tasks x 8. `configs/grpo.sh`
  sets 3. Tests: `tests/fast/rollout/test_resubmit_dropped_groups.py` (one drop, several drops, too many, off),
  with Miles' own rollout tests: 84 passed in the pinned image.
- auxr_1103_0033 was paused before step 9's update (21:31 UTC) and resumed from iter_0000007 (adapter + Adam state)
  with the fix (21:54 UTC): steps 9-10 use it, steps 1-8 used the old refill. The 320 trials of the stopped step-9
  attempt are marked abandoned by `monitor/compare.py` (Miles' restart line in train.log).
  A strict comparison with vanilla (old refill at its steps 5 and 6) has to keep that in mind.
- Time lost: ~30 min per drop.    Status: fixed (from auxr_1103_0033 step 9)

### 2026-11-03 auxr_1103_0033: the boxes' 15 h shutdown ended step 10 mid-rollout
- Symptom: step 10's trials began failing together; Miles logged "rerunning dropped group ... time 2/3" for every
  task within a minute (23:34-23:35 UTC), then stopped: `RuntimeError: group 1768 dropped 4 times (group_has_aborted),
  over --resubmit-dropped-groups 3`. The laptop's reap loop had printed "ssh failed" for both boxes since 23:30.
- Cause: the boxes were launched 08:33:54 UTC with user-data `shutdown -h +900` (`TTL_MIN`, a guard against
  forgotten boxes). Logins are refused 5 min before (23:29), and at 23:34 both shut down and terminated
  (`--instance-initiated-shutdown-behavior terminate`). The resume at 22:01 projected the end at ~01:00 UTC; nobody
  compared it with the boxes' end.
- Effect: step 9 had trained and saved (iter_0000008, 29.375%); step 10 never updated. The finish session then started
  the backup + teardown; it was stopped before anything was terminated, and the laptop's H200-delete watcher too.
- Fix: `ec2_box.sh serve` writes the scheduled shutdown (`/run/systemd/shutdown/scheduled`, epoch s) to
  `~/data/trials/shutdown_at`, readable with the H200's read-only sync key. `scripts/box_check.sh` (run by
  `launch.sh` after `links.sh start`, before tracker/train) checks per box: agent server answers through its tunnel
  (127.0.0.1:1100i/health), and shutdown >= NEED_MIN away, NEED_MIN = (STEPS - steps done) x STEP_MIN (90) +
  MARGIN_MIN (30). Any failure stops the links and exits. Tests: `tests/test_sandboxes.py` (box_check: pass, short
  TTL, no ping / unreadable time).
- Recovery: new boxes (on-demand, TTL_MIN=420), images rebaked by `prepare`, resumed from iter_0000008.
- Time lost: ~2.5 h (step 10's rollout + new boxes).    Status: fixed

### 2026-11-03 auxr_1103_0033: backup OK, but the H200 kept running ~2 h after the run
- Symptom: `finish.sh` logged BACKUP_OK (44 files, 48.0 GB) and terminated the boxes at 01:26 UTC, then "H200 not
  deleted: nebius CLI not authorized". The laptop's delete watcher never fired; the H200 idled until ~03:30 UTC (~$12).
- Cause: `finish.sh` uploaded the final `backup_status` to S3 only inside its own nebius-delete branch. Without a
  Nebius profile on the H200, S3 kept the "running ..." status the backup uploaded mid-way, which the laptop watcher
  (waiting for BACKUP_OK) never saw.
- Fix: `finish.sh` publishes the status to S3 right after the size check, whatever follows; the laptop watcher is
  now `scripts/teardown_watch.sh <run> <instance-id>` (summary.md step 3). The H200 no longer needs Nebius
  credentials. Tests: `tests/test_sandboxes.py` (finish publishes BACKUP_OK with and without nebius; the watcher
  deletes on BACKUP_OK, never on BACKUP_INCOMPLETE; the publish test fails on the old finish.sh).
- Time lost: ~2 h of an idle H200.    Status: fixed
