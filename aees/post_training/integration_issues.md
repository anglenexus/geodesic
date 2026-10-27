# Integration issues

Every failure seen while setting up or testing the distributed run (`dist_setup.md`, `integration_tests.md`): symptom,
cause, fix, and time lost. **Fix budget: 60 min of H200 time in total.** If a test is still red after that, stop,
write the state here, and decide with the user. Don't start the 10-hour run on a red test.

Once a fix is confirmed, move what works into `dist_setup.md`.

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
