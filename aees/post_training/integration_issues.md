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
