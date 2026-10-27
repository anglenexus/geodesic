# EC2 sandbox sizing: 2 boxes instead of 3 (2026-10-27)

## Verdict

**Use 2 × m7a.8xlarge with 32 trials each (64 in total, as now).** Step time should stay the same, and it saves
**~$1.85/h**, about **$20 per 10-hour run** (one third of the EC2 bill).

- **Step time is set by trial time limits and the total concurrency of 64, not by the boxes.**
  - In the last run, every task took the same time on each of the 3 boxes.
  - Right now each box is under 15% CPU and 7% RAM.
  - 2 × 32 keeps 64 slots, so the rollout time doesn't change.
- **A bigger instance type doesn't help.**
  - m7a prices scale linearly with size: 2 × m7a.12xlarge costs the same as 3 × m7a.8xlarge, and
    1 × m7a.16xlarge the same as 2 × 8xlarge.
  - 1 big box would put all 64 trial starts on one disk and one IP, which is the failure mode below.
- **Cheaper still:** 2 × m7i.8xlarge (Intel, $1.61/h list).
  - The same size class already ran Harbor cleanly at 48 per box.
  - Saves another ~$0.49/h against 2 × m7a.
  - Needs a one-line `TYPE` override; the images are amd64 either way.

## Evidence

| Source | Finding |
|---|---|
| Task requests (`task.toml`, 20 training tasks) | 19 tasks: 1 CPU; overfull-hbox: 2. RAM: 15 tasks 2 GB, 5 tasks 4 GB (crack-7z-hash, financial-document-processor, overfull-hbox, dna-insert). Storage 10 GB each. These are Docker **caps**, not reservations. |
| Worst case for 32 on one box (caps) | Mean 2.5 GB × 32 = 80 GB; the worst mix (4 × 8 trials of the 4 GB tasks) is 128 GB of caps. CPU caps 32–40. Real use is far below the caps (next row). |
| **Live, run grpoc_1027_2058, 21 trials per box, 10 samples over 10 min** | CPU mean 2–6%, peak 13%. Load mean 1.5–3.3, peak 6 (of 32 vCPU). RAM used 6–8 GB (of 123). iowait mean 2–5%, single-sample spikes 24–34% (container start or teardown). Steal 0. |
| Scaled to 32 per box (× 1.5) | CPU peak ~20%, RAM ~12 GB. iowait spikes more often: the one resource to watch. |
| Last run (grpo10h_r2, 1,280 trials, 21 per box) | Mean trial 13.8 / 14.0 / 14.5 min on boxes 2 / 1 / 3. Per-task means are within noise across boxes (e.g. mailman 31.5 / 31.7 / 32.1 min). 45% of trials hit the agent time limit (TimeLimitExceeded), so trials are bound by those limits. |
| Last run, failures by box | 0 "Command timed out" (tool-install) and 0 environment-start failures on all 3 boxes. AgentError 33 / 31 / 33, spread evenly across tasks: not load-related. |
| Harbor collection, 32-vCPU m7i.8xlarge at `-n 48` (2026-10-19) | 1 of 249 trials failed. |
| Harbor collection, i3 32-vCPU at 48 concurrent (2026-10-24, night1024_checks.md) | 0 "Command timed out"; 1.2% no-reward; agent timeouts at baseline; Docker CPU ~51%; load 37 only during the startup wave. That ran with a fast API model (more commands per minute than our Qwen serving at ~585 tok/s across 64 agents). |
| Failure mode (memory note, 2026-10-19) | 150 trials started at once on one m7i.48xlarge gave 41% tool-install timeouts (one gp3 volume and one IP; CPU was 2%). Rule: ≤ ~50 starts at once per instance. 32 is within it. |
| Harbor brief rule (`ec2/AGENT_BRIEF.md`) | Cap = 1.5 × vCPU = 48 on 32 vCPU. 32 is 2/3 of the cap. |

## Cost (us-east-2 on-demand, Linux)

The AWS pricing API is denied for `agent_aees`, so these are public list prices.

| Setup | $/h | 10-h run + ~1 h setup/tests |
|---|---|---|
| 3 × m7a.8xlarge (now) | 5.56 | ~$61 |
| **2 × m7a.8xlarge** | **3.71** | **~$41** |
| 2 × m7i.8xlarge | 3.23 | ~$36 |
| 2 × m7a.12xlarge | 5.56 | ~$61 (no saving) |

Spot is denied by the org's service control policy, so on-demand is assumed throughout.

## What would change (next launch, not the running run)

- **Laptop:** `scripts/kickoff.sh` calls `ec2_sandboxes.sh up` with the default of 3 (`N=${2:-3}` in
  `ec2_sandboxes.sh`). Change the default to 2, or pass `up 2`. Optionally `TYPE=m7i.8xlarge`.
- **H200:** `SLOTS=32`:
  - `scripts/launch.sh` defaults `SLOTS=21`, with a comment about 3 × 21;
  - `scripts/kickoff.sh` passes `SLOTS=21` to `links.sh`;
  - `links.sh` itself already defaults to 32.
- **Boxes:** the agent server's `CONCURRENCY` comes from `ec2_box.sh` `SLOTS=32`, which already matches. The
  dispatcher never sends more than its slots, so 32 = 32 is fine.
- **Optional:** give the root gp3 volume more IOPS and throughput (e.g. 6000 IOPS / 250 MB/s, a few cents an hour)
  to absorb the start/teardown iowait spikes at 32 per box. It's set in `ec2_sandboxes.sh`
  `--block-device-mappings`.
- **Docs:** `dist_setup.md` (layout, "32 per box"), `summary.md` (step 5 SLOTS) and the tests that assert 3 boxes
  or 21 slots, if any.

## Risks

- **The start wave.** Each rollout launches 32 trials at once per box, against 21 now. That's below the
  ~50-at-once limit, and 48 at once was clean on the same size. But the iowait spikes (already 24–34% at 21) will
  be more frequent. Watch "Command timed out after 120 seconds" on the first step; if it appears above ~1%, add
  gp3 IOPS or go back to 3 boxes.
- **Losing a box costs half the capacity, not a third.** The dispatcher retries on the surviving box, but a step
  would then run at 32 slots, nearly doubling the rollout. This hasn't happened in any run so far.
- **RAM caps.** If the longest-first order puts many 4 GB tasks on one box at once, the caps add up to 128 GB.
  Actual use is ~0.3 GB per trial, so this only matters if a task really fills its cap.
- **Not measured:** 32 Harbor trials per box under Miles/Qwen serving. The first step of the next run is the
  check: compare per-task trial minutes and tool-install timeouts with grpo10h_r2.
