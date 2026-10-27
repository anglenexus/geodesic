# Integration tests before the 10-hour GRPO run

Goal: prove every part of the distributed loop works in **≤ 20 min of H200 time**, then start training.
Setup: `dist_setup.md`. Failures and fixes: `integration_issues.md`. Results land in
`$DATA/monitor/$RUN/preflight.json` and show on the tracker's Preflight card. **Record each run in the log at the end.**

| Test | Budget | Needs |
|---|---|---|
| L: laptop unit tests | 0 (before kick-off) | – |
| R: EC2 rehearsal, no GPU | 0 H200 (~12 min, ~$0.40 of EC2) | AWS profile |
| T0: infrastructure | 2 min | setup done (conversion may still run) |
| T2: serving at 96 trials | 6–8 min | weights; GPU free (after conversion) |
| T3: mini GRPO, end to end | ≤ 10 min | everything |

## L: laptop (before the IP arrives)
`uv run pytest tests/` covers (1253 passed, 2 skipped on 2026-10-26):
- dispatcher: longest-first order, 32 per box, a box going down, abort broadcast, one log row per trial, 160
  connections at once (found: the standard server's backlog of 5 dropped connections; now 1024);
- scripts with stubbed `ssh`/`aws`/`docker`: EC2 launch and teardown, tunnels, trial sync;
- configs: `SAVE_INTERVAL`, batch-size and concurrency overrides, SGLang flags.

Pass: full suite green.

## R: EC2 rehearsal (`scripts/rehearse.sh`, before the IP)
Removes most of the fix budget: every non-GPU part runs for real once, on the same OS and AWS setup.
- One real box (`ec2_sandboxes.sh up 1`).
- A c7a.large standing in for the H200. It runs the real `links.sh` (tunnels, sync, dispatcher), the tracker, and
  `scripts/stub_model.py` in place of Miles' session server.
- 4 real Harbor trials (configure-git-webserver, sqlite-with-gcov × 2) go: dispatcher → `-L` tunnel → agent server →
  terminus-2 → `-R` tunnel → stub → real verifier → reward → dispatcher log → trial sync → tracker.

Pass:
- box prepared (setup.sh 2 3 5 8 on Ubuntu 24.04, 20 images);
- restricted keys accepted by sshd (`permitlisten`, `rrsync`);
- probe: 16/16 reverse listeners and the stub answers;
- 4 replies with `exit_status` Submitted (not `Error:`);
- 4 result.json synced;
- the tracker shows 4 dispatch rows and the box up.

Not covered: anything on the GPU, Miles, SGLang.

## T0: infrastructure (2 min, scripted, no GPU load)
- H200: driver ≥ 580; GPU visible in a container; weights present; Miles image present; ≥ 400 GB free.
- Each box: Docker up; the 20 training images present; agent server `/health` reachable from the H200 through the
  `-L` tunnels (11001–11003).
- Reverse tunnels: `preflight.sh t0` serves a stub on the H200's 127.0.0.1:30000 for 5 min. The laptop's
  `ec2_sandboxes.sh probe <h200-ip>` checks that each box has 16 listeners and that the stub answers through
  them, and appends its rows to the same preflight log.
- The image's SGLang knows every flag we pass (`--help`), so a renamed flag fails here, not in T2.
- Dispatcher: its `/health` reports 3 boxes up and 96 slots.
- Tracker: `/status.json` answers.

## T2: serving at 96 trials (6–8 min)
Most serving questions are already answered by the H200 benchmarks (serving/bench_results/blog_source.md,
Qwen3.5-27B; same architecture) and by the 2026-10-25 prototype:

| Known | Value |
|---|---|
| BF16, 32k context, mem 0.88 | running-request cap **46** (DeltaNet state slots bind: 5 slots per running request with the prefix cache) |
| FP8 | cap 61 |
| BF16 throughput, agents × 64 | 1,527 tok/s, TTFT p50 ~1 s; FP8 2,449 tok/s |
| FP8 KV cache | ~1.35× slower decode on the H200: keep the BF16 KV cache |
| Prefix-cache hit, mem 0.85 (prototype) | 88% |
| Prefix-cache hit, mem 0.36 (prototype) | 9–13%, TTFT 19–39 s |

The open question is our exact setup: BF16, LoRA r=64, 56k context, mem 0.85, 96 trials. About 80% of a trial is
spent waiting on the model, so ~75 requests are in flight at once. That is above the expected cap of ~40–46,
so requests would queue.

Steps:
1. **Static check, ~2–3 min** (no load). Start SGLang with the exact flags Miles passes (`configs/grpo.sh`) plus the
   adapter, and read the startup log: running-request cap, KV tokens, state slots.
   - If the cap is < 64, restart once with `--max-mamba-cache-size` / `--max-running-requests` raised,
     keeping KV ≥ ~1M tokens.
2. **Load, 3 min.** Run `proto/agent_load` with 96 agents replaying the real call profile
   (`proto/load_profile.json`), reading SGLang `/metrics`.

Pass:
- cap ≥ 64;
- prefix-cache hit ≥ 60%;
- TTFT p90 ≤ 5 s;
- no OOM, headroom ≥ 8 GB.

The flags that pass go into `configs/grpo.sh` before T3.

## T3: mini GRPO, end to end (≤ 10 min)
The real Miles run with the real dispatcher, boxes and tunnels, made short. Rollouts don't need to finish
naturally: trajectories already exist in `aees/harbor`.

| Setting | Value | Why |
|---|---|---|
| Tasks × samples | 2 × 4 (`configure-git-webserver`, `sqlite-with-gcov`: fast verifiers) | 8 trials per step; 4 = `grpo/infra_filter.py` MIN_VALID (smaller groups are all masked and the run stops) |
| `HARBOR_AGENT_MAX_ITERATIONS` | 2 (agent servers, T3 only) | trials end after 2 turns |
| `HARBOR_VERIFIER_TIMEOUT_SEC` | 60 (T3 only) | caps the verifier |
| `NUM_ROLLOUT` | 2 | step 2 runs on step 1's adapter |
| `--global-batch-size` | 8 | one update per step |
| `SAVE_INTERVAL` | 1 | checks the save |

Expected time:
- Miles start ~5 min (torch_dist load, SGLang with a warm JIT cache);
- 2 × (rollout ~1.5 min + train < 1 min + switch);
- save.

Pass:
- each trial returns a reward through the dispatcher, with no infra errors;
- Miles logs both steps (training finishes, peak memory logged);
- step 2's rollout is served by the updated weights (Miles weight version 1 → 2);
- `iter_0000001/adapter` saved;
- the tracker shows 2 steps and 16 trials (trial sync works);
- tunnels survive the run.

**Not covered** (checked during iteration 1 of the real run, which the tracker shows live):
- a non-zero gradient: 2-turn trials will mostly score 0, so the advantages are zero;
- a training micro-batch near 48k tokens: 4 short trajectories are far below it;
- 96 trials at once through Miles.

After T3, restart the agent servers without the two T3 variables (seconds).

## Log

| Date | Test | Result | Time | Notes |
|---|---|---|---|---|
| 2026-10-26 | L | pass | – | 1253 passed, 2 skipped |
| 2026-10-27 | R | stopped | 6 min | caught the agent server's Docker Hub guard; superseded by running the same path on the real boxes |
| 2026-10-27 | T0 | pass, 11/11 | 1 min | after fixing tmux's docker group (see issues) |
| 2026-10-27 | T2 at 96 | fail | 8 min | cap 39 (state slots), cache hit 0.355, TTFT p90 39 s, 42 ms/token, headroom 16 GB |
| 2026-10-27 | T3 | pass | 14 min (Miles start ~4.5) | 2 steps trained + saved; 16 trials via dispatcher and 3 boxes; synced; tracker 2 steps / 16 trials; rewards 0 (2-turn trials); Miles' own cap 42 |
