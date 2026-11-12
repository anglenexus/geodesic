# Aligning our GRPO setup with Tmax

Living plan: decisions are added to the log as they are made, and the sections below are revised to match.
Reference run for "ours": `auxr_1103_0033` (Qwen3.6-27B, LoRA r=64, 1×H200 colocated, thinking off, 48k).
Reference for "Tmax": Ivison et al., *Tmax: A simple recipe for terminal agents* ([arXiv 2606.23321](https://arxiv.org/pdf/2606.23321)),
code and released evaluation records at [github.com/hamishivi/tmax](https://github.com/hamishivi/tmax).

Where Tmax's evaluation and training settings differ, we align with their **evaluation** settings (task defaults,
as the TB leaderboard requires) unless a decision below says otherwise.

## Decision log

| Date (PT) | Decision | Why / evidence |
|---|---|---|
| 2026-11-11 | Compare our start rate with Tmax **per task**, not their headline number | On our 20 training tasks, Tmax's own base Qwen3.6-27B scores 24.0% (released TB 2.0 records); ours starts at 24.4%. Their 39.6–43.6% is over all 88–89 tasks. |
| 2026-11-11 | Verifier timeout: **each task's own `task.toml` limit** (removed the flat 600 s cap) | Tmax: `verifier.override_timeout_sec = None` in all 445 trial configs. The 600 s cap cut off 16 verifiers (8 mailman, 8 large-scale-text-editing) in auxr_1103_0033. Done in `scripts/ec2_box.sh`; `VERIFIER_TIMEOUT_SEC` remains an opt-in cap. |
| 2026-11-11 | Thinking **on**, kept across turns (interleaved), and **trained on** | Tmax keeps raw model tokens, thinking included, in the context and gives them loss mask 1 (`vllm_utils.py`: `current_prompt.extend(model_tokens)`, `response_masks.extend([1] * len(model_tokens))`). |
| 2026-11-11 | Rollout sampling temperature **1.0** | Tmax RL: `--temperature 1.0` (Table 13). Ours already 1.0. Their 0.7 / top_p 0.95 is only their eval agent's default. |
| 2026-11-11 | Context **65,536 for both rollouts and training** | Tmax: `--response_length 65536`. Training on a shorter context than the rollout would drop the end of long trajectories from the gradient. |
| 2026-11-11 | Hardware: **8×H100, on demand** (not preemptible) | H100 and H200 have the same compute; 8×H100 gives 2× the compute and 1.4× the memory bandwidth of 4×H200 for 1.67× the price. Nebius list: $4.50/GPU-h → $36/h. |
| 2026-11-11 | Concurrency **40 trials** (2 EC2 boxes × 20), not 64 | 64 was too much for one server (SGLang capped at 42 running requests; ~12 queued; ~50% cache hit). 40 leaves headroom. See "Concurrency design". |
| 2026-11-11 | First experiment: **~1 h rollout-only baseline** with the new settings, to get success rates before any training | See "Experiment 1". |
| 2026-11-12 | **Adopt 5b: 64 model calls** per trial (`HARBOR_AGENT_MAX_ITERATIONS=64`) | Tmax's step = one model call. Our max was 44 (thinking off); Tmax's thinking-on 27B: 5% of trials reach 64. |
| 2026-11-12 | **Adopt 5a: 120 s per command**, via a Terminus-2 patch (`grpo/command_timeout.patch`), not a harness switch | Keeps our harness, SFT data format and Q-function input. Semantics in "Implementation: Experiment 1". |
| 2026-11-12 | Training at 65k needs the sequence split across GPUs: **context parallelism in Miles**, scoped **after** Experiment 1 | Miles' DeltaNet layer has context-parallel code (`miles_plugins/models/linear_attn.py`, `cp_size > 1`); LoRA + CP is untested. |
| 2026-11-12 | Core-code review findings fixed: job detection by process group (a name check missed `bash`/`sh` scripts), no Ctrl-C when the terminal can't be read, kill after an ignored Ctrl-C, race re-check, driver uses GRPO's infra exclusions, no silently lost rows | Review of `rollout_only.py`, `dispatch.py`, `command_timeout.patch`; one finding (queued commands) kept as a documented difference |
| 2026-11-12 | First attempt `phase1_1112_0027` stopped after 12 min | A model call reaching 16,384 tokens (long thinking) ended the trial with reward 0 (response-length policy `abort`, chosen for Miles' session server): 6 of the first 21 trials. Tmax treats a truncated reply as a format error and continues. Partial data kept. |
| 2026-11-12 | Restarted as `phase1_1112_0027b` with Tmax's **eval** settings: response-length policy `recover` (truncated turn kept, agent told to retry, continues); **temperature 0.7, top_p 0.95** (Tmax's eval agent; their RL uses 1.0); **serving context 65,536** in total (was 81,920) | Reproduce Tmax's success rate as their eval measured it. SGLang, like vLLM, refuses a call when prompt + max_tokens exceeds the context (tested). Thinking on throughout (probe: reasoning_content present, preserve_thinking keeps it). |
| 2026-11-12 | **Grade context-limit stops with the verifier** (`HARBOR_TERMINUS_2_STOP_AT_CONTEXT=1` in phase1 mode) | As Tmax's agent: at the context limit the agent stops and the container is graded. In `phase1_1112_0027b` the 22 such trials were scored 0 ungraded; their containers are gone, so only a rerun can grade them. |
| 2026-11-12 | 120 s limit fixed: detection follows Terminus-2's `asciinema rec` terminal and compares with the shell's foreground group recorded before the first command | It never fired in `phase1_1112_0027b`; verified with a nested terminal (`script` + `sh -c` wrapper) and live trial containers. |
| 2026-11-12 | Review scope: **core code only** (`grpo/rollout_only.py`, `grpo/dispatch.py`, `grpo/command_timeout.patch`); shell scripts, the tracker and tests are not reviewed | User's instruction. |
| 2026-11-12 | Full TB 2.1 baseline on **Tmax's 88 tasks** (all 89 but configure-git-webserver, paper §H), 5 attempts, extendable to 8 | Reproduces the setting of Tmax's 43.6 ± 1.9 (Table 3; they ran 3 trials, default timeouts). Every task checked: list, dataset, limits and Tmax records agree on 89 names; 89/89 agent limits match task.toml; all 445 Tmax trials ran multiplier 1.0, no overrides. Run `tb21full_1112_1454b`, started 3:25 PM PT, cap $250. |
| 2026-11-12 | Extension: `touch $DATA/trials/$RUN/extend` before the 5th attempts finish queues attempts 6-8 into the same run | The extra attempts fill slots freed while the 200- and 120-min tasks run, ~+30-45 min. |

## Timeout settings: ours vs Tmax evaluation

| # | Setting | auxr_1103_0033 | Tmax evaluation (v0.1.0 agent, `a7f52442`) | Status |
|---|---|---|---|---|
| 1 | Container start | `task.toml` build timeout × 1 | × 1 | aligned |
| 2 | Agent setup | 360 s × 1 | 360 s × 1 | aligned |
| 3 | Agent phase (wall clock) | `task.toml` agent timeout × 1.0 (750–1800 s) | `timeout_multiplier 1.0`, no overrides | aligned |
| 4 | Verifier | flat 600 s | `task.toml` limit × 1 | **changed 2026-11-11 → aligned** |
| 5a | Per-command timeout | none: Terminus-2 waits ≤60 s, the command keeps running | **120 s, then the command is killed** (exit 124); the agent continues | **implemented 2026-11-12** for phase 1 (`grpo/command_timeout.patch`, `HARBOR_TERMINUS_2_COMMAND_TIMEOUT_SEC=120`) |
| 5b | Step limit | none (Terminus-2 default 1,000,000) | **64 steps**; 1 step = 1 model call = 1 bash command; format errors also use steps | **implemented 2026-11-12** for phase 1 (`HARBOR_AGENT_MAX_ITERATIONS=64`) |
| 5c | Model-call timeout | not set by us (LiteLLM default; to check) | 5 h (`LLM_TIMEOUT_SECONDS`) | not binding: the agent timeout fires first |
| 5d | Format-error limit | none | stop after 64 consecutive format errors | not a timeout; harness difference |
| 6 | Outer backstops | dispatcher: none; Miles `AGENT_TRIAL_TIMEOUT` 12 h | n/a | ours only; never fired |
| – | Resources (CPU, memory) | task defaults | task defaults | aligned |

Tmax's **RL training** rollouts used different limits from their evaluation: verifier `test_timeout` 120 s (June script)
or 600 s (current script), no wall-clock limit per rollout in the June code (a 600 s `SWERL_ROLLOUT_TIMEOUT_S` appears
only in the 2026-10-30 code), 64 steps and 120 s per command. **Open:** keep evaluation-style limits for our training
rollouts (current choice), or adopt theirs.

## Sanity checks for 5a and 5b

**What the limits mean in Tmax's agent** (`Vanillux2Agent/agent.py` at `a7f52442`, the version their eval ran):
each loop iteration makes one model call, parses **one** bash tool call (a command string, possibly a multi-line
script), and runs it with `environment.exec(..., timeout_sec=120)`. So 120 s applies to **one command**, and 64 is
the number of **model calls**. A command that times out returns exit code 124, and the agent keeps going.

**Our runs** (auxr_1103_0033, 1,986 trials with trajectories; thinking off). A Terminus-2 turn is one model call that
sends on average **2.43** commands.

| | Median | Mean | p90 | Max |
|---|---|---|---|---|
| Model calls (turns) per trial, all | 12 | 13.2 | 22 | **44** |
| Model calls per trial, passing (567) | 14 | 14.7 | 22 | 37 |
| Commands per trial, all | 26 | 32.0 | 61 | 161 |
| Commands per trial, passing | 28 | 33.1 | 55 | 127 |

- A 64-model-call limit would have stopped **0** of our trials (max 44).
- Counted as commands instead, 173 trials (8.7%) went past 64, including 36 passes. Tmax's step is a model call,
  so the faithful mapping is 64 Terminus-2 turns.
- With thinking on, call counts may change: Tmax's base Qwen3.6-27B (thinking on) used median 23, mean 24.6, p90 47
  steps on our 20 tasks, and **5%** of trials reached 64. Passing trials: median 27, max 64.

**Commands longer than 120 s.**
- Tmax, on our 20 tasks: 44 of 2,435 commands (**1.8%**) hit 120 s (all 89 tasks: 3.2%). The median command took 0.9 s.
- Ours: Terminus-2 doesn't record how long a command ran. A lower bound: **119 trials (6.0%)** waited ≥120 s in
  a row on a still-running command (consecutive wait-only turns), and **44 of them passed**. Under a 120 s kill,
  those passes would have needed the model to run the long work in the background.

**Decision (2026-11-12).** Both adopted; see the decision log and "Implementation: Experiment 1".

## Concurrency design (rough, 2026-11-11)

- **Serving:** 8×H100 as **4 SGLang engines × TP2**. Each engine: 54 GB weights, ~90 GB for the KV cache, ~10
  agents. Single-GPU engines would leave only ~15–20 GB of KV each and evict agents' histories.
- **Throughput assumption:** ~300 tok/s per GPU (matches the ~340 measured on the H200), scaled for the H100's lower
  memory bandwidth to **~1,800 tok/s** total.
- **Per agent:** 1,800 / 40 = **~45 tok/s** averaged over all time. During model calls (~25–30 of 40 agents in a
  call at once) that is **~60–75 tok/s per call**, against Tmax's measured ~48 (ours was ~9).
- **Rollout time for 160 trials (20 tasks × 8) through 40 slots**, simulated with Tmax's thinking-on trial durations
  on our tasks, capped at each task's limit, plus our measured setup and verifier time:
  - mean trial 12.4 min; ~2,000 trial-minutes; 50 min if perfectly packed;
  - trials started in random order: **median 70 min** (p90 73);
  - trials started **longest time limit first**: **median 57 min** (p90 59). Needs a small change in
    `grpo/dispatch.py`.
- **Training** at 65k: TP4 × 2 replicas, ~40 GB per GPU, roughly 10–15 min per step.
- **Not yet measured.** On real traffic, earlier serving numbers fell short of the arithmetic (585 tok/s in T2 against
  ~1,500 predicted).

## Experiment 1: rollout-only baseline (ran 2026-11-12, `phase1_1112_0027b`)

- **What:** base Qwen3.6-27B, thinking on, 65k context, 160 trials (20 tasks × 8), 40 at once, no training step.
- **Measures:** pass rate per task, timeout rate, model calls and commands per trial, tok/s per agent and per call,
  time to first token, prefix-cache hit.
- **Cost:** ~$55–65 (8×H100 at $36/h for ~1.5 h including setup, plus 2 EC2 boxes at $3.70/h).
- **Budget:** $180.22 of the current $200 cap is spent. **A new budget or a higher cap is needed first.**
- **Model:** base Qwen3.6-27B (no adapter).

## Implementation: Experiment 1 (2026-11-12)

No Miles: plain SGLang engines plus the existing dispatcher, tunnels and box agent servers, driven by a small client.

| Piece | File | What it does |
|---|---|---|
| Serving | `scripts/phase1_serve.sh` | 4 SGLang engines × TP2 (GPUs 0-1, 2-3, 4-5, 6-7) on 127.0.0.1:30001-30004; context 65,536 in total, as Tmax's vLLM `--max-model-len` (a call is refused once prompt + 16,384 output tokens would exceed it); `--reasoning-parser qwen3` so thinking comes back as `reasoning_content`; up to 16 running requests per engine; metrics and cache reports on |
| Driver (core) | `grpo/rollout_only.py` | 160 trials (20 tasks × 8) to the dispatcher at once; each trial pinned to one engine (its history stays in that engine's prefix cache), engines balanced by expected minutes; sampling: temperature 1.0, top_p 1.0, 16,384 tokens per call, `chat_template_kwargs` `enable_thinking` + `preserve_thinking`; `max_seq_len` 65,536; one row per trial in `rollouts.jsonl` |
| Order (core) | `grpo/dispatch.py --fixed-order`, `grpo/task_limits.json` | trials start by TB 2.1 agent time limit, longest first, for the whole run (no switch to observed means) |
| 120 s per command (core) | `grpo/command_timeout.patch` (Harbor, applied after `harbor_parser.patch`) | when keys that press Enter are sent while the shell is at its prompt (its process group is the terminal's foreground group), a timer starts. If a job is still in the foreground 120 s later, Terminus-2 sends Ctrl-C, and kills the job's process group if it is still there 2 s later; the next terminal output starts with "Command timed out after 120 seconds and was interrupted with Ctrl-C" (or "...killed"). Scripts run by `bash`/`sh` count as jobs. Keys typed into a running program start no timer. If the terminal can't be read, nothing is sent. Verified with real tmux in a container (idle, `sleep`, `bash -c`, `sh -c`, `&`, a job ignoring Ctrl-C). Off unless `HARBOR_TERMINUS_2_COMMAND_TIMEOUT_SEC` is set. **Known difference from Tmax:** commands typed in the same turn while an earlier one is still running are input to it; they share its timer and the Ctrl-C drops them (Tmax runs one command per step) |
| Box settings | `scripts/ec2_box.sh serve <run> phase1` | 20 slots; `MAX_LEN=65536`, `AGENT_MAX_OUTPUT_TOKENS=16384`, `HARBOR_AGENT_MAX_ITERATIONS=64`, `HARBOR_TERMINUS_2_COMMAND_TIMEOUT_SEC=120`; verifier at task limits |
| Host + boxes | `scripts/phase1_kickoff.sh` (laptop) | 8×H100 on Nebius eu-north1 (`gpu-h100-sxm`, `8gpu-128vcpu-1600gb`), **on-demand** EC2 boxes (`MARKET=on-demand`, 300 min TTL), serving-only setup (`SERVE_ONLY=1`: image + weights) |
| Run | `scripts/phase1.sh` (host) | engines, links (20 slots, fixed order), box check, tracker, driver, a 100-min deadline, then `finish.sh`: backup to S3, terminate boxes, delete the host |
| Tracker | `monitor/phase1.py` + `monitor/phase1.html`, `monitor/phase1_reference.json` | live progress, pass rate per task against Tmax and our thinking-off start, agent behavior, serving; port 8771 |

**Launch steps** (only after the budget is raised):
1. Laptop: restart the budget guard with the new cap (currently $200 with $180.22 spent).
2. Laptop: `bash scripts/phase1_kickoff.sh`, then `bash scripts/teardown_watch.sh <run> <instance-id> &`.
3. Host: `aws configure`; after `SETUP_DONE`: `RUN=<run> NEBIUS_INSTANCE_ID=<id> bash scripts/phase1.sh`.
4. Laptop: `PORT=8771 bash scripts/tunnel.sh <user>@<ip>` → http://localhost:8771.

**Not verified yet** (first minutes of the run): Nebius 8×H100 capacity in eu-north1; SGLang's `qwen3` reasoning
parser and Qwen3.6's `preserve_thinking` with the Miles image's SGLang version; the running-request cap each
engine reports (state slots); per-agent tokens/s.

## Experiment 1 results (2026-11-12, 1:00-1:48 AM PT)

Base Qwen3.6-27B, thinking on, Tmax eval settings (above), 160 trials, 0 infrastructure errors. Backup:
`s3://aees-grpo-backups-611728564241/phase1_1112_0027b/phase1.tgz` (trials of both attempts, rollouts.jsonl,
dispatch.jsonl, serving.jsonl, probe.json, final_status.json, SGLang logs) and `phase1.html` (open it, then "Open
snapshot" with final_status.json). Host and boxes torn down after the backup. Spend $57.38 of the $100 cap (both attempts, setup and teardown included).

| | Ours | Tmax (same 20 tasks) |
|---|---|---|
| Pass rate | **46.2%** (74/160) | 24.0% (24/100) |
| Agent time-limit stops | 18.8% | 16.0% |
| Context-limit stops (65k, ungraded here) | 13.8% (22) | graded by the verifier |
| Model calls per trial, median | 11 | 23.5 |
| Agent minutes per trial, median | 5.7 | 9.9 |
| Output tok/s per agent, during calls / overall | 53.9 / 34.5 | 47.8 / 26.6 |
| Share of agent time: model calls / commands | 64% / 31% | 56% / 20% |
| TTFT p50 / ITL p50 (serving) | ~130 ms / 12-13 ms | n/a |

**Not aligned in this run:** the 120 s per-command limit never fired (Terminus-2 runs the shell inside `asciinema
rec`; the check looked at the outer terminal; fixed and tested since, not yet run), and context-limit stops were
scored 0 without the verifier (Tmax grades them; `HARBOR_TERMINUS_2_STOP_AT_CONTEXT` added, not yet run). Per-task
results differ a lot from Tmax's 5-attempt eval (TB 2.1 here vs TB 2.0 there; different harness).

## Open questions

- Training-rollout limits: evaluation-style (current) or Tmax's training limits?
- Budget for Experiment 1.
- After Experiment 1: context parallelism for 65k training in Miles (with LoRA), and its GPU layout.
