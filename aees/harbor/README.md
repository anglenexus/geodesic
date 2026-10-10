# Running Agentic Evaluations with Harbor

This guide explains how to run agentic evaluations, such as SWE-bench Verified, with [Harbor](https://github.com/laude-institute/harbor). The agent can use a hosted API model (OpenAI or any other LiteLLM provider) or a model you serve yourself (SGLang or vLLM). Each run produces trajectories and rewards in Harbor's `ATIF` format, which AEES uses as the raw data for training the reward and steering models.

Every command below was tested with Harbor `v0.23.0` on an Apple Silicon Mac with Docker Desktop. Values in `<ANGLE_BRACKETS>` are placeholders for you to fill in.

---

## 0. How a run works

```
harbor run ──► for each task (× attempts):
               1. build/pull the task's Docker image        (environment setup)
               2. start the agent; it sends keystrokes to a tmux shell
                  inside the container and calls the LLM from the host
               3. run the task's tests in the container      (verifier → reward)
               4. write everything to <jobs-dir>/<timestamp>/<task>__<id>/
```

- The **agent's LLM calls come from your host machine**, not from inside the container. Use a `localhost` URL for a model served locally or through an SSH tunnel.
- The **verifier needs internet access.** Some tests call external hosts such as `httpbin.org`, and the SWE-bench grader installs `swebench` from PyPI. If the network drops, the trial gets a reward of 0 and **no exception is raised** (see Troubleshooting).

---

## 1. One-time setup

### 1.1 Install Harbor

```bash
uv tool install harbor          # provides the `harbor` command (aliases: hb, hr)
harbor --version                # tested with 0.23.0
harbor datasets list            # browse available benchmarks
```

### 1.2 Docker

```bash
docker version --format '{{.Server.Os}}/{{.Server.Arch}}'   # daemon must be running
docker login                                                # needed: anonymous pulls hit Docker Hub's 429 rate limit
```

**Apple Silicon (arm64) only:** SWE-bench images are published only for `amd64`.
1. In Docker Desktop, go to **Settings → General** and enable **"Use Rosetta for x86_64/amd64 emulation on Apple Silicon"**.
2. Set this variable on **every** `harbor run`, or export it in your shell profile:
   ```bash
   export DOCKER_DEFAULT_PLATFORM=linux/amd64
   ```
   Without it, every build fails with `no match for platform in manifest: not found`.

On an x86_64 Linux machine, skip both steps. Emulated runs on a Mac are slow (about 5–10 minutes per environment build), so use x86 Linux for large runs.

### 1.3 Keep the machine awake

On a laptop, wrap long runs in `caffeinate -i` (macOS). If the Mac sleeps, the network drops and every trial in progress fails verification.

---

## 2. Sanity check: the oracle agent

The `oracle` agent applies each task's reference solution instead of calling an LLM. If the oracle doesn't score about 1.0, the problem is your setup, not the model.

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 caffeinate -i harbor run \
  --dataset swebench-verified@1.0 \
  --agent oracle \
  --env docker \
  -o ~/code/jobs \
  -n 1 -r 1 \
  -i 'psf__requests-1142' -i 'psf__requests-2317'
```

Expected result: both trials score `1.0`. These two tasks have the smallest images and are good for quick checks.

---

## 3. Choosing tasks, concurrency, and attempts

| Flag | Meaning | Example |
|---|---|---|
| `--dataset <name>@<version>` | Benchmark from the registry | `swebench-verified@1.0` |
| `-i '<glob>'` | Include tasks matching a name pattern (repeatable) | `-i 'django__*'` |
| `-x '<glob>'` | Exclude tasks matching a name pattern (repeatable) | `-x 'sympy__*'` |
| `-l <N>` | Run at most N tasks, after filtering | `-l 20` |
| `-k <N>` | Attempts per task. More attempts give more trajectories per task, which is useful for reward-model data | `-k 4` |
| `-n <N>` | Trials run concurrently. Watch Docker disk and RAM, and stay under your API rate limit | `-n 4` |
| `-r <N>` | Retries for trials that raise exceptions, e.g. transient 429s | `-r 2` |
| `-o <dir>` | Where to write job folders (default: `./jobs` in the current directory) | `-o ~/code/jobs` |
| `--job-name <name>` | Human-readable job name | `--job-name qwen4b_smoke` |
| `--timeout-multiplier <f>` | Scale all task timeouts (useful under emulation) | `--timeout-multiplier 2` |

---

## 4. Running an LLM agent

The examples below use `terminus-2`, Harbor's built-in terminal agent. It calls models through LiteLLM, so any LiteLLM model string works with `-m`. To list every option the agent accepts:

```bash
harbor agent schema terminus-2
```

Pass agent options with `--ak key=value`. Values are parsed as JSON, so numbers, booleans, and objects work.

### 4.1 OpenAI API (or another hosted LiteLLM provider)

**Step 1: put the key in a private `.env` file. Don't commit it, and don't paste it into chats.**

```bash
mkdir -p ~/.config/harbor
cat > ~/.config/harbor/.env <<'EOF'
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
EOF
chmod 600 ~/.config/harbor/.env
```

**Step 2: check that the key can access the model. This costs nothing.**

```bash
curl -s https://api.openai.com/v1/models/<OPENAI_MODEL_NAME> \
  -H "Authorization: Bearer $(grep OPENAI_API_KEY ~/.config/harbor/.env | cut -d= -f2-)"
```

The command should return a JSON object containing the model's `id`. An `error` field means a wrong key or no access to that model.

**Step 3: check that LiteLLM knows the model's context window and prices.**

```bash
P=$(find ~/.local/share/uv/tools/harbor -name model_prices_and_context_window_backup.json | head -1)
python3 -c "
import json; d=json.load(open('$P')); m=d.get('<OPENAI_MODEL_NAME>')
print(m and {k:m.get(k) for k in ['max_input_tokens','max_output_tokens','input_cost_per_token','cache_read_input_token_cost','output_cost_per_token']})"
```

- If this prints values, Harbor uses them automatically. That sets the summarization threshold and `cost_usd`.
- If it prints `None`, pass `model_info` yourself (see Step 4b). **Otherwise Harbor assumes a 1M-token context window** and never summarizes before the real limit.

**Step 4a: run with a model LiteLLM already knows.**

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 caffeinate -i harbor run \
  --env-file ~/.config/harbor/.env \
  --dataset swebench-verified@1.0 \
  --agent terminus-2 \
  -m openai/<OPENAI_MODEL_NAME> \
  --env docker \
  -o ~/code/jobs --job-name <JOB_NAME> \
  -n <N_CONCURRENT> -k <N_ATTEMPTS> -r 1 \
  -i '<TASK_GLOB>'
```

**Step 4b: run with a model LiteLLM doesn't know, by supplying the specs.**

```bash
  ... same as above, plus:
  --ak 'model_info={"max_input_tokens": <CONTEXT_MINUS_OUTPUT>, "max_output_tokens": <MAX_OUTPUT>, "input_cost_per_token": <USD_PER_INPUT_TOKEN>, "output_cost_per_token": <USD_PER_OUTPUT_TOKEN>}'
```

Optional settings:

| Option | Purpose |
|---|---|
| `--ak reasoning_effort=<none\|minimal\|low\|medium\|high\|xhigh\|max\|default>` | Reasoning effort for reasoning models |
| `--ak temperature=<float>` | Sampling temperature. Leave unset for the provider default |
| `--ak max_turns=<int>` | Cap on agent turns. The default is effectively unlimited |
| `--ak use_responses_api=true` | Use OpenAI's Responses API instead of Chat Completions |

**Other providers** work the same way: put the provider's key in the `.env` file and change the `-m` prefix, e.g. `-m anthropic/<MODEL>` with `ANTHROPIC_API_KEY=...`, or `-m gemini/<MODEL>` with `GEMINI_API_KEY=...`.

### 4.2 Self-hosted model (SGLang / vLLM, OpenAI-compatible)

Use the `hosted_vllm/<served-model-name>` prefix. It works with any server that exposes `/v1/chat/completions`, including SGLang. Harbor **requires** `model_info` for this prefix.

**Step 1: reach the server.** If it runs on a remote GPU machine, forward the port over SSH and leave this terminal open:

```bash
ssh -N -L <LOCAL_PORT>:localhost:<REMOTE_PORT> <USER>@<GPU_HOST>
# e.g. ssh -N -L 30000:localhost:30000 ubuntu@<GPU_HOST>
```

**Step 2: read the server's model name and context length.**

```bash
curl -s localhost:<LOCAL_PORT>/v1/models
# → {"data":[{"id":"<SERVED_MODEL_NAME>", ..., "max_model_len":<CONTEXT_LENGTH>}]}

# SGLang only: tool parser, reasoning parser, context length
curl -s localhost:<LOCAL_PORT>/get_server_info | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ['model_path','served_model_name','context_length','tool_call_parser','reasoning_parser']})"
```

**Step 3: fill in the model specs.** Prompt and output share the server's context window, so choose the split so that:

```
max_input_tokens + max_output_tokens  <=  <CONTEXT_LENGTH>
```

| Field | What to put | Example (Qwen3-4B-Instruct, 32k) |
|---|---|---|
| `max_input_tokens` | `<CONTEXT_LENGTH> − max_output_tokens` | `28672` |
| `max_output_tokens` | Longest single response you allow; raise it for reasoning models | `4096` |
| `input_cost_per_token` | `0.0`, or your amortized GPU cost | `0.0` |
| `output_cost_per_token` | `0.0`, or your amortized GPU cost | `0.0` |

`terminus-2` summarizes its history once fewer than `proactive_summarization_threshold` tokens (default 8000) are free. With the example values, summarization starts at about 20k tokens of history.

**Step 4: run.**

```bash
HOSTED_VLLM_API_KEY=<SERVER_API_KEY_OR_none> \
DOCKER_DEFAULT_PLATFORM=linux/amd64 caffeinate -i harbor run \
  --dataset swebench-verified@1.0 \
  --agent terminus-2 \
  -m hosted_vllm/<SERVED_MODEL_NAME> \
  --ak api_base=http://localhost:<LOCAL_PORT>/v1 \
  --ak 'model_info={"max_input_tokens": <MAX_INPUT_TOKENS>, "max_output_tokens": <MAX_OUTPUT_TOKENS>, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}' \
  --env docker \
  -o ~/code/jobs --job-name <JOB_NAME> \
  -n <N_CONCURRENT> -k <N_ATTEMPTS> -r 1 \
  -i '<TASK_GLOB>'
```

This exact command (with `qwen3-4b`, port `30000`, and the example values above) has been tested end to end. It reached a reward on `psf__requests-1142`, and SGLang's prefix cache served about 85% of input tokens.

Requirements for the served model to drive `terminus-2`:
- It must follow the JSON response format in the agent's prompt. Small models sometimes get it wrong. Look for `Parser warnings` in `trial.log`.
- For token-level RL data, add `--ak collect_rollout_details=true` to record token IDs. The token IDs are incomplete for trajectories where summarization happened.

---

## 5. Outputs

```
<jobs-dir>/<timestamp or job-name>/
├── result.json                  # job aggregate: per-eval mean, pass@k, exception counts
├── job.log
└── <task>__<trial-id>/
    ├── result.json              # timings per phase, agent_result (tokens, cost), verifier_result (reward), exception_info
    ├── trial.log                # harness log (parser warnings, retries)
    ├── exception.txt            # only if the trial errored
    ├── agent/
    │   ├── trajectory.json      # ATIF trajectory (LLM agents only; oracle writes oracle.txt)
    │   ├── recording.cast       # asciinema terminal recording
    │   └── terminus_2.pane      # final tmux pane contents
    └── verifier/
        ├── reward.txt           # scalar reward
        ├── report.json          # per-test results (SWE-bench: FAIL_TO_PASS / PASS_TO_PASS)
        └── test-stdout.txt      # full test log
```

### Trajectory format (`agent/trajectory.json`, ATIF)

```
Trajectory{schema_version, session_id, agent{name, version, model_name}, steps[], final_metrics,
           subagent_trajectories, continued_trajectory_ref}
Step{step_id, source: system|user|agent, model_name, message, reasoning_content,
     tool_calls[{tool_call_id, function_name, arguments}],
     observation{results[{source_call_id, content}]},
     metrics{prompt_tokens, completion_tokens, cached_tokens}}
```

For `terminus-2`, every tool call is `bash_command` with `{"keystrokes": ..., "duration": ...}`, and the observation is the terminal output, capped at 10 KB per turn. Step 1 holds the system prompt plus the task.

### Inspecting results

```bash
harbor view jobs                          # interactive viewer

# Summary table for one job
python3 - <<'EOF'
import json, glob, sys
job = "<JOB_DIR>"   # e.g. ~/code/jobs/2026-10-02__20-18-35
for d in sorted(glob.glob(f"{job}/*/")):
    r = json.load(open(d + "result.json"))
    a = r.get("agent_result") or {}
    exc = (r.get("exception_info") or {}).get("exception_type")
    rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    try:
        steps = len(json.load(open(d + "agent/trajectory.json"))["steps"])
    except FileNotFoundError:
        steps = None
    print(f"{r['trial_name']:45s} reward={rew} exc={exc} steps={steps} "
          f"in={a.get('n_input_tokens')} cached={a.get('n_cache_tokens')} out={a.get('n_output_tokens')}")
EOF
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every trial raises `RuntimeError` with `no match for platform in manifest` | arm64 host and amd64-only image | `DOCKER_DEFAULT_PLATFORM=linux/amd64` plus Rosetta (§1.2) |
| `429 Too Many Requests` from `registry-1.docker.io` | Anonymous Docker Hub rate limit | `docker login`; add `-r 2` |
| Reward 0, no exception, `verifier/report.json` missing, `gaierror` / `Could not connect` in `test-stdout.txt` | Network dropped during verification (e.g. laptop slept) | Rerun the trial; use `caffeinate -i`. **Treat reward 0 without `report.json` as an infrastructure failure, not a model failure** |
| Trials finish in ~1 s with exceptions | Failure while building the environment | Read `<trial>/exception.txt` |
| LLM errors mid-trial against a self-hosted model | SSH tunnel dropped, or the context limit was exceeded | Keep the tunnel open; check that `model_info.max_input_tokens + max_output_tokens <= server context` |
| `RateLimitError` on every trial, `n_input_tokens: 0`, `trial.log` says `You have no credits remaining` | API account has no prepaid balance. LiteLLM maps OpenAI's billing 429 to `RateLimitError`, so it looks like throttling | Add credits under platform.openai.com → Settings → Billing (separate from ChatGPT subscriptions); retries won't help |
| `cost_usd: null` | Model has no price in LiteLLM (normal for `hosted_vllm` at 0 cost) | Ignore it, or set costs in `model_info` |
| Agent "solves" a task by `pip install --upgrade <package-under-test>` | A weak model working around the task | The grader catches this (reward 0); worth flagging for reward-model data |

---

## 7. Reference measurements (SWE-bench Verified, Apple Silicon, Docker + Rosetta)

| Run | Env build | Agent | Verifier | Reward | Tokens (in / cached / out) |
|---|---|---|---|---|---|
| oracle, `psf__requests-1142` | ~6 min (first image pull) | ~1 s | ~30 s | 1.0 | n/a |
| oracle, `psf__requests-2317` | cached | ~1 s | ~10 min | 1.0 | n/a |
| terminus-2 + Qwen3-4B-Instruct (SGLang), `psf__requests-1142` | cached | 5.5 min, 8 turns | ~15 s | 0.0 | 36.5k / 31.2k / 4.2k |
| terminus-2 + gpt-5.6-luna, `psf__requests-1142` | cached | 1.8 min, 16 turns | ~15 s | 0.0 ($0.017) | 183k / 164k / 7.7k |
| terminus-2 + gpt-5.6-luna, `psf__requests-2317` | cached | 1.6 min, 11 turns | ~11 min | 1.0 ($0.011) | 115k / 97k / 3.9k |

How token use scales: `terminus-2` resends the full history on every turn, so total input grows roughly with the square of the number of turns. A 40-turn trajectory costs roughly 1M input tokens before caching. Prefix caching, which OpenAI applies automatically and SGLang provides through its radix cache, typically covers 85–90% of input.
