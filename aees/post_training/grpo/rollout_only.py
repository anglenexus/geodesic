"""Rollouts without training (align_tmax_plan.md, Experiment 1): every task × attempts through the dispatcher.

Sends each trial to grpo/dispatch.py (which queues it, longest expected first, and forwards it to a box), the way
Miles does during GRPO, but with the model served by plain SGLang engines instead of Miles' session servers. Each
trial is pinned to one engine, so its growing history stays in that engine's prefix cache; trials are spread so
every engine gets about the same expected minutes. One JSONL row per finished trial.

  python3 grpo/rollout_only.py --engines http://127.0.0.1:30001/v1,http://127.0.0.1:30002/v1,... \
      --tasks ../harbor/state/tb21_selected_tasks.txt --attempts 8 --minutes grpo/task_limits.json \
      --out $DATA/trials/$RUN/rollouts.jsonl
With --extend-to 8 --extend-file $DATA/trials/$RUN/extend, creating that file before the first --attempts have all
finished queues attempts up to 8 into the same run (they fill the slots that free up while the longest tasks run).
The engine URLs are as the boxes see them: each box reaches the GPU host's ports 30000-30015 through the tunnels.
Standard library only (host python3), like grpo/dispatch.py.
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

try:
    from grpo.infra_filter import infra_cause
except ImportError:                                    # run as a script: grpo/ is on sys.path, not its parent
    from infra_filter import infra_cause


def sampling_params(max_tokens: int, temperature: float, top_p: float, thinking: bool) -> dict:
    """Terminus-2's per-call arguments (the agent server passes them as llm_call_kwargs). Thinking stays in the
    history: the agent server sets interleaved_thinking, and preserve_thinking makes Qwen3.6's template render it."""
    kwargs = {"enable_thinking": thinking, "preserve_thinking": thinking}
    return {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": kwargs}}


def plan(tasks: list[str], attempts: int, engines: list[str], minutes: dict[str, float], first: int = 0) -> list[dict]:
    """One entry per trial (attempts first..attempts-1), longest expected first; each goes to the engine with the
    fewest expected minutes so far."""
    default = statistics.mean(minutes.values()) if minutes else 1.0
    trials = sorted(((t, a) for t in tasks for a in range(first, attempts)), key=lambda ta: -minutes.get(ta[0], default))
    load = {e: 0.0 for e in engines}
    out = []
    for task, attempt in trials:
        engine = min(engines, key=lambda e: (load[e], engines.index(e)))
        load[engine] += minutes.get(task, default)
        out.append({"task": task, "attempt": attempt, "engine": engine})
    return out


def request_body(trial: dict, model: str, max_seq_len: int, params: dict) -> dict:
    return {"base_url": trial["engine"], "model": model, "api_key": "dummy", "instance_id": trial["task"],
            "agent_name": "terminus-2", "max_seq_len": max_seq_len, "sampling_params": params}


def post(url: str, body: dict, timeout: float | None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Runner:
    def __init__(self, dispatcher: str, out_path: str, timeout: float | None):
        self.dispatcher, self.out_path, self.timeout = dispatcher.rstrip("/"), out_path, timeout
        self.lock, self.rows = threading.Lock(), []

    def run_one(self, trial: dict, body: dict) -> dict:
        start = time.time()
        try:
            reply = post(f"{self.dispatcher}/run", body, self.timeout)
            if not isinstance(reply, dict):
                raise ValueError(f"reply is not an object: {reply!r:.100}")
        except Exception as e:                         # the dispatcher itself failed: not a policy outcome
            reply = {"reward": None, "exit_status": f"Error: Driver: {e!r}", "agent_metrics": {}}
        row = {**trial, "start": round(start, 3), "end": round(time.time(), 3), "reward": reply.get("reward"),
               "exit_status": reply.get("exit_status"), "agent_metrics": reply.get("agent_metrics", {})}
        with self.lock:
            self.rows.append(row)
            with open(self.out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            done = len(self.rows)
            passed = sum((r["reward"] or 0) > 0 for r in self.rows)
        print(f"{time.strftime('%H:%M:%S')} {done} done, {passed} passed: {trial['task']} #{trial['attempt']} "
              f"{row['exit_status']} reward={row['reward']}", flush=True)
        return row


EXTEND_POLL_SEC = 30


def extension_requested(path: str, futures: list) -> bool:
    """True as soon as the file exists; False once every future is done without it."""
    while not os.path.exists(path):
        if not wait(futures, timeout=EXTEND_POLL_SEC, return_when=ALL_COMPLETED).not_done:
            return os.path.exists(path)
    return True


def scored(row: dict) -> bool:
    """Whether the row counts as a policy outcome: the same infrastructure exclusions as GRPO (grpo/infra_filter.py)."""
    return infra_cause({"exit_status": row["exit_status"], "agent_metrics": row.get("agent_metrics") or {}}) is None


def summary(rows: list[dict]) -> dict:
    def rate(rs):
        rs = [r for r in rs if scored(r)]
        return round(sum((r["reward"] or 0) > 0 for r in rs) / len(rs), 4) if rs else None

    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    return {"trials": len(rows), "scored": sum(map(scored, rows)), "pass_rate": rate(rows),
            "per_task": {t: rate(rs) for t, rs in sorted(by_task.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dispatcher", default="http://127.0.0.1:11000")
    p.add_argument("--engines", required=True, help="OpenAI base URLs as the boxes see them, comma-separated")
    p.add_argument("--tasks", required=True, help="file with one task name per line")
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--minutes", help="JSON {task: expected minutes}, to spread trials across engines")
    p.add_argument("--model", default="openai/model", help="LiteLLM name; 'openai/' + SGLang's --served-model-name")
    p.add_argument("--max-seq-len", type=int, default=65536)
    p.add_argument("--max-tokens", type=int, default=16384, help="output cap per model call")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--timeout", type=float, default=None, help="seconds per trial at the driver (default: none)")
    p.add_argument("--out", required=True, help="JSONL, one row per finished trial (appended)")
    p.add_argument("--extend-to", type=int, default=0, help="attempts per task if --extend-file appears in time")
    p.add_argument("--extend-file", help="created (e.g. touch) before the first --attempts finish: run --extend-to")
    p.add_argument("--dry-run", action="store_true", help="print the plan and one request body, send nothing")
    a = p.parse_args(argv)
    if a.extend_to > a.attempts and not a.extend_file:
        p.error("--extend-to needs --extend-file")
    tasks = [t.strip() for t in open(a.tasks) if t.strip()]
    minutes = json.load(open(a.minutes)) if a.minutes else {}
    trials = plan(tasks, a.attempts, a.engines.split(","), minutes)
    extra = plan(tasks, a.extend_to, a.engines.split(","), minutes, first=a.attempts) if a.extend_to > a.attempts else []
    params = sampling_params(a.max_tokens, a.temperature, a.top_p, not a.no_thinking)
    if a.dry_run:
        per_engine = {e: sum(t["engine"] == e for t in trials) for e in a.engines.split(",")}
        print(json.dumps({"trials": len(trials), "extension_trials": len(extra), "per_engine": per_engine,
                          "first": request_body(trials[0], a.model, a.max_seq_len, params)}, indent=1))
        return 0
    runner = Runner(a.dispatcher, a.out, a.timeout)
    print(f"{len(trials)} trials ({len(tasks)} tasks x {a.attempts}) -> {a.dispatcher}", flush=True)
    with ThreadPoolExecutor(max_workers=len(trials) + len(extra)) as pool:   # the dispatcher queues every connection
        submit = lambda ts: [pool.submit(runner.run_one, t, request_body(t, a.model, a.max_seq_len, params)) for t in ts]
        futures = submit(trials)
        if extra and extension_requested(a.extend_file, futures):
            print(f"{a.extend_file} found: queuing attempts {a.attempts + 1}-{a.extend_to} ({len(extra)} trials)",
                  flush=True)
            futures += submit(extra)
    lost = [f.exception() for f in futures if f.exception()]
    for e in lost:                                     # e.g. a failed write: the row is missing from --out
        print(f"driver error, row not written: {e!r}", flush=True)
    print(json.dumps(summary(runner.rows)), flush=True)
    return 1 if lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
