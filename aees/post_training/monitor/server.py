"""Training tracker for one run on this box: reads what the run writes to disk and serves one page.

    RUN=grpo_pilot1 python3 monitor/server.py      # on the box
    bash scripts/tunnel.sh user@box                 # on the laptop, then open http://localhost:8770

Every 30 s it re-reads (read-only; the run is never touched):
    $DATA/ckpt/$RUN/events/*.jsonl                  Miles metrics: {"type": "metric", "timestamp", "metrics": {...}}
    $DATA/trials/$RUN/*/result.json                 Harbor trials of the training rollouts
    $DATA/ckpt/$RUN/iter_*/adapter                  saved checkpoints
    $DATA/trials/eval/$RUN/iter_N/{val,test}/*/result.json   evaluation of checkpoint N (Phase 4)
    $DATA/monitor/$RUN/preflight.jsonl              integration tests T0/T2/T3 (scripts/preflight.sh)
    $DATA/trials/$RUN/dispatch.jsonl, :11000/status EC2 sandboxes: one row per trial, live slots (grpo/dispatch.py)
    nvidia-smi, free, df, docker ps                 host health, appended to $DATA/monitor/$RUN/host.jsonl
and serves monitor/index.html with the result at /status.json, on 127.0.0.1 only. Standard library
only, so it runs with the host's python3 (3.10 or later), outside the Miles container.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # post_training/monitor
sys.path.insert(0, str(HERE.parent))                   # reuse post_training's own definitions:
from grpo.infra_filter import SETUP_EXCEPTIONS          # noqa: E402  infra = excluded from training
from grpo.split_tasks import LISTS, STATE, read_list    # noqa: E402  train / val / test task names
from sft.export_trajectories import PARSE_ERROR_PREFIX  # noqa: E402  terminus-2's reply to an unparseable turn

PORT = 8770
REFRESH_SEC = 30
DISK_ALERT_GB = 20                                     # shown red below this much free on $DATA
DISPATCH_PORT = 11000                                  # grpo/dispatch.py, when sandboxes run on EC2


class EventLog:
    """Miles' event files -> {step: {"t_rollout", "t_end", metric: value, ...}}, read incrementally.
    A metric line names its step in "rollout/step" or, in current Miles, a top-level "rollout_id"; other
    event types (audit records, often most of the file) are skipped without parsing."""

    def __init__(self, folder: Path):
        self.folder, self.offsets, self.steps = folder, {}, {}

    def refresh(self) -> dict[int, dict]:
        for path in sorted(self.folder.glob("**/*.jsonl")):
            with path.open("rb") as f:
                f.seek(self.offsets.get(path, 0))
                for line in f:
                    if not line.endswith(b"\n"):
                        break                          # half-written line: read it next time
                    self.offsets[path] = self.offsets.get(path, 0) + len(line)
                    if b'"type":"metric"' in line:
                        self._add(json.loads(line))
        return self.steps

    def _add(self, event: dict):
        m, t = event["metrics"], _unix(event["timestamp"])
        if (step := m.get("rollout/step", event.get("rollout_id"))) is None:   # Miles: top-level rollout_id
            return
        row = self.steps.setdefault(int(step), {})
        row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
        if any(k.startswith("rollout/") and k != "rollout/step" for k in m):
            row.setdefault("t_rollout", t)             # rollout of this step finished
        row["t_end"] = max(row.get("t_end", 0.0), t)   # last log of this step (training done)


def _unix(ts: str) -> float:
    """ISO time as Harbor and Miles write it ("2026-10-22T10:43:24.536536Z") -> unix seconds."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def trial_row(trial: Path) -> dict | None:
    """One Harbor trial folder -> {"task", "trial", "dir", "outcome", "reward", "turns", "tokens_in",
    "tokens_out", "parse_errors", "start", "end"} (times in unix seconds), or None while it's running.
    Outcome, first match wins (PLAN.md §8 item 6): infra (excluded from training) / passed (reward 1,
    even after a timeout) / timeout / context_limit (48k) / error (other exception or no reward) / failed."""
    try:
        r = json.loads((trial / "result.json").read_text())   # Harbor writes it once, when the trial ends
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not r.get("finished_at"):
        return None
    exc = (r.get("exception_info") or {}).get("exception_type", "")
    reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    outcome = ("infra" if exc in SETUP_EXCEPTIONS else "passed" if reward == 1 else
               "timeout" if exc == "AgentTimeoutError" else
               "context_limit" if exc == "ContextLengthExceededError" else
               "error" if exc or reward is None else "failed")
    agent = r.get("agent_result") or {}
    return {"task": r["task_name"].split("/")[-1], "trial": trial.name, "dir": str(trial),
            "outcome": outcome, "reward": reward, "turns": (agent.get("metadata") or {}).get("n_episodes"),
            "tokens_in": agent.get("n_input_tokens"), "tokens_out": agent.get("n_output_tokens"),
            "parse_errors": _parse_errors(trial), "start": _unix(r["started_at"]), "end": _unix(r["finished_at"])}


def _parse_errors(trial: Path) -> int:
    """Turns where terminus-2 couldn't parse the model's reply: its error notes in the trajectory."""
    try:
        return (trial / "agent" / "trajectory.json").read_text().count(PARSE_ERROR_PREFIX)
    except FileNotFoundError:
        return 0


class Trials:
    """Finished trials under one folder -> rows (block 3), each read once: a finished trial never changes.
    `pattern` matches result.json files, so running trials and stray files (logs) are never opened."""

    def __init__(self, root: Path, pattern: str):
        self.root, self.pattern, self.rows = root, pattern, {}

    def refresh(self) -> list[dict]:
        for result in self.root.glob(self.pattern):
            if result.parent not in self.rows and (row := trial_row(result.parent)) is not None:
                self.rows[result.parent] = row
        return list(self.rows.values())


def assign_steps(trials: list[dict], steps: dict[int, dict]) -> None:
    """Sets each training trial's "step": the first step whose rollout finished at or after the trial ended.
    Trials that ended after the last logged rollout belong to the step in progress."""
    ends = sorted((row["t_rollout"], step) for step, row in steps.items() if "t_rollout" in row)
    current = ends[-1][1] + 1 if ends else 0
    for t in trials:
        i = bisect.bisect_left(ends, (t["end"], -1))
        t["step"] = ends[i][1] if i < len(ends) else current


def label_evals(trials: list[dict]) -> None:
    """Eval trials live in .../iter_<N>/<val|test>/<trial>/: sets "checkpoint" (N) and "set"."""
    for t in trials:
        folder = Path(t["dir"])
        t["set"], t["checkpoint"] = folder.parent.name, int(folder.parent.parent.name.removeprefix("iter_"))


def checkpoints(save_dir: Path) -> list[dict]:
    """Saved adapters -> [{"step", "time"}]: step is the rollout id Miles saved at (iter_0000009 -> 9)."""
    return sorted(({"step": int(a.parent.name.removeprefix("iter_")), "time": a.stat().st_mtime}
                   for a in save_dir.glob("iter_*/adapter")), key=lambda c: c["step"])


class Host:
    """One host sample per refresh -> {"t", "gpu_mem_gb", "gpu_util", "ram_used_gb", "disk_free_gb",
    "containers"} (None where a tool is missing), appended to host.jsonl so a restart keeps the history."""

    def __init__(self, log: Path, data: Path):
        self.log, self.data = log, data
        log.parent.mkdir(parents=True, exist_ok=True)
        self.samples = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def sample(self) -> list[dict]:
        s = {"t": time.time(), "disk_free_gb": shutil.disk_usage(self.data).free / 1e9,
             "gpu_mem_gb": None, "gpu_util": None, "ram_used_gb": None, "containers": None}
        if gpu := _run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]):
            mem, util = gpu.splitlines()[0].split(",")
            s["gpu_mem_gb"], s["gpu_util"] = int(mem) / 1024, int(util)
        if Path("/proc/meminfo").exists():
            kb = {k: int(v.split()[0]) for k, v in (l.split(":") for l in Path("/proc/meminfo").read_text().splitlines())}
            s["ram_used_gb"] = (kb["MemTotal"] - kb["MemAvailable"]) / 1e6
        if (ps := _run(["docker", "ps", "-q"])) is not None:
            s["containers"] = len(ps.split())
        self.samples.append(s)
        with self.log.open("a") as f:
            f.write(json.dumps(s) + "\n")
        return self.samples


def jsonl_rows(path: Path) -> list[dict]:
    """Whole lines of a JSONL file that's still being appended to; [] if it doesn't exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().endswith("}")]


def dispatcher_status(port: int = DISPATCH_PORT) -> dict | None:
    """The dispatcher's live slots per box, or None when it isn't running (single-box runs)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def _run(cmd: list[str]) -> str | None:
    """A command's stdout, or None if it's missing, fails or takes over 10 s."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


class PriorRun:
    """The trained steps of an earlier run that this one continues (restarted from its adapter after a crash):
    its events (including the .trash_* folders Miles moves aside on restart) and their trials. Steps that never
    finished training, and the trials of aborted attempts, are left out."""

    def __init__(self, run: str, data: Path):
        self.save_dir = data / "ckpt" / run
        self.events = EventLog(self.save_dir)
        self.trials = Trials(data / "trials" / run, "*/result.json")

    def refresh(self) -> tuple[dict[int, dict], list[dict]]:
        steps, trials = self.events.refresh(), self.trials.refresh()
        assign_steps(trials, steps)
        trained = {k: v for k, v in steps.items() if any(m.startswith("train/") for m in v)}
        return trained, [t for t in trials if t["step"] in trained]


class Tracker:
    """Everything the page shows, rebuilt each refresh. `body` is the /status.json reply:
    {"run", "generated", "num_rollout", "steps": [{"step", "t_rollout", "t_end", <metric>: ...}],
     "trials": [block 3 row + "step"], "evals": [block 3 row + "checkpoint", "set"],
     "checkpoints": [{"step", "time"}], "host": [block 5 sample], "sets": {"train"|"val"|"test": [task]},
     "baseline": baseline.json or null, "disk_alert_gb", "preflight": [row], "dispatch": [row],
     "sandboxes": dispatcher /health or null}. Aggregates (rates, ETA, stall) are the page's job.
    With `prior_run`, that run's trained steps come first and this run's steps are numbered after them."""

    def __init__(self, run: str, data: Path, num_rollout: int, prior_run: str | None = None):
        self.run, self.num_rollout, self.save_dir = run, num_rollout, data / "ckpt" / run
        self.events = EventLog(self.save_dir / "events")
        self.train = Trials(data / "trials" / run, "*/result.json")
        self.evals = Trials(data / "trials" / "eval" / run, "iter_*/*/*/result.json")
        self.host = Host(data / "monitor" / run / "host.jsonl", data)
        self.preflight = data / "monitor" / run / "preflight.jsonl"
        self.dispatch = data / "trials" / run / "dispatch.jsonl"
        self.prior = PriorRun(prior_run, data) if prior_run else None
        self.sets = {name: read_list(STATE / file) for name, file in LISTS.items()}
        baseline = HERE / "baseline.json"                  # Phase 0 numbers, optional
        self.baseline = json.loads(baseline.read_text()) if baseline.exists() else None
        self.lock, self.body = threading.Lock(), b"{}"

    def refresh(self):
        steps, train, evals = self.events.refresh(), self.train.refresh(), self.evals.refresh()
        assign_steps(train, steps)
        label_evals(evals)
        ckpts = checkpoints(self.save_dir)
        if self.prior:
            prior_steps, prior_trials = self.prior.refresh()
            off = len(prior_steps)
            steps = {**prior_steps, **{k + off: v for k, v in steps.items()}}
            for t in train:
                t["step"] += off
            train = prior_trials + train
            ckpts = [c for c in checkpoints(self.prior.save_dir) if c["step"] < off] + \
                    [{**c, "step": c["step"] + off} for c in ckpts]
        body = json.dumps({
            "run": self.run, "generated": time.time(), "num_rollout": self.num_rollout,
            "steps": [{"step": k, **v} for k, v in sorted(steps.items())], "trials": train, "evals": evals,
            "checkpoints": ckpts, "host": self.host.sample(), "sets": self.sets,
            "baseline": self.baseline, "disk_alert_gb": DISK_ALERT_GB, "preflight": jsonl_rows(self.preflight),
            "dispatch": jsonl_rows(self.dispatch), "sandboxes": dispatcher_status()}).encode()
        with self.lock:
            self.body = body


def make_handler(tracker: Tracker):
    """Routes: / -> monitor/index.html, /status.json -> the latest reply, /download -> the same as a file."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                body, kind = (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
            elif path in ("/status.json", "/download"):
                with tracker.lock:
                    body, kind = tracker.body, "application/json"
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            if path == "/download":
                self.send_header("Content-Disposition", f'attachment; filename="{tracker.run}_status.json"')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass                                       # quiet: the page polls every 30 s

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.environ.get("RUN"), help="run name, = SAVE_DIR's last part (env RUN)")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA", "/data")))
    ap.add_argument("--num-rollout", type=int, default=int(os.environ.get("NUM_ROLLOUT", 50)))   # as configs/grpo.sh
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--prior-run", default=os.environ.get("PRIOR_RUN"),
                    help="earlier run this one continues: its trained steps are shown first (env PRIOR_RUN)")
    args = ap.parse_args()
    if not args.run:
        ap.error("set --run or RUN, e.g. RUN=grpo_pilot1")
    tracker = Tracker(args.run, args.data, args.num_rollout, args.prior_run)
    tracker.refresh()                                  # first reply ready before serving

    def loop():
        while True:
            time.sleep(REFRESH_SEC)
            try:
                tracker.refresh()
            except Exception as e:                     # one bad file must not stop the tracker;
                print(f"refresh failed: {e!r}", flush=True)   # the page shows the reply's age

    threading.Thread(target=loop, daemon=True).start()
    print(f"tracking {args.run}: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tracker)).serve_forever()


if __name__ == "__main__":
    main()
