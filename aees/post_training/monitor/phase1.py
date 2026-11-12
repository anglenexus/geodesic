"""Tracker for the rollout-only baseline (align_tmax_plan.md, Experiment 1): 20 tasks x 8 attempts of base
Qwen3.6-27B, thinking on, 40 at a time through grpo/dispatch.py, served by 4 SGLang engines. No training.

    RUN=<run> python3 monitor/phase1.py [--port 8771] [--data /data]     # on the GPU host
    bash scripts/tunnel.sh user@host                                       # then http://localhost:8771

Read on every request to /status.json (cheap: trial folders are re-parsed only when their files change):
    $DATA/trials/$RUN/rollouts.jsonl          grpo/rollout_only.py: one row per finished trial (task, attempt,
                                              engine, start, end, reward, exit_status, agent_metrics)
    $DATA/trials/$RUN/dispatch.jsonl          grpo/dispatch.py: one row per trial (box, queued, start, end, ...);
                                              fills in trials the driver hasn't written (yet)
    $DATA/trials/$RUN/<task>__<stamp>__<id>/  Harbor trial folders, synced from the boxes every 60 s:
                                              result.json and agent/trajectory.json (model calls, tokens,
                                              120 s command interrupts)
    monitor/phase1_reference.json             per-task references: Tmax's base model and our thinking-off run
Sampled every 10 s by a background thread:
    <dispatcher>/status                       live slots per box and the queue
    127.0.0.1:<port>/metrics                  each SGLang engine's Prometheus counters, gauges and latency
                                              histograms; every sample (with its derived tok/s, TTFT and ITL) is
                                              appended to $DATA/monitor/$RUN/serving.jsonl and read back at start
Serves monitor/phase1.html, the snapshot at /status.json and a one-paragraph /summary.txt, on 127.0.0.1 only.
finish.sh keeps the last /status.json; the page opens it with "Open snapshot". Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # post_training/monitor
PORT = 8771
PLANNED = 160                                          # 20 tasks x 8 attempts
SAMPLE_SEC = 10                                        # metrics and dispatcher sampled at most this often
HISTORY = 1080                                         # samples kept per engine: 3 h at 10 s
SERIES_STEP = 30                                       # seconds between points of the tok/s-over-time series
RECENT_SEC = 300                                       # window for "now" rates, cache hit and TTFT
MAX_CALLS = 64                                         # Tmax's step limit: a trial with this many calls was stopped
INTERRUPT = "Command timed out after 120 seconds and was interrupted"   # grpo/command_timeout.patch's note
TMAX_OVERALL = {"passed": 24, "n": 100}                # Tmax base Qwen3.6-27B on our 20 tasks (thinking on)
NOTHINK_OVERALL = {"passed": 117, "n": 480}            # auxr_1103_0033 steps 1-3 (thinking off)

# Prometheus names differ across SGLang versions: the first one an engine exports is used.
GEN_TOKENS = ("sglang:generation_tokens_total", "sglang:generation_tokens")
PROMPT_TOKENS = ("sglang:prompt_tokens_total", "sglang:prompt_tokens")
CACHED_TOKENS = ("sglang:cached_tokens_total", "sglang:cached_tokens")
CACHE_HIT_RATE = ("sglang:cache_hit_rate",)
RUNNING = ("sglang:num_running_reqs",)
QUEUED = ("sglang:num_queue_reqs",)
TTFT = ("sglang:time_to_first_token_seconds",)
ITL = ("sglang:inter_token_latency_seconds", "sglang:time_per_output_token_seconds")
E2E = ("sglang:e2e_request_latency_seconds",)
LATENCIES = {"ttft": TTFT, "itl": ITL, "e2e": E2E}
CMD_DURATION_CAP = 60                                  # Terminus-2 caps a command's requested wait at 60 s


# ---- the run's rows ----


def context_stop(row: dict) -> bool:
    """A trajectory ended by the 65k context: Harbor's length status, or a call the engine refused (AgentError with
    ContextLengthExceededError, when Terminus-2 doesn't stop at the context limit)."""
    exc = (row.get("agent_metrics") or {}).get("exception_type", "")
    return row.get("exit_status") == "SequenceLengthLimitExceeded" or exc == "ContextLengthExceededError"

def jsonl_rows(path: Path) -> list[dict]:
    """Whole lines of a JSONL file that's still being appended to; [] if it doesn't exist."""
    try:
        text = path.read_text()
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue                                   # half-written last line: read it next time
    return rows


def is_infra(row: dict) -> bool:
    """"Error: ..." exit statuses are the driver's, dispatcher's or agent server's failures, not the model's."""
    return str(row.get("exit_status") or "").startswith("Error")


def passed(row: dict) -> bool:
    return (row.get("reward") or 0) > 0               # as grpo/rollout_only.py's summary


def merge_rows(rollouts: list[dict], dispatch: list[dict]) -> list[dict]:
    """The driver's rows, plus dispatcher rows for trials the driver hasn't written: per task, dispatch rows
    beyond the driver's count (latest last) are added, marked "source": "dispatch"."""
    out = [{**r, "source": "rollouts"} for r in rollouts]
    have: dict[str, int] = {}
    for r in rollouts:
        have[r.get("task")] = have.get(r.get("task"), 0) + 1
    by_task: dict[str, list[dict]] = {}
    for r in dispatch:
        by_task.setdefault(r.get("task"), []).append(r)
    for task, rs in by_task.items():
        rs = sorted(rs, key=lambda r: r.get("end") or 0)
        out += [{**r, "source": "dispatch"} for r in rs[have.get(task, 0):]]
    return out


# ---- Harbor trial folders ----

def literal(x):
    """A value Harbor may have written as a Python repr string (str(dict), str(list)) -> the value; a plain
    string that isn't a repr stays a string; None on anything unparseable that should have been a container."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s or s[0] not in "[{(":
        return x
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _unix(ts) -> float | None:
    """ISO time as Harbor writes it ("2026-11-12T10:43:24.536536Z") -> unix seconds; None if missing or bad."""
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _duration(args) -> float:
    try:
        return min(CMD_DURATION_CAP, max(0.0, float(args.get("duration") or 0)))
    except (TypeError, ValueError):
        return 0.0


def trajectory_stats(steps: list, start: float | None = None) -> dict:
    """Agent steps of a trajectory -> {"calls", "prompt_tokens", "completion_tokens", "call_tokens" (output tokens
    of each call), "commands", "interrupts" (120 s per-command limit hits), "model_s", "cmd_s"}.
    Time split from the step timestamps: for each agent step, the gap since the previous step (or `start`, the agent's
    start) minus its commands' requested durations (each capped at 60 s) is model-call time; the durations are command
    time (Terminus-2 logs a step after its commands ran). model_s/cmd_s are None when no agent step has a usable gap."""
    calls, prompt, comp, call_tokens, commands, interrupts = 0, 0, 0, [], 0, 0
    model_s, cmd_s, timed, prev = 0.0, 0.0, False, start
    for s in steps if isinstance(steps, list) else []:
        if not isinstance(s, dict):
            continue
        obs = s.get("observation")
        if obs is not None:                            # the note can sit on any step that carries tool output
            interrupts += str(obs).count(INTERRUPT)
        ts = _unix(s.get("timestamp")) if s.get("timestamp") else None
        if s.get("source") != "agent":
            prev = max(prev or 0, ts or 0) or None     # time only moves forward from the agent's start
            continue
        calls += 1
        m = literal(s.get("metrics"))
        if isinstance(m, dict):
            p, c = m.get("prompt_tokens"), m.get("completion_tokens")
            prompt += p if isinstance(p, (int, float)) else 0
            if isinstance(c, (int, float)):
                comp += c
                call_tokens.append(c)
        tc, dur = literal(s.get("tool_calls")), 0.0
        if isinstance(tc, list):
            for t in tc:
                args = literal(t.get("arguments")) if isinstance(t, dict) else None
                if isinstance(args, dict):
                    commands += 1 if args.get("keystrokes") else 0
                    dur += _duration(args)
        if ts is not None and prev is not None and ts >= prev:
            model_s += max(0.0, ts - prev - dur)
            cmd_s += dur
            timed = True
        prev = max(prev or 0, ts or 0) or None
    return {"calls": calls, "prompt_tokens": prompt, "completion_tokens": comp, "call_tokens": call_tokens,
            "commands": commands, "interrupts": interrupts,
            "model_s": model_s if timed else None, "cmd_s": cmd_s if timed else None}


def trial_folder(folder: Path) -> dict:
    """One trial folder -> {"folder", "task", "finished", "reward", "exception", "start", "end", "agent_min",
    "other_s", **trajectory_stats}. other_s = agent time - model-call time - command time (setup, parsing, gaps).
    Missing or half-synced files give what's there (a running trial: finished False)."""
    row = {"folder": folder.name, "task": folder.name.split("__")[0], "finished": False, "reward": None,
           "exception": None, "start": None, "end": None, "agent_min": None}
    try:
        r = json.loads((folder / "result.json").read_text())
    except (OSError, ValueError):
        r = None
    if isinstance(r, dict):
        row["task"] = str(r.get("task_name") or row["task"]).split("/")[-1]
        row["exception"] = (r.get("exception_info") or {}).get("exception_type")
        row["reward"] = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ae = r.get("agent_execution") or {}
        row["start"], row["end"] = _unix(ae.get("started_at")), _unix(ae.get("finished_at"))
        if row["start"] and row["end"]:
            row["agent_min"] = (row["end"] - row["start"]) / 60
        row["finished"] = bool(r.get("finished_at") or r.get("verifier_result") or r.get("exception_info"))
    try:
        traj = json.loads((folder / "agent" / "trajectory.json").read_text())
    except (OSError, ValueError):
        traj = None
    row.update(trajectory_stats(traj.get("steps") if isinstance(traj, dict) else [], row["start"]))
    row["other_s"] = None
    if row["agent_min"] is not None and row["model_s"] is not None:
        row["other_s"] = max(0.0, row["agent_min"] * 60 - row["model_s"] - row["cmd_s"])
    return row


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


class Folders:
    """Trial folders under one run, each re-parsed only when result.json or trajectory.json changes."""

    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def refresh(self) -> list[dict]:
        rows = []
        for folder in sorted(self.root.glob("*__*")):
            if not folder.is_dir():
                continue
            key = (_mtime(folder / "result.json"), _mtime(folder / "agent" / "trajectory.json"))
            hit = self.cache.get(folder)
            if hit is None or hit[0] != key:
                hit = self.cache[folder] = (key, trial_folder(folder))
            rows.append(hit[1])
        return rows


# ---- statistics ----

def quantile(xs: list[float], q: float) -> float | None:
    """Linear-interpolated quantile; None for no data."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def describe(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": quantile(xs, 0.5), "mean": statistics.fmean(xs) if xs else None,
            "p90": quantile(xs, 0.9), "max": max(xs) if xs else None}


def rate(k: int, n: int) -> float | None:
    return k / n if n else None


# ---- the snapshot's parts ----

def progress(rows: list[dict], live: dict | None, planned: int, now: float) -> dict:
    """Done / running / waiting, elapsed since the first trial started, and a naive ETA (done so far per second)."""
    starts = [r["start"] for r in rows if r.get("start")]
    first = min(starts) if starts else None
    elapsed = now - first if first else None
    done = len(rows)
    boxes = (live or {}).get("boxes") or {}
    eta = None
    if elapsed and done and done < planned:
        eta = elapsed / done * (planned - done)
    return {"planned": planned, "done": done, "first_start": first, "elapsed": elapsed, "eta": eta,
            "last_end": max((r["end"] for r in rows if r.get("end")), default=None),
            "dispatcher": None if live is None else {
                "running": sum(b.get("running", 0) for b in boxes.values()),
                "free": sum(b.get("free", 0) for b in boxes.values()),
                "waiting": live.get("waiting"), "done": live.get("done"),
                "boxes": {k: {"running": b.get("running", 0), "free": b.get("free", 0)} for k, b in boxes.items()}}}


def outcomes(rows: list[dict]) -> dict:
    """Counts by exit status; "Error: ..." statuses are infrastructure errors, listed apart and not scored."""
    model, infra = {}, {}
    for r in rows:
        status = str(r.get("exit_status") or "unknown")
        side = infra if is_infra(r) else model
        side[status] = side.get(status, 0) + 1
    scored = [r for r in rows if not is_infra(r)]
    return {"by_status": dict(sorted(model.items(), key=lambda kv: -kv[1])),
            "infra": dict(sorted(infra.items(), key=lambda kv: -kv[1])),
            "scored": len(scored), "infra_total": sum(infra.values()),
            "passed": sum(passed(r) for r in scored),
            "passed_by_status": {s: sum(passed(r) for r in scored if str(r.get("exit_status")) == s) for s in model}}


def overall(rows: list[dict]) -> dict:
    scored = [r for r in rows if not is_infra(r)]
    k = sum(passed(r) for r in scored)
    timeouts = sum(r.get("exit_status") == "TimeLimitExceeded" for r in scored)
    seqlen = sum(map(context_stop, scored))
    return {"scored": len(scored), "passed": k, "pass_rate": rate(k, len(scored)),
            "timeouts": timeouts, "timeout_rate": rate(timeouts, len(scored)),
            "seqlen": seqlen, "seqlen_rate": rate(seqlen, len(scored)),
            "tmax": {**TMAX_OVERALL, "rate": TMAX_OVERALL["passed"] / TMAX_OVERALL["n"]},
            "ours_nothink": {**NOTHINK_OVERALL, "rate": NOTHINK_OVERALL["passed"] / NOTHINK_OVERALL["n"]}}


def used(f: dict) -> bool:
    """A finished trial folder in which the model was called (no call: the environment never came up)."""
    return f["finished"] and f["calls"] > 0


def per_task(rows: list[dict], folders: list[dict], reference: dict) -> list[dict]:
    """One row per reference task (reference order), then any other task seen. Pass counts come from the rows;
    agent behaviour (calls, interrupts, minutes) from the finished trial folders."""
    names = list(reference) + sorted({r.get("task") for r in rows if r.get("task") not in reference}
                                     | {f["task"] for f in folders if f["task"] not in reference})
    out = []
    for task in names:
        rs = [r for r in rows if r.get("task") == task]
        scored = [r for r in rs if not is_infra(r)]
        fs = [f for f in folders if f["task"] == task and used(f)]
        ref = reference.get(task) or {}
        k = sum(passed(r) for r in scored)
        out.append({
            "task": task, "limit_min": ref.get("limit_min"), "done": len(rs), "scored": len(scored),
            "passed": k, "pass_rate": rate(k, len(scored)), "infra": len(rs) - len(scored),
            "timeouts": sum(r.get("exit_status") == "TimeLimitExceeded" for r in scored),
            "seqlen": sum(map(context_stop, scored)),
            "results": [("pass" if passed(r) else "timeout" if r.get("exit_status") == "TimeLimitExceeded"
                         else "fail") for r in sorted(scored, key=lambda r: r.get("end") or 0)],
            "trials_parsed": len(fs), "stopped_64": sum(f["calls"] >= MAX_CALLS for f in fs),
            "calls_median": quantile([f["calls"] for f in fs], 0.5),
            "interrupts": sum(f["interrupts"] for f in fs),
            "agent_min_median": quantile([f["agent_min"] for f in fs], 0.5),
            "commands": sum(f["commands"] for f in fs), "calls_mean": describe([f["calls"] for f in fs])["mean"],
            "output_tokens_median": quantile([f["completion_tokens"] for f in fs], 0.5),
            **tok_rates(fs), "tmax": ref.get("tmax"), "ours_nothink": ref.get("ours_nothink")})
    return out


def tok_rates(fs: list[dict]) -> dict:
    """Per-agent output tokens/s over trials with timing: all output tokens / all agent time ("overall"), and
    / model-call time ("during calls"); plus the time split's shares of agent time."""
    timed = [f for f in fs if f.get("model_s") is not None and f.get("agent_min")]
    tokens = sum(f["completion_tokens"] for f in timed)
    agent = sum(f["agent_min"] * 60 for f in timed)
    model = sum(f["model_s"] for f in timed)
    cmd = sum(f["cmd_s"] for f in timed)
    return {"timed": len(timed), "tok_s_overall": tokens / agent if agent else None,
            "tok_s_calls": tokens / model if model else None,
            "share_model": model / agent if agent else None, "share_cmd": cmd / agent if agent else None,
            "share_other": max(0.0, 1 - (model + cmd) / agent) if agent else None}


def outcome_medians(fs: list[dict]) -> dict:
    """Median calls, commands, agent minutes and output tokens of one outcome's trajectories."""
    return {"n": len(fs), "calls": quantile([f["calls"] for f in fs], 0.5),
            "commands": quantile([f["commands"] for f in fs], 0.5),
            "agent_min": quantile([f["agent_min"] for f in fs], 0.5),
            "output_tokens": quantile([f["completion_tokens"] for f in fs], 0.5)}


def read_settings(path: Path | None) -> dict | None:
    """The run's settings (written by the launcher), or None if missing or unreadable."""
    try:
        d = json.loads(path.read_text()) if path else None
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def behaviour(folders: list[dict]) -> dict:
    """Agent behaviour over finished trials with at least one model call."""
    fs = [f for f in folders if used(f)]
    commands = sum(f["commands"] for f in fs)
    return {"trials": len(fs), "running": sum(not f["finished"] for f in folders), **tok_rates(fs),
            "model_min": describe([f["model_s"] / 60 for f in fs if f.get("model_s") is not None]),
            "cmd_min": describe([f["cmd_s"] / 60 for f in fs if f.get("cmd_s") is not None]),
            "other_min": describe([f["other_s"] / 60 for f in fs if f.get("other_s") is not None]),
            "commands_total": commands, "interrupt_rate_per_command": rate(sum(f["interrupts"] for f in fs), commands),
            "by_outcome": {k: outcome_medians([f for f in fs if ((f["reward"] or 0) > 0) == (k == "passed")])
                           for k in ("passed", "failed")},
            "calls": describe([f["calls"] for f in fs]),
            "commands": describe([f["commands"] for f in fs]),
            "output_tokens": describe([f["completion_tokens"] for f in fs]),
            "call_output_tokens": describe([c for f in fs for c in f["call_tokens"]]),
            "agent_min": describe([f["agent_min"] for f in fs]),
            "any_interrupt": rate(sum(f["interrupts"] > 0 for f in fs), len(fs)),
            "interrupts": sum(f["interrupts"] for f in fs),
            "hit_64": rate(sum(f["calls"] >= MAX_CALLS for f in fs), len(fs)),
            "per_trial": [{"task": f["task"], "calls": f["calls"], "output_tokens": f["completion_tokens"],
                           "interrupts": f["interrupts"], "agent_min": f["agent_min"], "commands": f["commands"],
                           "model_s": f.get("model_s"), "cmd_s": f.get("cmd_s"),
                           "passed": (f["reward"] or 0) > 0} for f in fs]}


# ---- Prometheus ----

def _labels(s: str) -> dict:
    """'a="x",b="y\\"z"' -> {"a": "x", "b": 'y"z'}."""
    out, i = {}, 0
    while i < len(s):
        eq = s.find("=", i)
        if eq < 0:
            break
        key = s[i:eq].strip().lstrip(",").strip()
        j, val = eq + 2, []                            # skip ="
        while j < len(s) and s[j] != '"':
            if s[j] == "\\" and j + 1 < len(s):
                j += 1
                val.append({"n": "\n"}.get(s[j], s[j]))
            else:
                val.append(s[j])
            j += 1
        out[key] = "".join(val)
        i = j + 1
    return out


def parse_prometheus(text: str) -> dict[str, list[tuple[dict, float]]]:
    """Prometheus text format -> {sample name: [(labels, value)]}. Comments and bad lines are skipped."""
    out: dict[str, list[tuple[dict, float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                labels, rest = rest.rsplit("}", 1)
                lab = _labels(labels)
            else:
                name, rest = line.split(None, 1)
                lab = {}
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        out.setdefault(name.strip(), []).append((lab, value))
    return out


def metric(m: dict, names: tuple[str, ...]) -> float | None:
    """Sum over label sets of the first of names present (e.g. one series per model or per TP rank)."""
    for n in names:
        if n in m:
            return sum(v for _, v in m[n])
    return None


def gauge_mean(m: dict, names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in m and m[n]:
            return statistics.fmean(v for _, v in m[n])
    return None


def histogram(m: dict, names: tuple[str, ...]) -> list[tuple[float, float]] | None:
    """A histogram's cumulative buckets [(le, count)], summed over label sets, sorted by le (+Inf last)."""
    for n in names:
        rows = m.get(n + "_bucket")
        if rows:
            b: dict[float, float] = {}
            for lab, v in rows:
                le = float(lab.get("le", "inf").replace("+Inf", "inf"))
                b[le] = b.get(le, 0.0) + v
            return sorted(b.items())
    return None


def hist_quantile(buckets: list[tuple[float, float]] | None, q: float) -> float | None:
    """Quantile from cumulative buckets, interpolated linearly inside the bucket (as Prometheus'
    histogram_quantile); the top finite bound if it falls in +Inf; None if empty."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    total, rank = buckets[-1][1], q * buckets[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= rank and total > 0:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


def bucket_delta(new: list | None, old: list | None) -> list | None:
    """Buckets of the observations between two samples (None if the bounds differ or the counter reset)."""
    if not new or not old or [le for le, _ in new] != [le for le, _ in old]:
        return None
    d = [(le, c - o) for (le, c), (_, o) in zip(new, old)]
    return None if any(c < 0 for _, c in d) else d


def engine_sample(text: str | None, t: float) -> dict:
    """One /metrics read -> the numbers the page uses; "up" False when the engine didn't answer."""
    if text is None:
        return {"t": t, "up": False}
    m = parse_prometheus(text)
    return {"t": t, "up": True, "gen_tokens": metric(m, GEN_TOKENS), "prompt_tokens": metric(m, PROMPT_TOKENS),
            "cached_tokens": metric(m, CACHED_TOKENS), "cache_hit_rate": gauge_mean(m, CACHE_HIT_RATE),
            "running": metric(m, RUNNING), "queued": metric(m, QUEUED),
            **{k: histogram(m, names) for k, names in LATENCIES.items()}}


def counter_rate(a: dict, b: dict, key: str) -> float | None:
    """Per-second increase of a counter from sample a to b; None if either is missing or it went down (restart)."""
    if not a.get("up") or not b.get("up") or a.get(key) is None or b.get(key) is None or b["t"] <= a["t"]:
        return None
    d = b[key] - a[key]
    return None if d < 0 else d / (b["t"] - a["t"])


def rate_series(samples: list[dict], key: str, step: float = SERIES_STEP) -> list[list]:
    """[[t, per-second rate]] between samples at least `step` seconds apart (None where the engine was down)."""
    out, prev = [], None
    for s in samples:
        if prev is None:
            prev = s
            continue
        if s["t"] - prev["t"] >= step:
            r = counter_rate(prev, s, key)
            out.append([s["t"], None if r is None else round(r, 2)])
            prev = s
    return out


def _back(samples: list[dict], window: float) -> dict | None:
    """The oldest up sample within `window` seconds of the last one."""
    if not samples:
        return None
    t = samples[-1]["t"] - window
    return next((s for s in samples if s["t"] >= t and s.get("up")), None)


def latency(last: dict, old: dict | None, key: str) -> dict:
    """p50/p90 of one latency histogram over the last RECENT_SEC (the bucket increase since `old`) and since the
    engine started; None where there are no observations."""
    allb = last.get(key) if last.get("up") else None
    return quants(bucket_delta(allb, old.get(key)) if old is not None and old is not last else None, allb)


def quants(recent: list | None, allb: list | None) -> dict:
    q = lambda b, x: hist_quantile(b, x) if b and b[-1][1] > 0 else None
    return {"p50": q(recent, 0.5), "p90": q(recent, 0.9), "n": recent[-1][1] if recent else 0,
            "p50_all": q(allb, 0.5), "p90_all": q(allb, 0.9), "n_all": allb[-1][1] if allb else 0}


def derive(samples: list[dict]) -> dict:
    """The derived numbers of the last sample, logged with it: tok/s since the previous sample and over RECENT_SEC,
    cache hit over RECENT_SEC, TTFT/ITL/E2E quantiles."""
    last = samples[-1]
    if not last.get("up"):
        return {}
    old = _back(samples, RECENT_SEC)
    prev = samples[-2] if len(samples) > 1 else None
    d = {"tok_s": counter_rate(prev, last, "gen_tokens") if prev else None,
         "tok_s_5m": counter_rate(old, last, "gen_tokens") if old is not None and old is not last else None,
         "cache_hit": None}
    if old is not None and old is not last:
        p, c = counter_rate(old, last, "prompt_tokens"), counter_rate(old, last, "cached_tokens")
        d["cache_hit"] = c / p if p and c is not None else None
    for k in LATENCIES:
        lat = latency(last, old, k)
        d[k + "_p50"], d[k + "_p90"] = lat["p50"], lat["p90"]
    return d


def engine_summary(port: int, samples: list[dict]) -> dict:
    """Current state of one engine from its sample history: running/queued now, output tok/s and prefix-cache hit
    over the last RECENT_SEC, latency quantiles (last 5 min and since start), and series over time."""
    last = samples[-1] if samples else None
    out = {"port": port, "up": bool(last and last.get("up")), "running": None, "queued": None, "tok_s": None,
           "cache_hit": None, "cache_hit_source": None, "gen_tokens": None,
           "series": rate_series(samples, "gen_tokens"), "latency_series": latency_series(samples),
           **{k: latency({"up": False}, None, k) for k in LATENCIES}}
    if not out["up"]:
        return out
    out.update(running=last.get("running"), queued=last.get("queued"), gen_tokens=last.get("gen_tokens"))
    old = _back(samples, RECENT_SEC)
    if old is not None and old is not last:
        out["tok_s"] = counter_rate(old, last, "gen_tokens")
        p, c = counter_rate(old, last, "prompt_tokens"), counter_rate(old, last, "cached_tokens")
        if p and c is not None:
            out["cache_hit"], out["cache_hit_source"] = c / p, "cached/prompt tokens, last 5 min"
    if out["cache_hit"] is None and last.get("prompt_tokens") and last.get("cached_tokens") is not None:
        out["cache_hit"], out["cache_hit_source"] = last["cached_tokens"] / last["prompt_tokens"], \
            "cached/prompt tokens, since engine start"
    if out["cache_hit"] is None and last.get("cache_hit_rate") is not None:
        out["cache_hit"], out["cache_hit_source"] = last["cache_hit_rate"], "engine's cache_hit_rate gauge"
    for k in LATENCIES:
        out[k] = latency(last, old, k)
    return out


def latency_series(samples: list[dict], step: float = SERIES_STEP) -> list[list]:
    """[[t, ttft_p50, ttft_p90, itl_p50, itl_p90]] from the logged derived values, one point per `step` seconds."""
    out, t = [], None
    for s in samples:
        d = s.get("d") or {}
        if not d or (t is not None and s["t"] - t < step):
            continue
        t = s["t"]
        out.append([t, d.get("ttft_p50"), d.get("ttft_p90"), d.get("itl_p50"), d.get("itl_p90")])
    return out


def combined_latency(history: dict[int, list[dict]], key: str) -> dict:
    """One latency histogram summed over the engines (those with the same bucket bounds as the first)."""
    recent, allb = None, None
    for samples in history.values():
        if not samples or not samples[-1].get("up") or not samples[-1].get(key):
            continue
        last, old = samples[-1], _back(samples, RECENT_SEC)
        b = last[key]
        allb = b if allb is None else _add(allb, b)
        r = bucket_delta(b, old.get(key)) if old is not None and old is not last else None
        if r:
            recent = r if recent is None else _add(recent, r)
    return quants(recent, allb)


def _add(a: list, b: list) -> list:
    return [(le, c + d) for (le, c), (_, d) in zip(a, b)] if [x for x, _ in a] == [x for x, _ in b] else a


def serving(history: dict[int, list[dict]]) -> dict:
    engines = [engine_summary(port, list(samples)) for port, samples in history.items()]
    up = [e for e in engines if e["up"]]
    total = lambda k: sum(e[k] for e in up if e[k] is not None) if any(e[k] is not None for e in up) else None
    return {"engines": engines, "up": len(up), "running": total("running"), "queued": total("queued"),
            "tok_s": total("tok_s"), "gen_tokens": total("gen_tokens"),
            **{k: combined_latency(history, k) for k in LATENCIES},
            "sampled": max((s[-1]["t"] for s in history.values() if s), default=None)}


# ---- live sources ----

def fetch(url: str, timeout: float = 3) -> str | None:
    """A URL's body, or None when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None


def encode_sample(port: int, s: dict) -> str:
    """One serving.jsonl line; histogram bounds as strings so +Inf stays valid JSON."""
    out = {"port": port, **s}
    for k in LATENCIES:
        if out.get(k):
            out[k] = [["+Inf" if math.isinf(le) else le, c] for le, c in out[k]]
    return json.dumps(out)


def decode_sample(line: str) -> tuple[int, dict] | None:
    try:
        s = json.loads(line)
        port = int(s.pop("port"))
        for k in LATENCIES:
            if s.get(k):
                s[k] = [(float("inf") if le == "+Inf" else float(le), float(c)) for le, c in s[k]]
        float(s["t"])
        return port, s
    except (ValueError, KeyError, TypeError):
        return None


class Live:
    """Background sampler: each engine's /metrics and the dispatcher's /status every SAMPLE_SEC. With `log`, each
    engine sample is appended there (with its derived values) and the file's samples are loaded at start."""

    def __init__(self, ports: list[int], dispatcher: str, log: Path | None = None):
        self.ports, self.dispatcher, self.log, self.lock = ports, dispatcher.rstrip("/"), log, threading.Lock()
        self.history = {p: deque(maxlen=HISTORY) for p in ports}
        self.status, self.status_t = None, None
        if log is not None and log.exists():
            for line in log.read_text().splitlines():
                if (ps := decode_sample(line)) and ps[0] in self.history:
                    self.history[ps[0]].append(ps[1])

    def sample(self) -> None:
        t = time.time()
        reads = {p: engine_sample(fetch(f"http://127.0.0.1:{p}/metrics"), t) for p in self.ports}
        body = fetch(f"{self.dispatcher}/status")
        try:
            status = json.loads(body) if body else None
        except ValueError:
            status = None
        lines = []
        with self.lock:
            for p, s in reads.items():
                self.history[p].append(s)
                s["d"] = derive(list(self.history[p])[-HISTORY:])
                lines.append(encode_sample(p, s))
            self.status, self.status_t = (status if isinstance(status, dict) else None), t
        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a") as f:
                f.write("".join(x + "\n" for x in lines))

    def loop(self) -> None:
        while True:
            try:
                self.sample()
            except Exception:                          # never let the sampler die; the page shows "no data"
                pass
            time.sleep(SAMPLE_SEC)

    def view(self) -> tuple[dict, dict | None, float | None]:
        with self.lock:
            return {p: list(h) for p, h in self.history.items()}, self.status, self.status_t


class Tracker:
    def __init__(self, run: str, data: Path, live: Live, planned: int = PLANNED, reference: Path | None = None,
                 settings: Path | None = None):
        self.run, self.live, self.planned = run, live, planned
        self.settings = settings or data / "monitor" / run / "settings.json"
        self.root = data / "trials" / run
        self.folders = Folders(self.root)
        ref = json.loads((reference or HERE / "phase1_reference.json").read_text())
        self.reference, self.reference_overall = ref["tasks"], ref.get("overall") or {}
        self.lock = threading.Lock()                   # one snapshot at a time: Folders' cache isn't thread-safe

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            rollouts, dispatch = jsonl_rows(self.root / "rollouts.jsonl"), jsonl_rows(self.root / "dispatch.jsonl")
            rows = merge_rows(rollouts, dispatch)
            folders = self.folders.refresh()
            history, status, status_t = self.live.view()
            return {"run": self.run, "generated": now, "planned": self.planned,
                    "sources": {"rollouts": len(rollouts), "dispatch": len(dispatch), "folders": len(folders),
                                "dispatcher_t": status_t},
                    "progress": progress(rows, status, self.planned, now), "outcomes": outcomes(rows),
                    "overall": overall(rows), "tasks": per_task(rows, folders, self.reference),
                    "behaviour": behaviour(folders), "serving": serving(history),
                    "reference": self.reference_overall, "settings": read_settings(self.settings)}


def summary_text(s: dict) -> str:
    """One paragraph for progress reports (/summary.txt)."""
    f = lambda v, d=0: "n/a" if v is None else f"{v:,.{d}f}"
    pc = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    ms = lambda v: "n/a" if v is None else f"{1000 * v:.0f} ms" if v < 1 else f"{v:.2f} s"
    p, o, b, sv = s["progress"], s["overall"], s["behaviour"], s["serving"]
    tm = (s.get("reference") or {}).get("tmax") or {}
    d = p.get("dispatcher")
    live = f"{d['running']} running, {d['waiting']} waiting" if d else "dispatcher not answering"
    return (f"{s['run']}: {p['done']}/{p['planned']} trials done ({live}; {o['scored']} scored, "
            f"{s['outcomes']['infra_total']} infra errors). Pass rate {pc(o['pass_rate'])} ({o['passed']}/{o['scored']}) "
            f"vs Tmax {pc(tm.get('pass_rate', o['tmax']['rate']))}, thinking-off {pc(o['ours_nothink']['rate'])}. "
            f"Timeouts {pc(o['timeout_rate'])} (Tmax {pc(tm.get('timeout_rate'))}), context limit {o['seqlen']}, "
            f"64-call stops {pc(b['hit_64'])}, 120 s interrupts {b['interrupts']}. "
            f"Median model calls {f(b['calls']['median'], 1)} (Tmax {f((tm.get('steps') or {}).get('median'), 1)}), "
            f"median agent min {f(b['agent_min']['median'], 1)} (Tmax {f((tm.get('agent_min') or {}).get('median'), 1)}). "
            f"Per-agent output tok/s during calls {f(b['tok_s_calls'], 1)} (Tmax {f(tm.get('tok_s_per_agent_during_calls'), 1)}), "
            f"overall {f(b['tok_s_overall'], 1)} (Tmax {f(tm.get('tok_s_per_agent_overall'), 1)}). "
            f"Serving (last 5 min): TTFT p50/p90 {ms(sv['ttft']['p50'])}/{ms(sv['ttft']['p90'])}, "
            f"ITL p50/p90 {ms(sv['itl']['p50'])}/{ms(sv['itl']['p90'])}, total output {f(sv['tok_s'])} tok/s, "
            f"{sv['up']}/{len(sv['engines'])} engines up.\n")


def make_handler(tracker: Tracker):
    """Routes: / -> monitor/phase1.html, /status.json -> a fresh snapshot, /summary.txt -> one paragraph of it."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                body, kind = (HERE / "phase1.html").read_bytes(), "text/html; charset=utf-8"
            elif path == "/status.json":
                body, kind = json.dumps(tracker.snapshot()).encode(), "application/json"
            elif path == "/summary.txt":
                body, kind = summary_text(tracker.snapshot()).encode(), "text/plain; charset=utf-8"
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass                                       # quiet: the page polls every 15 s

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.environ.get("RUN"), help="run name: $DATA/trials/<run> (env RUN)")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("DATA", "/data")))
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--planned", type=int, default=PLANNED, help="trials in the run (20 tasks x 8)")
    ap.add_argument("--engine-ports", default="30001,30002,30003,30004", help="SGLang ports on this host")
    ap.add_argument("--dispatcher", default="http://127.0.0.1:11000")
    ap.add_argument("--reference", type=Path, default=HERE / "phase1_reference.json")
    ap.add_argument("--settings-json", type=Path, help="run settings shown on the page "
                    "(default $DATA/monitor/$RUN/settings.json)")
    args = ap.parse_args(argv)
    if not args.run:
        ap.error("set --run or RUN")
    live = Live([int(p) for p in args.engine_ports.split(",") if p.strip()], args.dispatcher,
                args.data / "monitor" / args.run / "serving.jsonl")
    threading.Thread(target=live.loop, daemon=True).start()
    tracker = Tracker(args.run, args.data, live, args.planned, args.reference, args.settings_json)
    print(f"phase 1 tracker for {args.run}: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tracker)).serve_forever()


if __name__ == "__main__":
    main()
