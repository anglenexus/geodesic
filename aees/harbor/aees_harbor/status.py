"""Live run-status page: progress of Harbor jobs (running / queued / done), ETA, cost, host load.

    python -m aees_harbor.status --jobs-root ~/code/jobs --ledger state/ledger.sqlite
    # then open http://127.0.0.1:8766

Read-only. Reads job dirs directly (so running and queued trials show before ingestion) and the
ledger for overall progress toward the collection target. Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aees_harbor.browse import job_model, selected_tasks, test_candidates
from aees_harbor.ledger import Ledger
from aees_harbor.schedule import aggregate_cost

PAGE = Path(__file__).parent / "browser" / "status.html"
RECENT_JOBS = 6           # newest job dirs to report
THROUGHPUT_WINDOW_MIN = 20


def _ts(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class HostMonitor:
    """Samples `docker stats` and running harbor jobs in the background (each call takes ~2 s)."""

    def __init__(self, every_s: float = 15.0):
        self.snapshot: dict = {"sampled_at": None}
        threading.Thread(target=self._loop, args=(every_s,), daemon=True).start()

    def _loop(self, every_s: float) -> None:
        while True:
            self.snapshot = self.sample()
            time.sleep(every_s)

    @staticmethod
    def sample() -> dict:
        out = {"sampled_at": dt.datetime.now(dt.timezone.utc).isoformat(), "containers": 0,
               "cpu_pct": 0.0, "mem_gb": 0.0, "cpus": None, "mem_total_gb": None, "running_jobs": []}
        try:
            stats = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                                   capture_output=True, text=True, timeout=30).stdout.splitlines()
            for line in stats:
                s = json.loads(line)
                out["containers"] += 1
                out["cpu_pct"] += float(s["CPUPerc"].rstrip("%") or 0)
                used = s["MemUsage"].split("/")[0].strip()
                num, unit = float(used[:-3]), used[-3:]
                out["mem_gb"] += num / 1024 if unit == "MiB" else num if unit == "GiB" else num / 1024 ** 2
            info = json.loads(subprocess.run(["docker", "info", "--format", "{{json .}}"],
                                             capture_output=True, text=True, timeout=30).stdout or "{}")
            out["cpus"], out["mem_total_gb"] = info.get("NCPU"), (info.get("MemTotal") or 0) / 1024 ** 3
        except (subprocess.SubprocessError, OSError, ValueError, KeyError):
            out["error"] = "docker unavailable"
        ps = subprocess.run(["pgrep", "-fl", "harbor run"], capture_output=True, text=True).stdout
        out["running_jobs"] = sorted({tok.split("=", 1)[-1] for line in ps.splitlines()
                                      for tok in [line.split("--job-name")[-1].split()[0]]
                                      if "--job-name" in line})
        return out


def job_status(job: Path, now: dt.datetime) -> dict:
    """Totals, per-task cells, throughput and ETA for one Harbor job dir."""
    cfg = _json(job / "config.json") or {}
    tasks = [t for d in cfg.get("datasets", []) for t in (d.get("task_names") or [])]
    attempts = cfg.get("n_attempts") or 1
    cells = {t: [] for t in tasks}
    finished_at, cost, starts = [], 0.0, []
    for trial in job.iterdir():
        if not trial.is_dir() or "__" not in trial.name:
            continue
        r = _json(trial / "result.json") or {}
        task = r.get("task_name") or next((t for t in tasks if t.split("/")[-1] == trial.name.rsplit("__", 1)[0]), trial.name)
        if r.get("started_at"):
            starts.append(_ts(r["started_at"]))
        if r.get("finished_at"):
            rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            exc = (r.get("exception_info") or {}).get("exception_type")
            msg = ((r.get("exception_info") or {}).get("exception_message") or "").splitlines()[:1]
            state = "pass" if rew == 1.0 else "fail" if rew == 0.0 else "none"
            finished_at.append(_ts(r["finished_at"]))
            cost += (r.get("agent_result") or {}).get("cost_usd") or 0
            cells.setdefault(task, []).append({"s": state, "exc": exc, "msg": msg[0][:100] if msg else None,
                                               "trial": trial.name})
        else:
            cells.setdefault(task, []).append({"s": "running", "trial": trial.name})
    total = len(tasks) * attempts or sum(len(v) for v in cells.values())
    started = sum(len(v) for v in cells.values())
    done = len(finished_at)
    t0 = min(starts) if starts else None
    recent = [f for f in finished_at if (now - f).total_seconds() <= THROUGHPUT_WINDOW_MIN * 60]
    window = min(THROUGHPUT_WINDOW_MIN, (now - t0).total_seconds() / 60) if t0 else 0
    per_hour = len(recent) / window * 60 if window > 1 else None
    remaining = total - done
    count = lambda s: sum(c["s"] == s for v in cells.values() for c in v)
    return {
        "name": job.name, "model": job_model(job), "total": total, "attempts": attempts, "n_tasks": len(tasks),
        "concurrency": cfg.get("n_concurrent_trials"), "done": done, "running": started - done,
        "queued": max(0, total - started), "pass": count("pass"), "fail": count("fail"), "none": count("none"),
        "timeouts": sum(c.get("exc") == "AgentTimeoutError" for v in cells.values() for c in v),
        # exception type + first line of the message, most common first (raw facts, no classification)
        "exceptions": Counter(f"{c['exc']}: {c.get('msg') or ''}".rstrip(": ") for v in cells.values()
                              for c in v if c.get("exc")).most_common(),
        "cost_usd": cost, "started_at": t0.isoformat() if t0 else None,
        "last_finished_at": max(finished_at).isoformat() if finished_at else None,
        # finished jobs stop the clock at their last finished trial
        "elapsed_min": (((max(finished_at) if finished_at and started == done and not (total - started) else now)
                         - t0).total_seconds() / 60) if t0 else None,
        "throughput_per_hour": per_hour,
        "eta_min": remaining / per_hour * 60 if per_hour and remaining else (0 if not remaining else None),
        "cells": {t.split("/")[-1]: v for t, v in sorted(cells.items())},
    }


def _ledger_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    led = Ledger(path)
    try:
        scen = led.scenarios()
    finally:
        led.close()
    return {"scenarios": len(scen), "trials": sum(len(s["held"]) + len(s["passed"]) + len(s["failed"]) for s in scen)}


def remote_status(remotes_path: Path | None, now: dt.datetime) -> list[dict]:
    """Status of each EC2 remote, read from its local mirror (see aees_harbor.remote)."""
    if not remotes_path or not remotes_path.exists():
        return []
    from aees_harbor.remote import load_config, mirror_dirs
    cfg = load_config(remotes_path)
    out = []
    for r in cfg["remotes"]:
        jobs, state = mirror_dirs(cfg, r)
        plan = _json(state / Path(r.get("plan", "state/schedule_ec2.json")).name)
        dirs = sorted((d for d in jobs.iterdir() if (d / "config.json").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:RECENT_JOBS] if jobs.exists() else []
        if plan:
            for b in plan.get("batches", []):
                d = jobs / b["name"]
                if (d / "config.json").exists() and d not in dirs:
                    dirs.append(d)
        out.append({
            "config": {k: r.get(k) for k in ("name", "host", "region", "instance_id", "instance_type",
                                              "hourly_usd", "price_source", "instance")},
            "sync": _json(state / "sync.json") or {}, "host": _json(state / "host.json") or {},
            "schedule": plan, "aggregate_cost_usd": aggregate_cost({**plan, "jobs_root": str(jobs)}) if plan else None,
            "ledger": _ledger_summary(state / "ledger.sqlite"),
            "jobs": [job_status(d, now) for d in dirs],
        })
    return out


def status(jobs_root: Path, ledger_path: Path, host: dict, schedule_path: Path | None = None,
           remotes_path: Path | None = None) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    dirs = sorted((d for d in jobs_root.iterdir() if (d / "config.json").exists()),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:RECENT_JOBS]
    led = Ledger(ledger_path)
    try:
        scen = led.scenarios()
    finally:
        led.close()
    def totals(group: list[dict]) -> dict:
        return {"scenarios": len(group), "target": sum(s["target"] for s in group),
                "held": sum(len(s["held"]) for s in group), "passed": sum(len(s["passed"]) for s in group),
                "failed": sum(len(s["failed"]) for s in group)}
    overall = totals(scen)
    by_model = {m: totals([s for s in scen if s["fields"].get("model") == m])
                for m in sorted({s["fields"].get("model") for s in scen} - {None})}
    schedule = _json(schedule_path) if schedule_path else None
    if schedule:  # make sure every scheduled batch that has started gets a job card
        for b in schedule.get("batches", []):
            d = jobs_root / b["name"]
            if (d / "config.json").exists() and d not in dirs:
                dirs.append(d)
    agg = aggregate_cost(schedule) if schedule else None
    return {"now": now.isoformat(), "host": host, "overall": overall, "overall_by_model": by_model, "schedule": schedule, "aggregate_cost_usd": agg,
            "selected_tasks": selected_tasks(), "test_candidates": test_candidates(),
            "stop_file": bool(schedule_path and (schedule_path.parent / "STOP").exists()),
            "jobs": [job_status(d, now) for d in dirs], "remotes": remote_status(remotes_path, now)}


class Handler(BaseHTTPRequestHandler):
    jobs_root: Path
    ledger_path: Path
    schedule_path: Path
    remotes_path: Path | None
    monitor: HostMonitor

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.split("?", 1)[0] in ("/", "/index.html"):  # the page may carry ?model=...
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/api/status"):
            body = status(self.jobs_root, self.ledger_path, self.monitor.snapshot, self.schedule_path,
                          self.remotes_path)
            return self._send(200, json.dumps(body).encode(), "application/json")
        return self._send(404, b'{"error": "unknown endpoint"}', "application/json")

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Live status of Harbor collection runs")
    p.add_argument("--jobs-root", type=Path, default=Path("~/code/jobs"))
    p.add_argument("--ledger", type=Path, default=Path("state/ledger.sqlite"))
    p.add_argument("--schedule", type=Path, default=Path("state/schedule_overnight.json"))
    p.add_argument("--remotes", type=Path, default=Path("state/remotes.json"),
                   help="EC2 remotes config; their mirrors are shown too (see aees_harbor.remote)")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    Handler.schedule_path = args.schedule.resolve()
    Handler.remotes_path = args.remotes.resolve() if args.remotes.exists() else None
    Handler.jobs_root = args.jobs_root.expanduser().resolve()
    Handler.ledger_path = args.ledger.resolve()
    Handler.monitor = HostMonitor()
    print(f"Run status: http://127.0.0.1:{args.port}  (jobs: {Handler.jobs_root})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
