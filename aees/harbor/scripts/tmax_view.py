"""Feeds the Tmax-experiment tracker (monitor_tmax/phase1.py, a copy of post_training/monitor/phase1.py) from plain
`harbor run` job folders mirrored to this Mac, e.g. ~/code/jobs_remote/aees-l3/tb21_l1114_i3/<task>__<id>/.

    python3 scripts/tmax_view.py --run l1114 --jobs "~/code/jobs_remote/aees-l*/tb21_l1114_i*" \
        --data ~/code/tmax_view --planned <N> [--attempts <per task>] --loop 60

Each pass (re)builds, under --data:
    trials/<run>/<task>__<job>__<id>   symlink to every trial folder found under the --jobs glob (the tracker parses
                                       result.json, agent/trajectory.json and trial.log through it)
    trials/<run>/rollouts.jsonl        one row per FINISHED trial (result.json has finished_at), in grpo/rollout_only.py's
                                       schema: task, attempt (per task, by start time), engine (the remote/box name:
                                       the job folder's parent), start, end (unix, whole trial), reward
                                       (verifier_result.rewards.reward), exit_status, agent_metrics
    monitor/<run>/settings.json        the run's settings, shown on the page
exit_status follows the agent server (harbor-miles agent_server/results.py): no exception and a verifier result ->
"Submitted"; AgentTimeoutError / VerifierTimeoutError -> "TimeLimitExceeded"; MaxSeqLenExceededError /
SingleTurnMaxSeqLenExceededError -> "SequenceLengthLimitExceeded"; any other exception -> "AgentError" (the type is kept
in agent_metrics.exception_type, so ContextLengthExceededError counts as a context stop). Exception: setup failures
before the agent's first step (EnvironmentStartTimeoutError, AgentSetupTimeoutError) become "Error: <type>", which
the tracker lists as infrastructure errors and does not score (grpo/infra_filter.py's SETUP_EXCEPTIONS).
A result.json is re-read only when its mtime changes. Only symlinks are ever removed. Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path

TIMEOUTS = {"AgentTimeoutError", "VerifierTimeoutError"}
OUTPUT_LIMITS = {"MaxSeqLenExceededError", "SingleTurnMaxSeqLenExceededError"}
SETUP = {"EnvironmentStartTimeoutError", "AgentSetupTimeoutError"}


def unix(ts) -> float | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def secs(phase) -> float | None:
    a, b = unix((phase or {}).get("started_at")), unix((phase or {}).get("finished_at"))
    return round(b - a, 3) if a is not None and b is not None else None


def exit_status(r: dict) -> str:
    exc = (r.get("exception_info") or {}).get("exception_type")
    if exc:
        if exc in SETUP:
            return f"Error: {exc}"
        if exc in TIMEOUTS:
            return "TimeLimitExceeded"
        if exc in OUTPUT_LIMITS:
            return "SequenceLengthLimitExceeded"
        return "AgentError"
    return "Submitted" if r.get("verifier_result") is not None else "Unknown"


def parse(folder: Path, engine: str, job: str) -> dict | None:
    """A finished trial's row (attempt filled in later); None while it runs or if result.json is half-written."""
    try:
        r = json.loads((folder / "result.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(r, dict) or not r.get("finished_at"):
        return None
    ar = r.get("agent_result") or {}
    meta = ar.get("metadata") or {}
    times = meta.get("api_request_times_msec") or []
    m = {k: ar.get(k) for k in ("n_input_tokens", "n_cache_tokens", "n_output_tokens", "cost_usd")}
    m.update(n_episodes=meta.get("n_episodes"), summarization_count=meta.get("summarization_count"),
             api_requests=len(times), api_request_s=round(sum(times) / 1000, 3) if times else None,
             exception_type=(r.get("exception_info") or {}).get("exception_type") or "",
             total_time=secs(r), env_setup_time=secs(r.get("environment_setup")),
             agent_setup_time=secs(r.get("agent_setup")), agent_run_time=secs(r.get("agent_execution")),
             eval_time=secs(r.get("verifier")))
    return {"task": str(r.get("task_name") or folder.name.split("__")[0]).split("/")[-1],
            "engine": engine, "job": job, "trial": folder.name,
            "start": unix(r.get("started_at")), "end": unix(r.get("finished_at")),
            "reward": ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exit_status": exit_status(r), "agent_metrics": m}


class View:
    def __init__(self, args):
        self.a = args
        self.root = args.data.expanduser() / "trials" / args.run
        self.cache: dict[Path, tuple[float, dict | None]] = {}

    def trial_folders(self) -> list[tuple[Path, str, str]]:
        out = []
        for job in sorted(glob.glob(os.path.expanduser(self.a.jobs))):
            job = Path(job)
            if job.is_dir():
                out += [(t, job.parent.name, job.name) for t in sorted(job.glob("*__*")) if t.is_dir()]
        return out

    def link_name(self, trial: Path, job: str) -> str:
        task, _, tid = trial.name.partition("__")
        return f"{task}__{job}__{tid}"                 # the tracker's <task>__<stamp>__<id>; unique across jobs

    def once(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        found = self.trial_folders()
        want = {self.link_name(t, job): t.resolve() for t, _, job in found}
        for p in self.root.iterdir():                  # stale links: trial gone or no longer under the glob
            if p.is_symlink() and want.get(p.name) != Path(os.readlink(p)):
                p.unlink()
        for name, target in want.items():
            if not (self.root / name).is_symlink():
                (self.root / name).symlink_to(target)
        rows = []
        for t, engine, job in found:
            try:
                mt = (t / "result.json").stat().st_mtime
            except OSError:
                mt = None
            hit = self.cache.get(t)
            if hit is None or hit[0] != mt:
                row = parse(t, engine, job) if mt is not None else None
                if row:
                    row["trial"] = self.link_name(t, job)
                hit = self.cache[t] = (mt, row)
            if hit[1]:
                rows.append(dict(hit[1]))
        by_task: dict[str, list[dict]] = {}
        for r in sorted(rows, key=lambda r: (r["start"] or 0, r["trial"])):
            r["attempt"] = len(by_task.setdefault(r["task"], []))
            by_task[r["task"]].append(r)
        rows.sort(key=lambda r: (r["end"] or 0, r["trial"]))
        write(self.root / "rollouts.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
        self.write_settings(found, by_task)
        return {"folders": len(found), "finished": len(rows), "tasks": len(by_task)}

    def write_settings(self, found, by_task):
        a = self.a
        s = {"model": a.model, "reasoning_effort": a.reasoning_effort, "context": a.context,
             "max_tokens": a.max_tokens, "max_calls": a.max_calls, "command_timeout_sec": a.command_timeout_sec,
             "tasks": a.tasks}
        if a.attempts:
            s["attempts"] = a.attempts                 # an int: the tracker shows "attempts planned"
        else:
            s["attempts_note"] = "per task as scheduled (not uniform or not given)"
        n = [len(rs) for rs in by_task.values()]
        s["attempts_done"] = f"{min(n)}-{max(n)} per task over {len(n)} tasks" if n else "none yet"
        s["planned"] = a.planned
        s["boxes"] = sorted({e for _, e, _ in found})
        s["source"] = "harbor run job folders, mirrored to this Mac"
        path = a.data.expanduser() / "monitor" / a.run / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, json.dumps(s, indent=1) + "\n")


def write(path: Path, text: str) -> None:
    """Atomic replace, and only when the content changed (keeps mtimes stable for the tracker)."""
    try:
        if path.read_text() == text:
            return
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", required=True, help='glob of job dirs, e.g. "~/code/jobs_remote/aees-l*/tb21_l1114_i*"')
    ap.add_argument("--data", type=Path, required=True, help="output root: trials/<run>/, monitor/<run>/")
    ap.add_argument("--planned", type=int, required=True, help="trials planned (for settings; pass the same to the tracker)")
    ap.add_argument("--attempts", type=int, default=0, help="attempts per task, if uniform")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-calls", type=int, default=64)
    ap.add_argument("--command-timeout-sec", type=int, default=120)
    ap.add_argument("--loop", type=float, default=0, help="seconds between passes; 0 = one pass")
    a = ap.parse_args(argv)
    v = View(a)
    while True:
        t = time.time()
        st = v.once()
        print(f"{time.strftime('%H:%M:%S')} {a.run}: {st['folders']} trial folders, {st['finished']} finished, "
              f"{st['tasks']} tasks ({time.time() - t:.1f} s)", flush=True)
        if not a.loop:
            return
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
