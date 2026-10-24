"""Unattended batch scheduler: run Harbor batches back-to-back, ingest each into the ledger.

    caffeinate -i python -m aees_harbor.schedule --plan state/schedule_overnight.json

The plan file lists batches; this process updates it in place, so the status page can show
planned / waiting / running / done batches live. Before every batch it checks:
  * a STOP file (state/STOP) -> stop cleanly, no new batch
  * the deadline             -> no new batch starts after it
  * the budget cap           -> no new batch once the AGGREGATE cost reaches it
  (aggregate = every trial in every batch of the plan + plan["budget_counts_jobs"])
  * the OpenAI API           -> a 4-token request; stop on auth/credit errors, retry transient ones
  * Docker and free disk
Budget kill switch: while a batch runs, a watchdog re-computes the aggregate cost every
`watchdog_interval_s` (default 60 s). If it reaches budget_usd, the running harbor job's whole
process group is cancelled (SIGINT, then SIGTERM/SIGKILL after `cancel_grace_s`, default 60 s),
no further batch starts, and the trials that did finish are ingested. The STOP file is graceful (never interrupts a batch).
Each finished batch is ingested (all trials held for review).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from aees_harbor.ingest import ingest_job
from aees_harbor.ledger import Ledger

HERE = Path(__file__).resolve().parent.parent  # aees/harbor


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def log(plan_path: Path, msg: str) -> None:
    line = f"{now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(plan_path.with_suffix(".log"), "a") as f:
        f.write(line + "\n")


def save(plan_path: Path, plan: dict) -> None:
    tmp = plan_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2))
    tmp.replace(plan_path)  # atomic, so the status page never reads a half-written file


def api_key(env_file: Path) -> str:
    for line in env_file.expanduser().read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"OPENAI_API_KEY not found in {env_file}")


def preflight_api(env_file: Path, model: str, tries: int = 3) -> tuple[bool, str]:
    """(ok, detail). ok=False means stop: auth/credit/model errors, or still failing after retries."""
    body = json.dumps({"model": model.split("/", 1)[-1], "max_completion_tokens": 4,
                       "messages": [{"role": "user", "content": "OK"}]}).encode()
    for attempt in range(1, tries + 1):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
            "Authorization": f"Bearer {api_key(env_file)}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return True, f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            detail = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            if e.code in (401, 403, 404) or "insufficient_quota" in detail or "no credits" in detail.lower():
                return False, detail  # account-level problem: retrying won't help
        except (urllib.error.URLError, TimeoutError) as e:
            detail = f"network: {e}"
        time.sleep(30 * attempt)
    return False, f"still failing after {tries} tries: {detail}"


def harbor_running() -> list[str]:
    """Command lines of running `harbor run` processes. Uses ps for the full command line:
    `pgrep -fl` prints it on macOS but only the process name on Linux (which hid running jobs on EC2)."""
    pids = subprocess.run(["pgrep", "-f", "harbor run"], capture_output=True, text=True).stdout.split()
    out = []
    for pid in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if "harbor run" in cmd and "pgrep" not in cmd:
            out.append(f"{pid} {cmd}")
    return out


def job_cost(job_dir: Path) -> float:
    total = 0.0
    for r in job_dir.glob("*__*/result.json"):
        try:
            total += (json.loads(r.read_text()).get("agent_result") or {}).get("cost_usd") or 0
        except json.JSONDecodeError:
            pass
    return total


def aggregate_cost(plan: dict) -> float:
    """Total cost of every trial in every planned batch plus the extra jobs counted toward the budget."""
    root = Path(plan["jobs_root"]).expanduser()
    names = [b["name"] for b in plan["batches"]] + list(plan.get("budget_counts_jobs", []))
    return sum(job_cost(root / n) for n in names if (root / n).exists())


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def cancel(proc: subprocess.Popen, plan_path: Path, grace_s: int) -> None:
    """Cancel a running harbor job and everything it started.

    harbor runs in its own process group (see Popen below). SIGINT goes to the whole group so
    harbor tears down its containers; after `grace_s` the group gets SIGTERM, then SIGKILL.
    (On 2026-10-18, SIGINT to harbor alone let in-flight trials keep running for the full
    300 s grace and overshoot the cap by $1.85.)
    """
    _signal_group(proc, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        log(plan_path, f"harbor still running {grace_s}s after SIGINT; sending SIGTERM to its process group")
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log(plan_path, "harbor still running after SIGTERM; sending SIGKILL to its process group")
        _signal_group(proc, signal.SIGKILL)
        proc.wait()


def docker_ok() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def ingest_into_ledger(plan: dict, job: Path) -> dict:
    """Ingest a finished job. Never raises: an ingest problem must not stop the overnight run
    (the job can always be re-ingested later from the browser)."""
    try:
        led = Ledger(HERE / plan["ledger"])
        try:
            return ingest_job(led, job)
        finally:
            led.close()
    except Exception as e:  # noqa: BLE001 - deliberately broad; logged and recorded on the batch
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def harbor_cmd(plan: dict, batch: dict) -> list[str]:
    tasks = [t.strip() for t in (HERE / plan["tasks_file"]).read_text().split() if t.strip()]
    cmd = ["harbor", "run", "--env-file", str(Path(plan["env_file"]).expanduser()),
           "--dataset", plan["dataset"], "--agent", plan["agent"], "-m", plan["model"],
           "--env", "docker", "-o", str(Path(plan["jobs_root"]).expanduser()), "--job-name", batch["name"],
           "-n", str(batch["concurrency"]), "-k", str(batch["attempts"]), "-r", "0", "-y"]
    for k, v in plan["agent_kwargs"].items():
        cmd += ["--ak", f"{k}={json.dumps(v)}"]
    for t in tasks:
        cmd += ["-i", f"{plan['task_prefix']}{t}"]
    return cmd


def run(plan_path: Path, dry_run: bool) -> None:
    plan = json.loads(plan_path.read_text())
    jobs_root = Path(plan["jobs_root"]).expanduser()
    deadline = dt.datetime.fromisoformat(plan["no_new_batch_after"]) if plan.get("no_new_batch_after") else None
    log(plan_path, f"scheduler started (dry_run={dry_run}); "
                   f"deadline {f'{deadline:%Y-%m-%d %H:%M}' if deadline else 'none'}, budget ${plan['budget_usd']}")
    for batch in plan["batches"]:
        if batch["status"] in ("done", "skipped", "stopped"):
            continue
        if batch["status"] == "external":  # a batch started outside the scheduler: wait, then ingest
            while not dry_run and any(batch["name"] in l for l in harbor_running()):
                time.sleep(60)
            job = jobs_root / batch["name"]
            counts = ingest_into_ledger(plan, job) if job.exists() and not dry_run else {}
            batch.update(status="done", finished_at=now().isoformat(), cost_usd=round(job_cost(job), 4), ingest=counts)
            save(plan_path, plan)
            log(plan_path, f"{batch['name']} (external) finished, ingest {counts}")
            continue
        batch["status"] = "waiting"
        save(plan_path, plan)
        while not dry_run and plan.get("wait_for_other_harbor", True) and harbor_running():
            time.sleep(60)
        spent = aggregate_cost(plan)
        reason = None
        if (HERE / "state" / "STOP").exists():
            reason = "STOP file present"
        elif deadline and now() > deadline:
            reason = f"past deadline {deadline:%Y-%m-%d %H:%M}"
        elif spent >= plan["budget_usd"]:
            reason = f"budget reached (${spent:.2f} of ${plan['budget_usd']})"
        elif shutil.disk_usage(jobs_root).free < plan["min_free_disk_gb"] * 1024 ** 3:
            reason = "low disk space"
        elif not docker_ok():
            reason = "docker not running"
        else:
            ok, detail = (True, "dry run") if dry_run else preflight_api(Path(plan["env_file"]), plan["model"])
            if not ok:
                reason = f"API preflight failed: {detail}"
        if reason:
            batch.update(status="stopped", note=reason)
            for later in plan["batches"]:
                if later["status"] == "planned":
                    later.update(status="skipped", note=reason)
            save(plan_path, plan)
            log(plan_path, f"stopping before {batch['name']}: {reason}")
            return
        batch.update(status="running", started_at=now().isoformat())
        save(plan_path, plan)
        log(plan_path, f"launching {batch['name']}: {batch['attempts']} attempts x tasks, -n {batch['concurrency']}")
        cmd = plan.get("command_override") or harbor_cmd(plan, batch)  # override is for tests only
        env = {**os.environ, "DOCKER_DEFAULT_PLATFORM": "linux/amd64"}
        killed = None
        with open(plan_path.with_name(f"{batch['name']}.harbor.log"), "w") as out:
            if dry_run:
                rc = 0
            else:
                proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=True)  # own process group, see cancel()
                while proc.poll() is None:  # budget watchdog
                    time.sleep(plan.get("watchdog_interval_s", 60))
                    spent = aggregate_cost(plan)
                    batch["live_aggregate_cost_usd"] = round(spent, 4)
                    save(plan_path, plan)
                    if spent >= plan["budget_usd"] and proc.poll() is None:
                        killed = f"budget kill switch: aggregate ${spent:.2f} >= ${plan['budget_usd']}"
                        log(plan_path, f"{batch['name']}: {killed}; cancelling harbor")
                        cancel(proc, plan_path, plan.get("cancel_grace_s", 60))
                rc = proc.returncode
        job = jobs_root / batch["name"]
        counts = ingest_into_ledger(plan, job) if job.exists() else {}
        batch.update(status="killed" if killed else "done" if rc == 0 else "failed", finished_at=now().isoformat(),
                     exit_code=rc, cost_usd=round(job_cost(job), 4) if job.exists() else 0, ingest=counts,
                     **({"note": killed} if killed else {}))
        save(plan_path, plan)
        log(plan_path, f"{batch['name']} finished rc={rc}, cost ${batch['cost_usd']}, ingest {counts}")
        if killed:
            for later in plan["batches"]:
                if later["status"] == "planned":
                    later.update(status="skipped", note=killed)
            save(plan_path, plan)
            log(plan_path, "stopped by budget kill switch")
            return
    log(plan_path, "all batches processed")


def main() -> None:
    p = argparse.ArgumentParser(description="Run planned Harbor batches back-to-back")
    p.add_argument("--plan", type=Path, default=HERE / "state" / "schedule_overnight.json")
    p.add_argument("--dry-run", action="store_true", help="check everything, launch nothing")
    a = p.parse_args()
    run(a.plan.resolve(), a.dry_run)


if __name__ == "__main__":
    main()
