"""Mirror EC2 runner instances to this machine so the local web tools can track them.

    python -m aees_harbor.remote sync               # once
    python -m aees_harbor.remote sync --loop 300    # every 5 minutes

Reads state/remotes.json (not in git: hosts, key paths). For each remote it pulls, read-only:
  * job folders        -> <mirror_root>/<name>/           (rsync; terminal recordings skipped)
  * ledger snapshot    -> state/remote/<name>/ledger.sqlite (sqlite backup on the host, then copy)
  * schedule + its log -> state/remote/<name>/
  * host snapshot      -> state/remote/<name>/host.json  (docker CPU/memory, disk, running jobs)
  * EC2 state + cost   -> state/remote/<name>/sync.json  (uptime x hourly on-demand price)
It never writes to the remote host except a temporary ledger snapshot in /tmp.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor
CONFIG = HERE / "state" / "remotes.json"

HOST_PROBE = r'''
import json, subprocess, shutil, os
out = {"containers": 0, "cpu_pct": 0.0, "mem_gb": 0.0, "cpus": os.cpu_count(), "running_jobs": []}
for line in subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                           capture_output=True, text=True).stdout.splitlines():
    s = json.loads(line); out["containers"] += 1
    out["cpu_pct"] += float(s["CPUPerc"].rstrip("%") or 0)
    used = s["MemUsage"].split("/")[0].strip(); num, unit = float(used[:-3]), used[-3:]
    out["mem_gb"] += num / 1024 if unit == "MiB" else num if unit == "GiB" else num / 1024 ** 2
with open("/proc/meminfo") as f:
    out["mem_total_gb"] = int(f.readline().split()[1]) / 1024 ** 2
du = shutil.disk_usage(os.path.expanduser("~")); out["disk_free_gb"] = du.free / 1024 ** 3
ps = subprocess.run(["pgrep", "-fa", "harbor run|aees_harbor"], capture_output=True, text=True).stdout
out["running_jobs"] = sorted({l.split("--job-name")[1].split()[0] for l in ps.splitlines() if "--job-name" in l})
out["scheduler_alive"] = "aees_harbor.schedule" in ps
print(json.dumps(out))
'''


def load_config(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text())
    cfg.setdefault("mirror_root", "~/code/jobs_remote")
    cfg.setdefault("state_root", "state/remote")
    return cfg


def mirror_dirs(cfg: dict, r: dict) -> tuple[Path, Path]:
    """(local mirror of the remote jobs root, local state dir for this remote)."""
    jobs = Path(cfg["mirror_root"]).expanduser() / r["name"]
    state = (HERE / cfg["state_root"] / r["name"]).resolve()
    return jobs, state


def stopped_time(reason: str | None) -> dt.datetime | None:
    """Stop time from EC2's StateTransitionReason, e.g. 'User initiated (2026-10-19 19:42:00 GMT)'.
    Only covers the last launch -> stop; an instance stopped and restarted is billed from the restart."""
    m = re.search(r"\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) GMT\)", reason or "")
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc) if m else None


def _ssh(r: dict) -> list[str]:
    return ["-i", str(Path(r["key"]).expanduser()), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]


def _run(cmd: list[str], timeout: int = 600, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def ec2_status(r: dict) -> tuple[dict, str | None]:
    """EC2 state, uptime and compute cost from describe-instances: (ec2 dict, error or None)."""
    ec2 = {"hourly_usd": r.get("hourly_usd"), "price_source": r.get("price_source"), "instance_type": r.get("instance_type")}
    aws = _run(["aws", "ec2", "describe-instances", "--region", r["region"], "--instance-ids", r["instance_id"],
                "--query", "Reservations[0].Instances[0].[State.Name,LaunchTime,StateTransitionReason]",
                "--output", "json"], timeout=60)
    if aws.returncode == 0:
        st, launched, reason = json.loads(aws.stdout)
        end = dt.datetime.now(dt.timezone.utc)
        stopped_at = stopped_time(reason) if st in ("stopped", "stopping") else None
        if stopped_at:  # a stopped instance stops billing compute: freeze uptime at the stop time
            end = stopped_at
        hours = max(0.0, (end - dt.datetime.fromisoformat(launched)).total_seconds() / 3600)
        known = st == "running" or stopped_at is not None
        ec2.update(state=st, launch_time=launched, hours=round(hours, 3),
                   stopped_at=stopped_at.isoformat() if stopped_at else None,
                   cost_usd=round(hours * r["hourly_usd"], 2) if known and r.get("hourly_usd") else None)
    else:
        return ec2, f"aws: {aws.stderr.strip()[:200]}"
    return ec2, None


def sync_one(cfg: dict, r: dict) -> dict:
    jobs, state = mirror_dirs(cfg, r)
    jobs.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    target = f"{r['user']}@{r['host']}"
    ssh = ["ssh", *_ssh(r), target]
    errors: list[str] = []
    notes: list[str] = []

    ec2, err = ec2_status(r)
    if err:
        errors.append(err)
    if ec2.get("state") not in (None, "running"):  # stopped/terminated: the host is unreachable
        notes.append(f"instance {ec2['state']}: host not reachable, showing the last synced data")
    else:
        rs = _run(["rsync", "-az", "--exclude", "recording.cast", "--exclude", "terminus_2.pane",
                   "-e", "ssh " + " ".join(_ssh(r)), f"{target}:{r['remote_jobs']}/", f"{jobs}/"], timeout=1800)
        if rs.returncode not in (0, 24):  # 24 = files vanished during transfer (normal while running)
            errors.append(f"rsync jobs: {rs.stderr.strip()[:300]}")

        snap = (f"cd {r['remote_harbor']} && [ -f state/ledger.sqlite ] && .venv/bin/python -c "
                "\"import sqlite3; s=sqlite3.connect('state/ledger.sqlite'); d=sqlite3.connect('/tmp/aees_ledger_snapshot.sqlite'); "
                "s.backup(d); d.close()\"")
        if _run(ssh + [snap], timeout=120).returncode == 0:
            tmp = state / "ledger.sqlite.part"
            cp = _run(["scp", "-q", *_ssh(r), f"{target}:/tmp/aees_ledger_snapshot.sqlite", str(tmp)], timeout=600)
            if cp.returncode == 0:
                tmp.replace(state / "ledger.sqlite")
            else:
                errors.append(f"ledger copy: {cp.stderr.strip()[:200]}")
        else:
            notes.append("no ledger on host yet")

        plan = r.get("plan", "state/schedule_ec2.json")
        for f in (plan, plan.replace(".json", ".log")):
            cp = _run(["scp", "-q", *_ssh(r), f"{target}:{r['remote_harbor']}/{f}", str(state / Path(f).name)], timeout=120)
            if cp.returncode != 0 and f == plan:
                notes.append("no schedule on host yet")

        probe = _run(ssh + ["python3", "-"], input=HOST_PROBE, timeout=120)
        if probe.returncode == 0:
            host = json.loads(probe.stdout)
            host["sampled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (state / "host.json").write_text(json.dumps(host))
        else:
            errors.append(f"host probe: {probe.stderr.strip()[:200]}")

    result = {"name": r["name"], "synced_at": dt.datetime.now(dt.timezone.utc).isoformat(), "ok": not errors,
              "errors": errors, "notes": notes, "ec2": ec2}
    (state / "sync.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["sync"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--loop", type=int, default=0, help="repeat every N seconds")
    a = p.parse_args()
    while True:
        cfg = load_config(a.config)
        for r in cfg["remotes"]:
            try:
                res = sync_one(cfg, r)
                print(f"{res['synced_at']} {r['name']}: ok={res['ok']} errors={res['errors']}", flush=True)
            except Exception as e:  # noqa: BLE001 - keep looping; next sync may succeed
                print(f"{dt.datetime.now().isoformat()} {r['name']}: sync failed: {e}", flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
