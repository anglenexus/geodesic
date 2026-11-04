"""From the laptop: the spend cap and kill switch for an unattended run (auxiliary reward, 2026-11-02: hard cap $180).

    nohup caffeinate -i python3 scripts/budget_guard.py --hard 180 --soft 165 --soft-cmd '<cmd>' &

Every --poll seconds it lists what is billing, straight from the clouds: Nebius instances in both projects (H100s in
eu-north1, the H200 in us-central1) and running EC2 instances tagged Project=aees in us-east-2. Each instance's cost
is its hourly rate × the time since the cloud's own creation/launch time, until it is no longer listed; the ledger
(state/budget_ledger.json) keeps ended instances, so the total only grows. A failed listing counts every instance
it last saw as still running. A margin (--margin, 5%) covers disks, egress and S3.

- total ≥ soft: runs --soft-cmd once (the orderly stop: back up, then tear down).
- total ≥ hard: deletes every Nebius instance in the projects and terminates every aees EC2 instance, verifies
  nothing is left, writes state/budget_killed and exits.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NEBIUS_PROJECTS = ("project-e00qzjr2pr00d8qnh3d9xa", "project-u00wkf1akc00c0hjpbz059")   # eu-north1, us-central1
NEBIUS_RATES = {"gpu-h100-sxm": 4.50, "gpu-h200-sxm": 5.40}                             # $/GPU-h, list price
EC2_RATES = {"m7a.8xlarge": 1.85}                                                       # $/h, on demand
EC2_REGION = "us-east-2"
UNKNOWN_RATE = 10.0                                                                     # anything else: assume costly


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def nebius_cmd() -> list[str]:
    exe = "nebius" if subprocess.run(["which", "nebius"], capture_output=True).returncode == 0 \
        else str(Path.home() / ".nebius/bin/nebius")
    return [exe, "--profile", "aees"]


def nebius_live() -> dict[str, dict] | None:
    """id → {rate, start} for every Nebius instance in the projects; None if a listing failed."""
    out = {}
    for project in NEBIUS_PROJECTS:
        r = subprocess.run(nebius_cmd() + ["compute", "instance", "list", "--parent-id", project, "--format", "json"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        for item in json.loads(r.stdout or "{}").get("items", []):
            res = item.get("spec", {}).get("resources", {})
            gpus = int(str(res.get("preset", "1gpu")).split("gpu")[0] or 1) if "gpu" in str(res.get("preset", "")) else 1
            rate = NEBIUS_RATES.get(res.get("platform"), UNKNOWN_RATE) * gpus
            out[item["metadata"]["id"]] = {"rate": rate, "start": epoch(item["metadata"]["created_at"]),
                                           "cloud": "nebius", "project": project}
    return out


def ec2_live() -> dict[str, dict] | None:
    r = subprocess.run(["aws", "ec2", "describe-instances", "--region", EC2_REGION, "--filters",
                        "Name=tag:Project,Values=aees", "Name=instance-state-name,Values=pending,running,stopping",
                        "--query", "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]", "--output", "json"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    return {i: {"rate": EC2_RATES.get(t, UNKNOWN_RATE), "start": epoch(launch), "cloud": "ec2"}
            for i, t, launch in json.loads(r.stdout or "[]")}


def update(ledger: dict, cloud: str, live: dict | None, now: float) -> None:
    """Record what `cloud` lists now. Listed instances are open until now; ones not listed any more are closed at
    their last sighting. If the listing failed (None), every open instance of that cloud stays open until now."""
    for iid, info in (live or {}).items():
        entry = ledger.setdefault(iid, {**info, "end": None})
        entry["last_seen"], entry["end"] = now, None
    for iid, entry in ledger.items():
        if entry["cloud"] != cloud or entry["end"] is not None:
            continue
        if live is None:
            entry["last_seen"] = now
        elif iid not in live:
            entry["end"] = entry["last_seen"]


def spend(ledger: dict, margin: float) -> float:
    return (1 + margin) * sum(e["rate"] * max(0.0, e["last_seen"] - e["start"]) / 3600 for e in ledger.values())


def kill(ledger: dict) -> bool:
    """Delete every open instance; True when neither cloud lists anything afterwards."""
    for iid, e in ledger.items():
        if e["end"] is None and e["cloud"] == "nebius":
            subprocess.run(nebius_cmd() + ["compute", "instance", "delete", "--id", iid], capture_output=True, timeout=600)
    ids = [i for i, e in ledger.items() if e["end"] is None and e["cloud"] == "ec2"]
    if ids:
        subprocess.run(["aws", "ec2", "terminate-instances", "--region", EC2_REGION, "--instance-ids", *ids],
                       capture_output=True, timeout=120)
    left_n, left_e = nebius_live(), ec2_live()
    return left_n == {} and left_e == {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hard", type=float, default=180.0)
    p.add_argument("--soft", type=float, default=165.0)
    p.add_argument("--soft-cmd", default="")
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--state", type=Path, default=Path(__file__).resolve().parent.parent / "state")
    p.add_argument("--once", action="store_true", help="one poll, then exit (tests)")
    args = p.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    path = args.state / "budget_ledger.json"
    ledger = json.loads(path.read_text()) if path.exists() else {}
    soft_done = (args.state / "budget_soft_done").exists()
    while True:
        now = time.time()
        update(ledger, "nebius", nebius_live(), now)
        update(ledger, "ec2", ec2_live(), now)
        total = spend(ledger, args.margin)
        path.write_text(json.dumps(ledger, indent=1))
        live = sum(e["end"] is None for e in ledger.values())
        rate = (1 + args.margin) * sum(e["rate"] for e in ledger.values() if e["end"] is None)
        print(f"{datetime.now():%H:%M:%S} spend ${total:.2f} (cap ${args.hard:.0f}), {live} live, ${rate:.2f}/h", flush=True)
        (args.state / "budget_status").write_text(f"{total:.2f} {args.hard:.0f} {live} {rate:.2f} {int(now)}\n")
        if total >= args.soft and not soft_done and args.soft_cmd:
            print(f"SOFT CAP ${args.soft:.0f}: {args.soft_cmd}", flush=True)
            (args.state / "budget_soft_done").write_text(f"{int(now)}\n")
            soft_done = True
            subprocess.Popen(args.soft_cmd, shell=True)
        if total >= args.hard:
            print(f"HARD CAP ${args.hard:.0f}: deleting everything", flush=True)
            for _ in range(5):
                if kill(ledger):
                    break
                time.sleep(30)
            (args.state / "budget_killed").write_text(f"{int(time.time())} ${total:.2f}\n")
            print("BUDGET_KILLED", flush=True)
            sys.exit(0)
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
