"""Whole-run kill switch on the laptop: API + EC2 compute across a run's EC2 boxes, against one cap.

    python3 scripts/run_guard.py --remotes "aees-l*" --cap 531 [--loop 60]

API = cost_usd summed over the boxes' mirrored trials (~/code/jobs_remote/<remote>/<job>/*/result.json);
EC2 = each box's compute cost from its last sync (state/remote/<remote>/sync.json). Each box's own scheduler also has an
API-only kill switch. At the cap this stops every running box's run: it creates the scheduler's STOP file (no new
batch) and sends SIGINT to `harbor run` (in-flight trials end; harbor tears down its containers). It never terminates
instances: their trials must be copied back and merged first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import glob
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor


def totals(cfg: dict, pattern: str) -> tuple[float, float, list[dict]]:
    api = ec2 = 0.0
    boxes = [r for r in cfg["remotes"] if fnmatch.fnmatch(r["name"], pattern)]
    mirror = Path(cfg.get("mirror_root", "~/code/jobs_remote")).expanduser()
    for r in boxes:
        for f in glob.glob(str(mirror / r["name"] / "*" / "*" / "result.json")):
            try:
                api += (json.loads(Path(f).read_text()).get("agent_result") or {}).get("cost_usd") or 0
            except (OSError, json.JSONDecodeError):
                pass
        try:
            ec2 += ((json.loads((HERE / "state" / "remote" / r["name"] / "sync.json").read_text()).get("ec2") or {})
                    .get("cost_usd") or 0)
        except (OSError, json.JSONDecodeError):
            pass
    return api, ec2, boxes


def stop_all(boxes: list[dict]) -> None:
    for r in boxes:
        cmd = ("touch ~/geodesic/aees/harbor/state/STOP; pkill -INT -f 'harbor run' || true")
        subprocess.run(["ssh", "-n", "-i", str(Path(r["key"]).expanduser()), "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                        f"{r['user']}@{r['host']}", cmd], capture_output=True, timeout=60)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--remotes", required=True, help="glob of remote names, e.g. 'aees-l*'")
    p.add_argument("--cap", type=float, required=True, help="USD, API + EC2 compute")
    p.add_argument("--loop", type=int, default=60)
    a = p.parse_args()
    log = HERE / "state" / "run_guard.log"
    while True:
        cfg = json.loads((HERE / "state" / "remotes.json").read_text())
        api, ec2, boxes = totals(cfg, a.remotes)
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} api ${api:.2f} + ec2 ${ec2:.2f} = ${api + ec2:.2f} / cap ${a.cap:.0f}"
        if api + ec2 >= a.cap:
            stop_all(boxes)
            line += " -> CAP REACHED: STOP file + SIGINT sent to every box"
        with open(log, "a") as fh:
            fh.write(line + "\n")
        if api + ec2 >= a.cap or not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
