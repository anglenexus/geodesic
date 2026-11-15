"""Whole-run kill switch on the laptop: API + EC2 compute across a run's EC2 boxes, against one cap.

    python3 scripts/run_guard.py --remotes "aees-l*" --cap 531 [--loop 60]
    python3 scripts/run_guard.py --remotes "aees-l*" --cap 46 --ec2-usd-per-hour 1.6128 --since "2026-11-15 00:30"

API = cost_usd summed over the boxes' mirrored trials (~/code/jobs_remote/<remote>/<job>/*/result.json);
EC2 = each box's compute cost from its last sync (state/remote/<remote>/sync.json), or, with --ec2-usd-per-hour and
--since (local time), boxes x rate x hours since then (until a box's end time in --ended-json, if listed there).
Mirrored trials started before --since are not counted. Each box's own scheduler also has an
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


def totals(cfg: dict, pattern: str, rate: float | None = None, since: dt.datetime | None = None,
           ended: dict[str, str] | None = None) -> tuple[float, float, list[dict]]:
    api = ec2 = 0.0
    boxes = [r for r in cfg["remotes"] if fnmatch.fnmatch(r["name"], pattern)]
    mirror = Path(cfg.get("mirror_root", "~/code/jobs_remote")).expanduser()
    for r in boxes:
        for f in glob.glob(str(mirror / r["name"] / "*" / "*" / "result.json")):
            try:
                res = json.loads(Path(f).read_text())
                if since and (res.get("started_at") or "") < since.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M"):
                    continue
                api += (res.get("agent_result") or {}).get("cost_usd") or 0
            except (OSError, json.JSONDecodeError):
                pass
        if rate is not None and since is not None:
            end = dt.datetime.fromisoformat((ended or {})[r["name"]]) if r["name"] in (ended or {}) else dt.datetime.now()
            ec2 += rate * max(0.0, (end - since).total_seconds()) / 3600
            continue
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
    p.add_argument("--ec2-usd-per-hour", type=float, help="per box; EC2 cost by uptime since --since")
    p.add_argument("--since", type=dt.datetime.fromisoformat, help="run start, local time, e.g. '2026-11-15 00:30'")
    p.add_argument("--ended-json", type=Path, help="JSON with key 'ended_at': {remote: local ISO time it was terminated}")
    a = p.parse_args()
    log = HERE / "state" / "run_guard.log"
    while True:
        cfg = json.loads((HERE / "state" / "remotes.json").read_text())
        ended = (json.loads(a.ended_json.read_text()).get("ended_at") if a.ended_json and a.ended_json.exists() else None)
        api, ec2, boxes = totals(cfg, a.remotes, a.ec2_usd_per_hour, a.since, ended)
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
