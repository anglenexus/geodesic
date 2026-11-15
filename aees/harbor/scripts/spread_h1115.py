"""Spread task attempts over all six boxes (any task on any box), as concurrent `harbor run` jobs per box.

    python3 scripts/spread_h1115.py --tag b1x --attempts '{"crack-7z-hash": 39, "write-compressor": 32}' \\
        [--weights '{"write-compressor": 12, ...}'] [--per-box 40] [--dry-run]

Each task's attempts are split as evenly as possible over the boxes (remainders rotate so totals stay even). On a box,
every task becomes its own job tb21_h1115_i<k>_<tag>_<task> (matches the trackers' tb21_h1115_i* globs), started at
once in tmux session h_<tag>_<task>, with -n set so the box runs at most --per-box trials at a time: each job's share of
the slots follows attempts x weight (expected minutes per trial), so a box's jobs end about together.

Run indices are not assigned on the boxes: the laptop ingests every mirrored trial folder into its ledger
(aees_harbor.ingest), and a folder can be recorded only once, so tasks need not be tied to boxes.
Settings come from the box's state/schedule_h1115_i1.json (agent kwargs, model, harbor build, env file).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor
RUN = "h1115"

REMOTE = r'''
import json, shlex, subprocess, sys
sys.path.insert(0, ".")
from aees_harbor.schedule import harbor_cmd
plan = json.load(open("state/schedule_h1115_i1.json"))
for job, task, k, n in json.loads(sys.argv[1]):
    tf = f"/tmp/tasks_{job}.txt"
    open(tf, "w").write(task + "\n")
    cmd = harbor_cmd(plan, dict(name=job, attempts=k, concurrency=n, tasks_file=tf))
    line = " ".join(shlex.quote(c) for c in cmd) + f" > ~/{job}.harbor.log 2>&1"
    sess = "h_" + job.split("_", 3)[-1]
    subprocess.run(["tmux", "kill-session", "-t", sess], capture_output=True)
    subprocess.run(["tmux", "new", "-d", "-s", sess, "export DOCKER_DEFAULT_PLATFORM=linux/amd64; " + line], check=True)
    print(job, "attempts", k, "-n", n)
'''


def boxes() -> list[dict]:
    cfg = json.loads((HERE / "state" / "remotes.json").read_text())
    return [r for r in cfg["remotes"] if r["name"].startswith("aees-l")]


def split(attempts: dict[str, int], nbox: int) -> list[dict[str, int]]:
    """Per box {task: attempts}: even split, remainders rotated across tasks."""
    out = [dict() for _ in range(nbox)]
    off = 0
    for t, a in sorted(attempts.items(), key=lambda kv: -kv[1]):
        base, rem = divmod(a, nbox)
        for b in range(nbox):
            k = base + (1 if (b - off) % nbox < rem else 0)
            if k:
                out[b][t] = k
        off = (off + rem) % nbox
    return out


def slots(share: dict[str, int], weights: dict[str, float], per_box: int) -> dict[str, int]:
    """-n per job: proportional to attempts x weight, at least 1, at most its attempts, summing to <= per_box."""
    if sum(share.values()) <= per_box:
        return dict(share)  # everything starts at once
    w = {t: k * weights.get(t, 10.0) for t, k in share.items()}
    tot = sum(w.values())
    n = {t: max(1, min(share[t], int(per_box * w[t] / tot))) for t in share}
    while sum(n.values()) > per_box:  # the 1-slot minimum can overshoot: take back from the largest
        t = max((t for t in n if n[t] > 1), key=lambda t: n[t])
        n[t] -= 1
    spare = per_box - sum(n.values())
    for t in sorted(share, key=lambda t: -w[t]):  # hand out rounding leftovers to the heaviest jobs
        if spare <= 0:
            break
        add = min(spare, share[t] - n[t])
        n[t] += add
        spare -= add
    return n


def launch(tag: str, attempts: dict[str, int], weights: dict[str, float], per_box: int, dry: bool) -> list[str]:
    bs = boxes()
    plan = []
    for r, share in zip(bs, split(attempts, len(bs))):
        n = slots(share, weights, per_box)
        jobs = [(f"tb21_{RUN}_i{r['name'][-1]}_{tag}_{t}", t, k, n[t]) for t, k in share.items()]
        plan.append((r, jobs))

    def go(item):
        r, jobs = item
        if dry or not jobs:
            return f"{r['name']}: {[(j[1], j[2], j[3]) for j in jobs]}" + (" [dry-run]" if dry else "")
        p = subprocess.run(["ssh", "-i", str(Path(r["key"]).expanduser()), "-o", "ConnectTimeout=15", f"{r['user']}@{r['host']}",
                            f"cd {r['remote_harbor']} && .venv/bin/python - {json.dumps(json.dumps(jobs))}"],
                           input=REMOTE, capture_output=True, text=True, timeout=120)
        return f"{r['name']}: rc={p.returncode} {p.stdout.strip()} {p.stderr.strip()[-300:] if p.returncode else ''}"

    with ThreadPoolExecutor(len(plan)) as ex:
        return list(ex.map(go, plan))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--attempts", required=True, type=json.loads)
    p.add_argument("--weights", type=json.loads, default={})
    p.add_argument("--per-box", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    for line in launch(a.tag, a.attempts, a.weights, a.per_box, a.dry_run):
        print(line)


if __name__ == "__main__":
    main()
