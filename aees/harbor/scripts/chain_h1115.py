"""Unattended h1115 run on the laptop: batch 1 -> checks -> batch 2 on all boxes -> verified ledger ingest -> teardown.

    python3 scripts/chain_h1115.py [--loop 120] [--dry-run]

Any task runs on any box: run indices are assigned here, on the laptop, by ingesting every mirrored trial folder
(aees_harbor.ingest; a folder is recorded once), so nothing ties tasks to boxes.

1. Wait until no `harbor run` is left on any box (batch 1: jobs tb21_h1115_i<k>_b1 and _b1x_<task>), then sync and
   evaluate the user's two conditions, written to state/h1115_b1_report.{json,md}:
     reasoning     encrypted reasoning on >= 95% of agent steps, a summary on >= 20%, and the previous step's
                   reasoning re-sent on >= 90% of >= 20 checked steps (token arithmetic, o200k tokenizer)
     trajectories  >= 95% of each task's 50 attempts finished, no reward <= 3% (earlier EC2 lesson), kill switch quiet
2. If both hold: batch 2 = for each task ceil((300 - s) / p) attempts at its batch-1 rate p (at most MAX_ATTEMPTS;
   tasks with no success are left out and listed), spread over all six boxes by scripts/spread_h1115.py at 40 per box
   (240 at once). Not started if the estimate (API at batch-1 $/trial per task; EC2 = 6 boxes x the expected hours)
   is above MAX_B2_USD. The laptop run_guard is restarted with cap = counted so far + 140% of the batch-2 estimate, and
   each box gets a backstop (tmux h1115_backstop) that interrupts its harbor jobs at 2x its share of the API estimate.
3. When no harbor job is left on any box (batch 2 done, or the checks failed, or a kill switch fired): sync, verify the
   mirror holds every trial folder each box has, back up the laptop ledger, ingest every mirrored h1115 job into a
   copy, then into the ledger, then terminate all six instances. Any failed step terminates nothing and is logged.
Log: state/h1115_chain.log; state: state/h1115_chain.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from aees_harbor.ingest import ingest_job  # noqa: E402
from aees_harbor.ledger import Ledger  # noqa: E402
from aees_harbor.remote import load_config, sync_one  # noqa: E402
import run_guard  # noqa: E402
import spread_h1115  # noqa: E402

RUN = "h1115"
TARGET = 300          # successes per task, batches 1 + 2 (user, 2026-11-15 00:55)
B1_ATTEMPTS = 50
MAX_ATTEMPTS = 2000   # per task in batch 2 (count-dataset-tokens needs ~1,300 at 22%)
MAX_B2_USD = 400.0    # API + EC2; above this batch 2 is not started (left for the user)
PER_BOX = 40
RATE = 1.6128         # m7i.8xlarge $/h
MIRROR = Path("~/code/jobs_remote").expanduser()
STATE = HERE / "state" / f"{RUN}_chain.json"
LOG = HERE / "state" / f"{RUN}_chain.log"
GUARD_SINCE = dt.datetime(2026, 11, 15, 0, 19)
GUARD_CMD = ["caffeinate", "-i", "python3", "scripts/run_guard.py", "--remotes", "aees-l*", "--ec2-usd-per-hour", str(RATE),
             "--since", f"{GUARD_SINCE:%Y-%m-%d %H:%M}", "--ended-json", str(STATE), "--loop", "60"]
BACKSTOP = r'''
import glob, json, subprocess, sys, time
cap = float(sys.argv[1])
while True:
    c = 0.0
    for f in glob.glob("/home/ubuntu/jobs/tb21_h1115_i*_b2_*/*__*/result.json"):
        try:
            c += (json.load(open(f)).get("agent_result") or {}).get("cost_usd") or 0
        except Exception:
            pass
    if c >= cap:
        subprocess.run(["pkill", "-INT", "-f", "[h]arbor run --env"])
        open("/home/ubuntu/h1115_backstop.log", "a").write(f"{time.ctime()} backstop: ${c:.2f} >= ${cap}\n")
        break
    if subprocess.run(["pgrep", "-f", "[h]arbor run --env"], capture_output=True).returncode != 0:
        break
    time.sleep(60)
'''


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"phase": "b1"}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def boxes(cfg: dict) -> list[dict]:
    return [r for r in cfg["remotes"] if r["name"].startswith("aees-l")]


def ssh(r: dict, cmd: str, timeout: int = 120, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-i", str(Path(r["key"]).expanduser()), "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                           f"{r['user']}@{r['host']}", cmd], capture_output=True, text=True, timeout=timeout, input=input)


def all_idle(cfg: dict) -> bool | None:
    """True if no box runs `harbor run`; None if a box can't be reached."""
    def q(r):
        p = ssh(r, "pgrep -f '[h]arbor run --env' >/dev/null && echo busy || echo idle")
        return p.stdout.strip() if p.returncode == 0 else None
    with ThreadPoolExecutor(6) as ex:
        states = list(ex.map(q, boxes(cfg)))
    return None if None in states else all(s == "idle" for s in states)


def sync_all(cfg: dict) -> list[str]:
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda r: sync_one(cfg, r), boxes(cfg)))
    return [f"{x['name']}: {x['errors']}" for x in res if not x["ok"]]


def guard_fired() -> bool:
    p = HERE / "state" / "run_guard.log"
    return p.exists() and any("CAP REACHED" in l and l >= f"{GUARD_SINCE:%Y-%m-%d %H:%M}" for l in p.read_text().splitlines())


# ---------- batch-1 evaluation ----------

def trials(job_glob: str) -> list[dict]:
    out = []
    for res in glob.glob(job_glob + "/*__*/result.json"):
        d = Path(res).parent
        try:
            r = json.loads(Path(res).read_text())
        except json.JSONDecodeError:
            continue
        if not r.get("finished_at"):
            continue
        steps = []
        try:
            steps = [s for s in json.loads((d / "agent" / "trajectory.json").read_text())["steps"] if s.get("source") == "agent"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        meta = (r.get("agent_result") or {}).get("metadata") or {}
        out.append(dict(task=(r.get("task_name") or "").split("/")[-1], dir=str(d),
                        reward=((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                        exc=(r.get("exception_info") or {}).get("exception_type"),
                        cost=(r.get("agent_result") or {}).get("cost_usd") or 0.0, steps=steps,
                        ctx_stop=bool(meta.get("stopped_at_context_limit")),
                        started=r.get("started_at"), finished=r.get("finished_at")))
    return out


def carry_check(sample: list[dict]) -> tuple[int, int]:
    """Steps whose input grew by the previous step's reasoning beyond reply + terminal output: (consistent, checked)."""
    ok = n = 0
    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 - no tokenizer: nothing checked, and the check fails
        return 0, 0
    for t in sample:
        st = t["steps"]
        for a, b in zip(st, st[1:]):
            ma, mb = a.get("metrics") or {}, b.get("metrics") or {}
            r = ((ma.get("extra") or {}).get("reasoning_tokens")) or 0
            if r < 50 or not ma.get("completion_tokens") or not mb.get("prompt_tokens"):
                continue
            obs = "\n".join(x.get("content") or "" for x in (a.get("observation") or {}).get("results", []))
            resid = mb["prompt_tokens"] - ma["prompt_tokens"] - (ma["completion_tokens"] - r) - len(enc.encode(obs))
            n += 1
            ok += abs(resid - r) < abs(resid)
    return ok, n


def b1_glob() -> str:
    return str(MIRROR / "aees-l*" / f"tb21_{RUN}_i*_b1*")


def evaluate(train: list[str]) -> dict:
    ts = trials(b1_glob())
    steps = [s for t in ts for s in t["steps"]]
    enc = sum(1 for s in steps if ((s.get("metrics") or {}).get("extra") or {}).get("reasoning_encrypted"))
    summ = sum(1 for s in steps if s.get("reasoning_content"))
    ok, n = carry_check(sorted(ts, key=lambda t: -len(t["steps"]))[:40])
    no_reward = sum(1 for t in ts if t["reward"] is None)
    per = {t: dict(n=0, s=0, cost=0.0, ctx_stop=0, timeouts=0, no_reward=0, secs=[]) for t in train}
    for t in ts:
        if t["task"] not in per:
            continue
        d = per[t["task"]]
        d["n"] += 1; d["s"] += t["reward"] == 1; d["cost"] += t["cost"]; d["ctx_stop"] += t["ctx_stop"]
        d["timeouts"] += t["exc"] == "AgentTimeoutError"; d["no_reward"] += t["reward"] is None
        if t["started"] and t["finished"]:
            d["secs"].append((dt.datetime.fromisoformat(t["finished"].replace("Z", "+00:00"))
                              - dt.datetime.fromisoformat(t["started"].replace("Z", "+00:00"))).total_seconds())
    for d in per.values():
        d["median_min"] = round(sorted(d["secs"])[len(d["secs"]) // 2] / 60, 1) if d["secs"] else None
        d.pop("secs")
    checks = {
        "encrypted_reasoning_on_>=95%_of_steps": bool(steps) and enc / len(steps) >= 0.95,
        "summary_on_>=20%_of_steps": bool(steps) and summ / len(steps) >= 0.20,
        "reasoning_resent_on_>=90%_of_>=20_checked_steps": n >= 20 and ok / n >= 0.90,
        "every_task_>=95%_of_50_finished": all(d["n"] >= 0.95 * B1_ATTEMPTS for d in per.values()),
        "no_reward_<=3%": bool(ts) and no_reward / len(ts) <= 0.03,
        "kill_switch_not_fired": not guard_fired(),
    }
    return dict(trials=len(ts), steps=len(steps), encrypted_steps=enc, summary_steps=summ, carry_ok=ok, carry_checked=n,
                no_reward=no_reward, per_task=per, checks=checks, passed=all(checks.values()))


def plan_b2(ev: dict, nbox: int = 6) -> dict:
    attempts, left_out, api, minutes = {}, [], 0.0, 0.0
    for t, d in ev["per_task"].items():
        if d["s"] >= TARGET or d["n"] == 0:
            continue
        if d["s"] == 0:
            left_out.append(t)
            continue
        a = min(MAX_ATTEMPTS, math.ceil((TARGET - d["s"]) / (d["s"] / d["n"])))
        attempts[t] = a
        api += a * d["cost"] / d["n"]
        minutes += a * (d["median_min"] or 15)
    hours = minutes / 60 / (nbox * PER_BOX) + 0.5  # all slots busy, + 30 min for the tail and teardown
    ec2 = nbox * RATE * hours
    return dict(attempts=attempts, left_out=left_out, api_est=round(api, 2), ec2_est=round(ec2, 2),
                total_est=round(api + ec2, 2), hours_est=round(hours, 2), trials=sum(attempts.values()),
                weights={t: ev["per_task"][t]["median_min"] or 15 for t in attempts},
                cost_per_trial={t: ev["per_task"][t]["cost"] / max(ev["per_task"][t]["n"], 1) for t in attempts})


def write_report(ev: dict, b2: dict | None, decision: str) -> None:
    (HERE / "state" / f"{RUN}_b1_report.json").write_text(json.dumps(dict(evaluation=ev, batch2=b2, decision=decision), indent=1))
    lines = [f"# {RUN} batch 1 report ({dt.datetime.now():%Y-%m-%d %H:%M} PT)", "", f"**Decision:** {decision}", "",
             "## Checks", *[f"- {'PASS' if v else 'FAIL'} {k}" for k, v in ev["checks"].items()], "",
             f"Trials {ev['trials']}, agent steps {ev['steps']}: encrypted reasoning on {ev['encrypted_steps']}, summary on "
             f"{ev['summary_steps']}; reasoning re-sent on {ev['carry_ok']}/{ev['carry_checked']} checked steps; "
             f"no reward {ev['no_reward']}.", "", "## Per task (batch 1)", "",
             "| Task | Trials | Successes | Rate | Context stops | Timeouts | Median min | $/trial | Batch-2 attempts |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t, d in sorted(ev["per_task"].items(), key=lambda kv: kv[1]["s"] / max(kv[1]["n"], 1)):
        lines.append(f"| {t} | {d['n']} | {d['s']} | {100 * d['s'] / max(d['n'], 1):.0f}% | {d['ctx_stop']} | {d['timeouts']} | "
                     f"{d['median_min']} | {d['cost'] / max(d['n'], 1):.4f} | {(b2 or {}).get('attempts', {}).get(t, '-')} |")
    if b2:
        lines += ["", f"Batch 2: {b2['trials']} trials on 6 boxes x {PER_BOX}; estimate API ${b2['api_est']} + EC2 ${b2['ec2_est']} "
                  f"(~{b2['hours_est']} h) = ${b2['total_est']}; left out (no success in batch 1): {', '.join(b2['left_out']) or 'none'}"]
    (HERE / "state" / f"{RUN}_b1_report.md").write_text("\n".join(lines) + "\n")


def launch_b2(cfg: dict, b2: dict, dry: bool) -> None:
    for line in spread_h1115.launch("b2", b2["attempts"], b2["weights"], PER_BOX, dry):
        log(f"batch 2 launch: {line}")
    box_cap = round(2 * b2["api_est"] / len(boxes(cfg)), 2)
    api, ec2, _ = run_guard.totals(cfg, "aees-l*", RATE, GUARD_SINCE, load_state().get("ended_at"))
    cap = round(api + ec2 + 1.4 * b2["total_est"], 2)
    if dry:
        log(f"[dry-run] would start box backstops at ${box_cap} and set run_guard cap ${cap}")
        return
    for r in boxes(cfg):
        ssh(r, "cat > /tmp/h1115_backstop.py", input=BACKSTOP)
        ssh(r, f"tmux kill-session -t h1115_backstop 2>/dev/null; sleep 20; tmux new -d -s h1115_backstop 'python3 /tmp/h1115_backstop.py {box_cap}'")
    subprocess.run(["pkill", "-f", "scripts/run_guard.py"])
    subprocess.Popen(GUARD_CMD[:4] + ["--cap", str(cap)] + GUARD_CMD[4:], cwd=HERE, stdout=open(HERE / "state" / "run_guard.out", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    subprocess.run(["pkill", "-f", f"tmax_view.py --run {RUN}"])
    subprocess.Popen(["python3", "scripts/tmax_view.py", "--run", RUN, "--jobs", str(MIRROR / "aees-l*" / f"tb21_{RUN}_i*"),
                      "--data", str(Path("~/code/tmax_view").expanduser()), "--planned", str(1000 + b2["trials"]),
                      "--reasoning-effort", "high", "--loop", "60"], cwd=HERE,
                     stdout=open(HERE / "state" / f"tmax_view_{RUN}.out", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    log(f"box backstops at ${box_cap} each; run_guard cap -> ${cap} (counted ${api + ec2:.2f} + 140% of ${b2['total_est']})")


# ---------- finish: verify, ingest, terminate ----------

def finish(cfg: dict, dry: bool) -> bool:
    errs = sync_all(cfg)
    if errs:
        log(f"sync errors {errs}; retrying later"); return False
    for r in boxes(cfg):
        p = ssh(r, f"cd {r['remote_jobs']} && ls -d tb21_{RUN}_*/*__* 2>/dev/null")
        on_box = set(p.stdout.split())
        local = {str(Path(d).relative_to(MIRROR / r["name"])) for d in glob.glob(str(MIRROR / r["name"] / f"tb21_{RUN}_*" / "*__*"))}
        if p.returncode != 0 or not on_box <= local:
            log(f"{r['name']}: mirror misses {len(on_box - local)} of {len(on_box)} trial folders; not finishing"); return False
    jobs = sorted(Path(j) for j in glob.glob(str(MIRROR / "aees-l*" / f"tb21_{RUN}_i*")))
    led = HERE / "state" / "ledger.sqlite"
    stamp = dt.datetime.now().strftime("%m%d_%H%M")
    backup = HERE / "state" / f"ledger_backup_before_{RUN}_{stamp}.sqlite"
    test = HERE / "state" / f"ledger_ingesttest_{RUN}.sqlite"
    try:
        shutil.copy2(led, backup)
        shutil.copy2(led, test)
        tl = Ledger(test)
        tcounts = [ingest_job(tl, j) for j in jobs]
        tl.close()
        test.unlink()
        if dry:
            log(f"[dry-run] ingest test ok on {len(jobs)} jobs: {sum(c['ingested'] for c in tcounts)} trials; would ingest and terminate")
            return True
        rl = Ledger(led)
        counts = [ingest_job(rl, j) for j in jobs]
        rl.close()
    except Exception as e:  # noqa: BLE001 - keep the boxes
        log(f"ingest failed ({type(e).__name__}: {e}); ledger backup {backup.name}; not terminating"); return False
    n = sum(c["ingested"] for c in counts)
    log(f"ingested {n} trials from {len(jobs)} jobs into the ledger (backup {backup.name}; skipped "
        f"{sum(c['skipped_target_full_or_frozen'] for c in counts)}, already there {sum(c['already_in_ledger'] for c in counts)})")
    s = load_state()
    for r in boxes(cfg):
        t = subprocess.run(["aws", "ec2", "terminate-instances", "--region", r.get("region", "us-east-2"), "--instance-ids",
                            r["instance_id"], "--query", "TerminatingInstances[0].CurrentState.Name", "--output", "text"],
                           capture_output=True, text=True, timeout=120)
        s.setdefault("ended_at", {})[r["name"]] = dt.datetime.now().isoformat(timespec="seconds")
        log(f"{r['name']}: terminate -> {t.stdout.strip() or t.stderr.strip()[:200]}")
    save_state(s)
    return True


# ---------- main loop ----------

def step(dry: bool) -> bool:
    cfg = load_config()
    s = load_state()
    idle = all_idle(cfg)
    if idle is None:
        log("a box is unreachable; retrying"); return False
    if not idle:
        return False
    if s["phase"] == "b1":
        log("no harbor job left on any box: batch 1 ended; syncing and evaluating")
        errs = sync_all(cfg)
        if errs:
            log(f"sync errors {errs}; retrying later"); return False
        train = (HERE / "state" / "tb21_split1114_train.txt").read_text().split()
        ev = evaluate(train)
        b2 = plan_b2(ev, len(boxes(cfg))) if ev["passed"] else None
        if not ev["passed"]:
            decision = "batch 2 NOT started: checks failed; boxes will be finished and terminated"
        elif b2["total_est"] > MAX_B2_USD:
            decision = f"batch 2 NOT started: estimate ${b2['total_est']} > ${MAX_B2_USD}; boxes will be finished and terminated"
        elif not b2["attempts"]:
            decision = "batch 2 not needed (every task at target or left out)"
        else:
            decision = f"batch 2 started: {b2['trials']} trials on 6 boxes x {PER_BOX}, estimate ${b2['total_est']} (~{b2['hours_est']} h)"
        write_report(ev, b2, decision)
        log(decision)
        if decision.startswith("batch 2 started"):
            launch_b2(cfg, b2, dry)
            s["phase"] = "b2"
        else:
            s["phase"] = "finish"
        if not dry:
            save_state(s)
        return dry
    # b2 ended (no harbor job left) or finish: verify, ingest, terminate
    if finish(cfg, dry):
        if not dry:
            s["phase"] = "done"; save_state(s)
        log("done")
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loop", type=int, default=120)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    log(f"chain started (dry_run={a.dry_run})")
    while True:
        try:
            if step(a.dry_run):
                break
        except Exception as e:  # noqa: BLE001 - keep going; logged
            log(f"error: {type(e).__name__}: {e}")
        if a.dry_run:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
