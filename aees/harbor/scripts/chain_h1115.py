"""Unattended h1115 run on the laptop: batch 1 -> checks -> batch 2 -> per-box finish and teardown.

    python3 scripts/chain_h1115.py [--loop 120] [--dry-run]

Batch 1 (50 attempts x 20 training tasks) runs from each box's plan state/schedule_h1115_i<k>.json. When every box's
batch 1 has ended, this script syncs, evaluates the user's two conditions, and writes state/h1115_b1_report.{json,md}:

  reasoning  encrypted reasoning stored on >= 95% of agent steps, a summary on >= 20% of them, and the next call's
             input carrying the previous step's reasoning (checked by token arithmetic on a sample)
  trajectories  batch 1 finished on every box (not stopped by a kill switch), and trials with no reward (no verifier
             result) <= 3% (the earlier EC2 lesson)

If both hold, batch 2 is appended to each box's plan: for every task, enough attempts for 200 successes in total at
its batch-1 success rate (ceil((200 - s) / p), at most MAX_ATTEMPTS; tasks with no success are left out and listed).
One batch per task, -n 40, on the same box as batch 1 (exactly-once run indices). The estimate (API at each task's
batch-1 cost per trial; EC2 by the longest box) must be <= MAX_B2_USD, else batch 2 is not started. The laptop
run_guard is restarted with cap = spent so far + 140% of the batch-2 estimate; each box's own API backstop is 2x
its estimate.

Every box whose plan has ended (or that the checks stop) is finished one by one: sync, verify the mirror holds every
trial on the box, back up the laptop ledger, test the merge on a copy, merge (box paths rewritten to the mirror),
then terminate the instance. Any failed step leaves the box running and is logged; nothing is terminated unverified.
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
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor
sys.path.insert(0, str(HERE))
from aees_harbor.merge import merge  # noqa: E402
from aees_harbor.remote import load_config, sync_one  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))
import run_guard  # noqa: E402

RUN = "h1115"
TARGET = 200          # successes per task, batches 1 + 2
MAX_ATTEMPTS = 1000   # per task in batch 2
MAX_B2_USD = 400.0    # API + EC2; above this batch 2 is not started (left for the user)
CONC = 40
ENDED = ("done", "failed", "killed", "stopped", "skipped")
STATE = HERE / "state" / f"{RUN}_chain.json"
LOG = HERE / "state" / f"{RUN}_chain.log"
GUARD_SINCE = dt.datetime(2026, 11, 15, 0, 19)
GUARD_CMD = ["caffeinate", "-i", "python3", "scripts/run_guard.py", "--remotes", "aees-l*", "--ec2-usd-per-hour", "1.6128",
             "--since", f"{GUARD_SINCE:%Y-%m-%d %H:%M}", "--ended-json", str(STATE), "--loop", "60"]


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"phase": "b1", "finished_boxes": []}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def boxes(cfg: dict) -> list[dict]:
    return [r for r in cfg["remotes"] if r["name"].startswith("aees-l")]


def ssh(r: dict, cmd: str, timeout: int = 120, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-i", str(Path(r["key"]).expanduser()), "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                           f"{r['user']}@{r['host']}", cmd], capture_output=True, text=True, timeout=timeout, input=input)


def box_plan(r: dict) -> dict | None:
    p = ssh(r, f"cat {r['remote_harbor']}/{r['plan']}")
    return json.loads(p.stdout) if p.returncode == 0 else None


def mirror(r: dict) -> Path:
    return Path("~/code/jobs_remote").expanduser() / r["name"]


def guard_fired() -> bool:
    log_path = HERE / "state" / "run_guard.log"
    return log_path.exists() and any("CAP REACHED" in l and l >= "2026-11-15 00:19" for l in log_path.read_text().splitlines())


# ---------- batch-1 evaluation ----------

def trials(job_glob: str) -> list[dict]:
    out = []
    for res in glob.glob(job_glob + "/*__*/result.json"):
        d = Path(res).parent
        try:
            r = json.loads(Path(res).read_text())
        except json.JSONDecodeError:
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
    """Steps whose input grew by the previous step's reasoning (beyond reply + terminal output), via o200k tokens.
    Run on a box (tiktoken is in the harbor venv there); returns (consistent, checked)."""
    ok = n = 0
    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 - no tokenizer on the laptop: skip, reported as not checked
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


def evaluate(cfg: dict, plans: dict) -> dict:
    ts = trials(str(Path("~/code/jobs_remote").expanduser() / "aees-l*" / f"tb21_{RUN}_i*_b1"))
    steps = [s for t in ts for s in t["steps"]]
    enc = sum(1 for s in steps if ((s.get("metrics") or {}).get("extra") or {}).get("reasoning_encrypted"))
    summ = sum(1 for s in steps if s.get("reasoning_content"))
    ok, n = carry_check(sorted(ts, key=lambda t: -len(t["steps"]))[:40])
    no_reward = sum(1 for t in ts if t["reward"] is None)
    per = {}
    for t in ts:
        d = per.setdefault(t["task"], dict(n=0, s=0, cost=0.0, ctx_stop=0, timeouts=0, no_reward=0, secs=[]))
        d["n"] += 1; d["s"] += t["reward"] == 1; d["cost"] += t["cost"]; d["ctx_stop"] += t["ctx_stop"]
        d["timeouts"] += t["exc"] == "AgentTimeoutError"; d["no_reward"] += t["reward"] is None
        if t["started"] and t["finished"]:
            d["secs"].append((dt.datetime.fromisoformat(t["finished"].replace("Z", "+00:00"))
                              - dt.datetime.fromisoformat(t["started"].replace("Z", "+00:00"))).total_seconds())
    statuses = {name: [b["status"] for b in p["batches"]] for name, p in plans.items()}
    checks = {
        "encrypted_reasoning_on_>=95%_of_steps": bool(steps) and enc / len(steps) >= 0.95,
        "summary_on_>=20%_of_steps": bool(steps) and summ / len(steps) >= 0.20,
        "reasoning_carried_on_>=90%_of_>=20_checked_steps": n >= 20 and ok / n >= 0.90,
        "batch1_done_on_every_box": all(s and s[0] == "done" for s in statuses.values()),
        "kill_switch_not_fired": not guard_fired(),
        "no_reward_<=3%": bool(ts) and no_reward / len(ts) <= 0.03,
        "every_task_has_trials": len(per) == 20,
    }
    return dict(trials=len(ts), steps=len(steps), encrypted_steps=enc, summary_steps=summ, carry_ok=ok, carry_checked=n,
                no_reward=no_reward, per_task={k: {**v, "secs": None, "median_min": round(sorted(v["secs"])[len(v["secs"]) // 2] / 60, 1)
                                                if v["secs"] else None} for k, v in per.items()},
                box_status=statuses, checks=checks, passed=all(checks.values()))


def plan_b2(ev: dict, plans: dict, cfg: dict) -> dict:
    task_box = {t: r["name"] for r in boxes(cfg) for t in (HERE / "state" / f"tasks_{RUN}_i{r['name'][-1]}.txt").read_text().split()}
    attempts, left_out, api, box_hours = {}, [], 0.0, {}
    for t, d in ev["per_task"].items():
        if d["s"] >= TARGET:
            continue
        if d["s"] == 0:
            left_out.append(t); continue
        a = min(MAX_ATTEMPTS, math.ceil((TARGET - d["s"]) / (d["s"] / d["n"])))
        attempts[t] = a
        api += a * d["cost"] / d["n"]
        box_hours[task_box[t]] = box_hours.get(task_box[t], 0.0) + a * (d["median_min"] or 15) / 60 / CONC
    ec2 = 1.6128 * sum(box_hours.values()) + 1.6128 * 0.5 * len(box_hours)  # per box until its own teardown, + 30 min
    return dict(attempts=attempts, left_out=left_out, task_box=task_box, api_est=round(api, 2), ec2_est=round(ec2, 2),
                total_est=round(api + ec2, 2), box_hours={k: round(v, 2) for k, v in box_hours.items()},
                trials=sum(attempts.values()))


def write_report(ev: dict, b2: dict | None, decision: str) -> None:
    (HERE / "state" / f"{RUN}_b1_report.json").write_text(json.dumps(dict(evaluation=ev, batch2=b2, decision=decision), indent=1))
    lines = [f"# {RUN} batch 1 report ({dt.datetime.now():%Y-%m-%d %H:%M} PT)", "", f"**Decision:** {decision}", "",
             "## Checks", *[f"- {'PASS' if v else 'FAIL'} {k}" for k, v in ev["checks"].items()], "",
             f"Trials {ev['trials']}, agent steps {ev['steps']}: encrypted reasoning on {ev['encrypted_steps']}, summary on "
             f"{ev['summary_steps']}; reasoning carried on {ev['carry_ok']}/{ev['carry_checked']} checked steps; no reward {ev['no_reward']}.",
             "", "## Per task (batch 1)", "", "| Task | Trials | Successes | Rate | Context stops | Timeouts | $/trial | Batch-2 attempts |",
             "|---|---|---|---|---|---|---|---|"]
    for t, d in sorted(ev["per_task"].items(), key=lambda kv: kv[1]["s"] / max(kv[1]["n"], 1)):
        lines.append(f"| {t} | {d['n']} | {d['s']} | {100 * d['s'] / max(d['n'], 1):.0f}% | {d['ctx_stop']} | {d['timeouts']} | "
                     f"{d['cost'] / max(d['n'], 1):.4f} | {(b2 or {}).get('attempts', {}).get(t, '-')} |")
    if b2:
        lines += ["", f"Batch 2: {b2['trials']} trials, estimate API ${b2['api_est']} + EC2 ${b2['ec2_est']} = ${b2['total_est']}; "
                  f"left out (no success in batch 1): {', '.join(b2['left_out']) or 'none'}"]
    (HERE / "state" / f"{RUN}_b1_report.md").write_text("\n".join(lines) + "\n")


def launch_b2(cfg: dict, plans: dict, b2: dict, dry: bool) -> None:
    for r in boxes(cfg):
        i = r["name"][-1]
        plan = plans[r["name"]]
        mine = {t: a for t, a in b2["attempts"].items() if b2["task_box"][t] == r["name"]}
        if not mine:
            continue
        files = {}
        for t, a in sorted(mine.items(), key=lambda kv: -kv[1]):
            tf = f"state/tasks_{RUN}_i{i}_b2_{t}.txt"
            files[tf] = t + "\n"
            plan["batches"].append(dict(name=f"tb21_{RUN}_i{i}_b2_{t}", attempts=a, concurrency=CONC, status="planned",
                                        est_trials=a, tasks_file=tf, note=f"batch 2: {t}"))
        box_api = sum(a * b2_cost_per_trial(t) for t, a in mine.items())
        plan["budget_usd"] = round(plan["budget_usd"] + 2 * box_api, 2)
        plan["name"] = plan["name"].replace("(batch 1)", "(batches 1 + 2)")
        if dry:
            log(f"[dry-run] {r['name']}: would add {mine}, backstop ${plan['budget_usd']}")
            continue
        for tf, body in files.items():
            ssh(r, f"cat > {r['remote_harbor']}/{tf}", input=body)
        ssh(r, f"cat > {r['remote_harbor']}/{r['plan']}", input=json.dumps(plan, indent=2) + "\n")
        p = ssh(r, f"cd {r['remote_harbor']} && rm -f state/STOP && tmux kill-session -t sched 2>/dev/null; "
                   f"tmux new -d -s sched \"export PATH=\\$HOME/.local/bin:\\$PATH; .venv/bin/python -m aees_harbor.schedule "
                   f"--plan {r['plan']} 2>&1 | tee -a ~/sched.out\"; sleep 15; tail -1 ~/sched.out")
        log(f"{r['name']}: batch 2 started ({mine}); backstop ${plan['budget_usd']}: {p.stdout.strip()[-150:]}")
    api, ec2, _ = run_guard.totals(cfg, "aees-l*", 1.6128, GUARD_SINCE, load_state().get("ended_at"))
    spent = api + ec2
    cap = round(spent + 1.4 * b2["total_est"], 2)
    if not dry:
        subprocess.run(["pkill", "-f", "scripts/run_guard.py"])
        subprocess.Popen(GUARD_CMD[:4] + ["--cap", str(cap)] + GUARD_CMD[4:], cwd=HERE,
                         stdout=open(HERE / "state" / "run_guard.out", "a"), stderr=subprocess.STDOUT, start_new_session=True)
        planned = 1000 + b2["trials"]
        subprocess.run(["pkill", "-f", f"tmax_view.py --run {RUN}"])
        subprocess.Popen(["python3", "scripts/tmax_view.py", "--run", RUN, "--jobs", f"~/code/jobs_remote/aees-l*/tb21_{RUN}_i*",
                          "--data", str(Path("~/code/tmax_view").expanduser()), "--planned", str(planned), "--reasoning-effort", "high",
                          "--loop", "60"], cwd=HERE, stdout=open(HERE / "state" / f"tmax_view_{RUN}.out", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    log(f"run_guard cap -> ${cap} (spent ${spent:.2f} + 140% of ${b2['total_est']})")


_COST: dict[str, float] = {}


def b2_cost_per_trial(task: str) -> float:
    return _COST.get(task, 0.03)


# ---------- per-box finish ----------

def finish_box(cfg: dict, r: dict, dry: bool) -> bool:
    name, i = r["name"], r["name"][-1]
    if ssh(r, "pgrep -f '[h]arbor run --env' >/dev/null && echo busy || echo idle").stdout.strip() != "idle":
        return False
    res = sync_one(cfg, r)
    if not res["ok"]:
        log(f"{name}: sync errors {res['errors']}; not finishing"); return False
    count = ssh(r, f"ls -d {r['remote_jobs']}/tb21_{RUN}_i{i}_*/*__*/result.json 2>/dev/null | wc -l").stdout.strip()
    local = len(glob.glob(str(mirror(r) / f"tb21_{RUN}_i{i}_*" / "*__*" / "result.json")))
    if not count.isdigit() or int(count) != local:
        log(f"{name}: mirror has {local} trials, box has {count}; not finishing"); return False
    src = HERE / "state" / "remote" / name / "ledger.sqlite"
    if not src.exists():
        log(f"{name}: no ledger snapshot; not finishing"); return False
    stamp = dt.datetime.now().strftime("%m%d_%H%M")
    led = HERE / "state" / "ledger.sqlite"
    backup = HERE / "state" / f"ledger_backup_before_{RUN}_{name}_{stamp}.sqlite"
    rules = [(r["remote_jobs"], str(mirror(r)))]
    try:
        shutil.copy2(led, backup)
        test = HERE / "state" / f"ledger_mergetest_{RUN}_{name}.sqlite"
        shutil.copy2(led, test)
        merge(test, [src], rules)
        test.unlink()
        if dry:
            log(f"[dry-run] {name}: merge test ok; would merge and terminate {r['instance_id']}"); return True
        merge(led, [src], rules)
    except Exception as e:  # noqa: BLE001 - any failure: keep the box, log
        log(f"{name}: merge failed ({type(e).__name__}: {e}); ledger backup {backup.name}; not terminating"); return False
    (HERE / "state" / "remote" / name / f"ledger_{RUN}_final.sqlite").write_bytes(src.read_bytes())
    t = subprocess.run(["aws", "ec2", "terminate-instances", "--region", r.get("region", "us-east-2"), "--instance-ids",
                        r["instance_id"], "--query", "TerminatingInstances[0].CurrentState.Name", "--output", "text"],
                       capture_output=True, text=True, timeout=120)
    log(f"{name}: {local} trials verified and merged (backup {backup.name}); terminate -> {t.stdout.strip() or t.stderr.strip()[:200]}")
    return t.returncode == 0


# ---------- main loop ----------

def step(dry: bool) -> bool:
    cfg = load_config()
    s = load_state()
    plans = {r["name"]: box_plan(r) for r in boxes(cfg) if r["name"] not in s["finished_boxes"]}
    if any(p is None for p in plans.values()):
        log(f"cannot read plans from {[k for k, v in plans.items() if v is None]}; retrying"); return False
    if s["phase"] == "b1":
        if not all(p["batches"][0]["status"] in ENDED for p in plans.values()):
            return False
        log("batch 1 ended on every box; syncing and evaluating")
        for r in boxes(cfg):
            sync_one(cfg, r)
        ev = evaluate(cfg, plans)
        for t, d in ev["per_task"].items():
            _COST[t] = d["cost"] / max(d["n"], 1)
        b2 = plan_b2(ev, plans, cfg) if ev["passed"] else None
        if not ev["passed"]:
            decision = "batch 2 NOT started: checks failed; boxes are finished and terminated"
        elif b2["total_est"] > MAX_B2_USD:
            decision = f"batch 2 NOT started: estimate ${b2['total_est']} > ${MAX_B2_USD}; boxes are finished and terminated"
        elif not b2["attempts"]:
            decision = "batch 2 not needed (every task at target or left out)"
        else:
            decision = f"batch 2 started: {b2['trials']} trials, estimate ${b2['total_est']}"
        write_report(ev, b2, decision)
        log(decision)
        if decision.startswith("batch 2 started"):
            launch_b2(cfg, plans, b2, dry)
        s["phase"] = "b2" if decision.startswith("batch 2 started") else "finish"
        save_state(s)
        return False
    # b2 / finish: finish each box whose plan has ended
    for r in boxes(cfg):
        if r["name"] in s["finished_boxes"]:
            continue
        p = plans[r["name"]]
        if s["phase"] == "b2" and not all(b["status"] in ENDED for b in p["batches"]):
            continue
        if finish_box(cfg, r, dry) and not dry:
            s["finished_boxes"].append(r["name"])
            s.setdefault("ended_at", {})[r["name"]] = dt.datetime.now().isoformat(timespec="seconds")
            save_state(s)
    if len(s["finished_boxes"]) == len(boxes(cfg)):
        log("all boxes finished and terminated"); s["phase"] = "done"; save_state(s); return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--loop", type=int, default=120)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    log(f"chain started (dry_run={a.dry_run})")
    while True:
        try:
            if step(a.dry_run) or a.dry_run:
                break
        except Exception as e:  # noqa: BLE001 - keep going; logged
            log(f"error: {type(e).__name__}: {e}")
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
