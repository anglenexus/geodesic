"""Per-task statistics for Terminal-Bench from the public TB 2.0 leaderboard trials."""
import collections, glob, json, os, statistics as st, sys, tomllib, filecmp, csv

LB = sys.argv[1]           # .../tb2lb/submissions/terminal-bench/2.0
TB20, TB21 = sys.argv[2], sys.argv[3]   # task-definition dirs for 2.0 and 2.1
OUT = sys.argv[4]

ALIASES = {"install-windows-3-11": "install-windows-3.11"}  # two spellings of one task

# --- 1. load every trial-level record: (submission, job, task, reward or None) ---
rows, skipped_21 = [], 0
for p in glob.glob(f"{LB}/*/*/*/result.json"):
    sub, job = p[len(LB) + 1:].split("/")[:2]
    if "2.1" in job:            # a few submissions ran TB 2.1 tasks; not comparable with 2.0
        skipped_21 += 1
        continue
    r = json.load(open(p))
    if "task_name" not in r:
        continue
    task = r["task_name"].split("/")[-1]
    task = ALIASES.get(task, task)
    vr = (r.get("verifier_result") or {}).get("rewards") or {}
    reward = vr.get("reward")
    rows.append((sub, job, task, None if reward is None else float(reward)))
print(f"trials={len(rows)} submissions={len({r[0] for r in rows})} tasks={len({r[2] for r in rows})} "
      f"skipped_tb21_trials={skipped_21}")

# --- 2. which tasks changed between 2.0 and 2.1 (instructions/tests/env/resources) ---
def to_mb(e):
    """2.0 writes memory = "2G"; 2.1 writes memory_mb = 2048. Normalize both to MB."""
    if e.get("memory_mb") is not None:
        return int(e["memory_mb"])
    m = str(e.get("memory", "")).strip().upper()
    return int(float(m[:-1]) * 1024) if m.endswith("G") else int(float(m[:-1] or 0)) if m.endswith("M") else None

def resources(path):
    c = tomllib.load(open(path, "rb"))
    e, a, v = c.get("environment", {}), c.get("agent", {}), c.get("verifier", {})
    return (float(e.get("cpus") or 0), to_mb(e), float(a.get("timeout_sec") or 0),
            float(v.get("timeout_sec") or 0))

def changed(t):
    out = []
    for sub in ["instruction.md", "tests/test.sh", "tests/test_outputs.py", "environment/Dockerfile"]:
        a, b = f"{TB20}/{t}/{sub}", f"{TB21}/{t}/{sub}"
        if os.path.exists(a) != os.path.exists(b) or (os.path.exists(a) and not filecmp.cmp(a, b, shallow=False)):
            out.append(sub.split("/")[-1])
    r20, r21 = resources(f"{TB20}/{t}/task.toml"), resources(f"{TB21}/{t}/task.toml")
    for name, x, y in zip(("cpus", "memory", "agent_timeout", "verifier_timeout"), r20, r21):
        if x != y:
            out.append(f"{name}:{x}->{y}")
    return out

# --- 3. per-task statistics ---
by_task = collections.defaultdict(list)
for sub, job, task, rew in rows:
    by_task[task].append((sub, job, rew))

def rate(xs):
    xs = [x for x in xs if x is not None]
    return (sum(x >= 1.0 for x in xs) / len(xs)) if xs else None

stats = []
for task, trs in sorted(by_task.items()):
    n = len(trs); infra = sum(r is None for _, _, r in trs)
    per_sub = collections.defaultdict(list); per_job = collections.defaultdict(list)
    for sub, job, rew in trs:
        per_sub[sub].append(rew); per_job[(sub, job)].append(rew)
    sub_rates = {s: rate(v) for s, v in per_sub.items() if sum(x is not None for x in v) >= 3}
    vals = [v for v in sub_rates.values() if v is not None]
    # run-to-run noise: std of job-level pass rates within the same submission (subs with >=2 jobs)
    noise = []
    for s in per_sub:
        jr = [rate(v) for (s2, j), v in per_job.items() if s2 == s and sum(x is not None for x in v) >= 2]
        jr = [x for x in jr if x is not None]
        if len(jr) >= 2:
            noise.append(st.pstdev(jr))
    t2 = [v for s, v in sub_rates.items() if s.lower().startswith(("terminus2", "terminus-2")) and v is not None]
    # GPT-only submissions: exclude mixed-model entries (e.g. "OB-1_GPT-5.4-...-Claude-Opus-4.6")
    gpt = [v for s, v in sub_rates.items() if v is not None and "gpt" in s.lower()
           and not any(m in s.lower() for m in ("claude", "opus", "gemini", "multiple"))]
    stats.append(dict(
        task=task, trials=n, infra_rate=round(infra / n, 3), n_subs=len(vals),
        field_rate=round(st.mean(vals), 3) if vals else None,
        spread=round(st.pstdev(vals), 3) if len(vals) > 1 else None,
        run_noise=round(st.mean(noise), 3) if noise else None,
        terminus2_rate=round(st.mean(t2), 3) if t2 else None, n_terminus2=len(t2),
        gpt_rate=round(st.mean(gpt), 3) if gpt else None,
        changed_in_2_1="|".join(changed(task)) if os.path.isdir(f"{TB21}/{task}") else "NOT_IN_2.1",
    ))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
print(f"wrote {OUT} ({len(stats)} tasks)")
