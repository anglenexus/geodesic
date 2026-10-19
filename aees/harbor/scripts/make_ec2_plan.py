"""Make this instance's schedule from the EC2 template: its task share, budget share, job names.

    python scripts/make_ec2_plan.py --instance 1 --instances 2      # writes state/schedule_ec2.json

Tasks are split round-robin (task i goes to instance i % N + 1), so instances never run the same
task, i.e. never share a ledger scenario, and their ledgers can be merged without index clashes.
Job names get an `i<K>_` prefix so job folders from different instances never collide.
Refuses to overwrite an existing plan (it may be in use by a running scheduler) unless --force.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # aees/harbor

p = argparse.ArgumentParser()
p.add_argument("--instance", type=int, required=True)
p.add_argument("--instances", type=int, default=1)
p.add_argument("--template", type=Path, default=HERE / "ec2" / "schedule_ec2_10k.json")
p.add_argument("--tasks", type=Path, default=HERE / "state" / "tb21_selected_tasks.txt")
p.add_argument("--budget-total", type=float, default=None, help="default: the template's budget_usd")
p.add_argument("--max-concurrency", type=int, default=None,
               help="cap every batch's concurrency (for smaller instances, e.g. 1.5 x vCPUs)")
p.add_argument("--out", type=Path, default=HERE / "state" / "schedule_ec2.json")
p.add_argument("--force", action="store_true")
a = p.parse_args()

if not 1 <= a.instance <= a.instances:
    sys.exit(f"--instance must be in 1..{a.instances}")
if a.out.exists() and not a.force:
    sys.exit(f"{a.out} already exists (a scheduler may be using it); pass --force to replace it")

plan = json.loads(a.template.read_text())
tasks = [t.strip() for t in a.tasks.read_text().split() if t.strip()]
mine = [t for i, t in enumerate(tasks) if i % a.instances == a.instance - 1]
share = len(mine) / len(tasks)

tasks_file = HERE / "state" / f"tasks_i{a.instance}.txt"
tasks_file.write_text("\n".join(mine) + "\n")
budget_total = a.budget_total if a.budget_total is not None else plan["budget_usd"]

plan["name"] = f"{plan['name']} | instance {a.instance} of {a.instances}: {len(mine)} tasks"
plan["tasks_file"] = str(tasks_file.relative_to(HERE))
plan["budget_usd"] = math.ceil(budget_total * share)
plan["instance"] = {"id": a.instance, "of": a.instances, "tasks": mine}
for b in plan["batches"]:
    b["name"] = f"i{a.instance}_{b['name']}"
    b["est_trials"] = b["attempts"] * len(mine)
    if a.max_concurrency:
        b["concurrency"] = min(b["concurrency"], a.max_concurrency)
    b["status"] = "planned"
a.out.write_text(json.dumps(plan, indent=2))

total = sum(b["est_trials"] for b in plan["batches"])
print(f"instance {a.instance}/{a.instances}: {len(mine)} tasks -> {a.out}")
print(f"  tasks: {', '.join(mine)}")
print(f"  {total} trajectories in {len(plan['batches'])} batches, aggregate budget ${plan['budget_usd']}, "
      f"concurrency {sorted({b['concurrency'] for b in plan['batches']})}")
