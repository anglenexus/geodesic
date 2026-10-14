"""Load finished Harbor trials from a job directory into the ledger, unjudged.

Each trial is matched to its scenario (task / model / agent / tool mode / settings), given the
next free run index in start-time order, and logged with raw facts. Its index is put on hold
for review. Re-ingesting the same job adds nothing (trial dirs are unique in the ledger).

    python -m aees_harbor.ingest ~/code/jobs/tb21_pilot_b1 --ledger state/ledger.sqlite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aees_harbor.ledger import Ledger, Scenario

# How each agent emits actions; part of the scenario identity.
TOOL_MODES = {"terminus-2": "terminus-json-keystrokes", "oracle": "oracle-solution-script"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def scenario_of(result: dict, harbor_version: str) -> Scenario:
    agent = result["agent_info"]
    model = agent.get("model_info") or {}
    return Scenario(
        dataset=result["source"], dataset_ref=result["task_id"].get("ref", ""),
        task=result["task_name"], task_checksum=result["task_checksum"],
        model=f"{model.get('provider')}/{model.get('name')}" if model else "",
        agent=agent["name"], agent_version=agent["version"],
        tool_mode=TOOL_MODES.get(agent["name"], "unknown"), harbor_version=harbor_version,
        agent_kwargs=result["config"]["agent"].get("kwargs") or {},
    )


def raw_facts(trial: Path, result: dict, job_name: str) -> dict:
    """What Harbor recorded, copied as-is. No pass/fail/infra judgement here."""
    exc = result.get("exception_info") or {}
    agent = result.get("agent_result") or {}
    phases = {k: result.get(k) for k in ("environment_setup", "agent_setup", "agent_execution", "verifier")}
    try:
        n_steps = len(_load(trial / "agent" / "trajectory.json")["steps"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        n_steps = None
    return {
        "job": job_name, "trial_name": result.get("trial_name"),
        "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
        "exception_type": exc.get("exception_type"),
        "exception_head": (exc.get("exception_message") or "")[:500],
        "test_report": any((trial / "verifier" / f).exists() for f in ("ctrf.json", "report.json")),
        "n_steps": n_steps,
        "tokens_in": agent.get("n_input_tokens"), "tokens_cached": agent.get("n_cache_tokens"),
        "tokens_out": agent.get("n_output_tokens"), "cost_usd": agent.get("cost_usd"),
        "started_at": result.get("started_at"), "finished_at": result.get("finished_at"),
        "phases": {k: v for k, v in phases.items() if v},
    }


def ingest_job(ledger: Ledger, job_dir: Path, target_runs: int = 2000) -> dict:
    """Ingest every finished trial of one Harbor job. Returns counts of what happened."""
    job_dir = Path(job_dir).expanduser().resolve()
    try:
        harbor_version = _load(job_dir / "lock.json").get("harbor", {}).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        harbor_version = "unknown"
    pending = []
    for trial in sorted(p for p in job_dir.iterdir() if (p / "result.json").exists()):
        result = _load(trial / "result.json")
        if not result.get("finished_at"):
            continue  # still running
        pending.append((result["started_at"], trial, result))
    counts = {"ingested": 0, "already_in_ledger": 0, "skipped_target_full_or_frozen": 0}
    for _, trial, result in sorted(pending, key=lambda x: x[0]):  # run indices follow start time
        if ledger.find_trial(str(trial)):
            counts["already_in_ledger"] += 1
            continue
        key = ledger.register(scenario_of(result, harbor_version), target_runs)
        leased = ledger.lease_next(key, 1, job_name=job_dir.name)
        if not leased:
            counts["skipped_target_full_or_frozen"] += 1
            continue
        ledger.record_trial(key, leased[0], str(trial), raw_facts(trial, result, job_dir.name))
        counts["ingested"] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("job_dirs", nargs="+", type=Path)
    p.add_argument("--ledger", type=Path, default=Path("state/ledger.sqlite"))
    p.add_argument("--target-runs", type=int, default=2000)
    args = p.parse_args()
    ledger = Ledger(args.ledger)
    for job in args.job_dirs:
        print(job, ingest_job(ledger, job, args.target_runs))


if __name__ == "__main__":
    main()
