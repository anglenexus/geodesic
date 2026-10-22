import json
import subprocess
import time

import pytest

from aees_harbor import schedule


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network and no Docker needed: the preflight checks are stubbed."""
    monkeypatch.setattr(schedule, "preflight_api", lambda *a, **k: (True, "stub"))
    monkeypatch.setattr(schedule, "docker_ok", lambda: True)


def _plan(tmp_path, command, budget, batches=("b1", "b2")):
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    plan = {
        "name": "test", "jobs_root": str(jobs), "ledger": str(tmp_path / "ledger.sqlite"),
        "env_file": "unused", "model": "openai/x", "budget_usd": budget, "budget_counts_jobs": [],
        "min_free_disk_gb": 0, "no_new_batch_after": None, "watchdog_interval_s": 1,
        "cancel_grace_s": 2, "wait_for_other_harbor": False, "command_override": command,
        "batches": [{"name": b, "attempts": 1, "concurrency": 1, "status": "planned"} for b in batches],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, jobs


def _alive(marker: str) -> bool:
    return subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 0


def test_kill_switch_stops_a_harbor_that_ignores_sigint_and_its_children(tmp_path):
    jobs = tmp_path / "jobs"
    script = tmp_path / "fake_harbor.sh"
    # Ignores SIGINT (like harbor finishing in-flight trials), has a child, and spends $1.20.
    script.write_text(f"""#!/bin/bash
trap '' INT
sleep 6543 &
J={jobs}/b1; mkdir -p $J/t__1 $J/t__2
echo '{{"agent_result":{{"cost_usd":0.6}}}}' > $J/t__1/result.json
echo '{{"agent_result":{{"cost_usd":0.6}}}}' > $J/t__2/result.json
wait
""")
    script.chmod(0o755)
    plan_path, _ = _plan(tmp_path, [str(script)], budget=1.0)
    start = time.time()
    schedule.run(plan_path, dry_run=False)
    elapsed = time.time() - start
    plan = json.loads(plan_path.read_text())
    assert [b["status"] for b in plan["batches"]] == ["killed", "skipped"]
    assert "budget kill switch" in plan["batches"][0]["note"]
    assert not _alive("fake_harbor.sh") and not _alive("sleep 6543")  # whole group is gone
    assert elapsed < 30  # 1 s watchdog + 2 s grace + 30 s SIGTERM cap; normally ~4 s


def test_no_batch_starts_once_the_aggregate_budget_is_reached(tmp_path):
    plan_path, jobs = _plan(tmp_path, ["/bin/echo", "should-not-run"], budget=1.0)
    (jobs / "b1" / "t__1").mkdir(parents=True)
    (jobs / "b1" / "t__1" / "result.json").write_text(json.dumps({"agent_result": {"cost_usd": 1.5}}))
    plan = json.loads(plan_path.read_text())
    plan["batches"][0]["status"] = "done"
    plan_path.write_text(json.dumps(plan))
    schedule.run(plan_path, dry_run=False)
    plan = json.loads(plan_path.read_text())
    assert plan["batches"][1]["status"] == "stopped" and "budget reached" in plan["batches"][1]["note"]
    assert not (tmp_path / "b2.harbor.log").exists()  # never launched
