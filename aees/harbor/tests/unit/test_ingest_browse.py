import json

from aees_harbor.browse import derive_actions, trial_detail, trials
from aees_harbor.ingest import ingest_job
from aees_harbor.ledger import Ledger


def _trial(job, name, task, started, reward, steps):
    d = job / name
    (d / "agent").mkdir(parents=True)
    (d / "verifier").mkdir()
    (d / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"summary": {"tests": 1, "passed": int(reward)}, "tests": []}}))
    (d / "verifier" / "reward.txt").write_text(str(int(reward)))
    (d / "agent" / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.8", "steps": steps}))
    (d / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{task}", "source": "terminal-bench/terminal-bench-2-1",
        "task_id": {"ref": "sha256:abc"}, "task_checksum": "c1", "trial_name": name,
        "agent_info": {"name": "terminus-2", "version": "2.0.0", "model_info": {"provider": "openai", "name": "gpt-5.6-luna"}},
        "config": {"agent": {"kwargs": {"store_all_messages": True}}},
        "verifier_result": {"rewards": {"reward": reward}}, "agent_result": {"cost_usd": 0.01},
        "started_at": started, "finished_at": started.replace("00:00", "00:05"),
    }))


REPLY = json.dumps({"analysis": "look", "plan": "list files",
                    "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}], "task_complete": True})
STEPS = [{"step_id": 1, "source": "user", "message": "task"},
         {"step_id": 2, "source": "agent", "message": REPLY,
          "observation": {"results": [{"content": "total 0"}]}}]


def _job(tmp_path):
    job = tmp_path / "job1"
    job.mkdir()
    (job / "lock.json").write_text(json.dumps({"harbor": {"version": "0.23.0"}}))
    _trial(job, "kv__B", "kv", "2026-10-13T10:00:00", 1.0, STEPS)  # started second
    _trial(job, "kv__A", "kv", "2026-10-13T09:00:00", 0.0, STEPS)  # started first -> run 1
    return job


def test_derive_actions_from_terminus_reply():
    d = derive_actions(REPLY)
    assert d["analysis"] == "look" and d["plan"] == "list files"
    assert [a["name"] for a in d["actions"]] == ["bash_keystrokes", "mark_task_complete"]
    assert d["actions"][0]["arguments"] == {"keystrokes": "ls -la\n", "duration": 0.1}


def test_derive_actions_tolerates_text_around_json_and_rejects_non_terminus():
    assert derive_actions("Sure!\n" + REPLY + "\nthanks")["plan"] == "list files"
    assert derive_actions("plain text, no json") is None
    assert derive_actions(json.dumps({"other": 1})) is None


def test_ingest_assigns_indices_by_start_time_and_is_idempotent(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    job = _job(tmp_path)
    assert ingest_job(led, job)["ingested"] == 2
    assert ingest_job(led, job) == {"ingested": 0, "already_in_ledger": 2, "skipped_target_full_or_frozen": 0}
    [s] = led.scenarios()
    assert s["held"] == [1, 2] and s["next_index"] == 3
    by_index = {t["run_index"]: t["facts"]["trial_name"] for t in led.trial_facts(s["key"])}
    assert by_index == {1: "kv__A", 2: "kv__B"}


def test_browser_views_read_back_trial_with_derived_actions(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    ingest_job(led, _job(tmp_path))
    key = led.scenarios()[0]["key"]
    listed = trials(led, None)
    assert [(t["run_index"], t["status"], t["reward"]) for t in listed] == [(1, "held", 0.0), (2, "held", 1.0)]
    d = trial_detail(led, key, 2, 1)
    step = d["segments"][0]["trajectory"]["steps"][1]
    assert [a["name"] for a in step["derived"]["actions"]] == ["bash_keystrokes", "mark_task_complete"]
    assert d["verifier"]["reward_txt"] == "1" and d["facts"]["reward"] == 1.0
    assert trial_detail(led, key, 9, 1) is None


def test_ingest_endpoint_only_accepts_job_dirs_directly_under_jobs_root(tmp_path):
    import pytest
    from aees_harbor.browse import job_dir
    root = tmp_path / "jobs"
    root.mkdir()
    job = _job(root)  # creates root/job1 with lock.json
    assert job_dir(root, "job1") == job.resolve()
    for bad in ["", ".", "..", "../x", "job1/kv__A", "missing"]:
        with pytest.raises(ValueError):
            job_dir(root, bad)
    (root / "not_a_job").mkdir()
    with pytest.raises(ValueError):
        job_dir(root, "not_a_job")
