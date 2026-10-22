import datetime as dt
import json

from aees_harbor import browse, status
from aees_harbor.ingest import ingest_job
from aees_harbor.ledger import Ledger
from tests.unit.test_ingest_browse import _job


def _remote_setup(tmp_path, monkeypatch):
    """A fake EC2 remote: its ledger points at /home/ubuntu/jobs, its jobs are mirrored locally."""
    mirror_root, state_root = tmp_path / "mirror", tmp_path / "state_remote"
    jobs_mirror = mirror_root / "r1"
    jobs_mirror.mkdir(parents=True)
    job = _job(jobs_mirror)  # mirrored job dir: <mirror>/r1/job1
    remote_ledger = state_root / "r1" / "ledger.sqlite"
    remote_ledger.parent.mkdir(parents=True)
    led = Ledger(remote_ledger)
    ingest_job(led, job)
    # make the stored paths look like they came from the EC2 host
    led._db.execute("UPDATE trials SET trial_dir = replace(trial_dir, ?, '/home/ubuntu/jobs')", (str(jobs_mirror),))
    led._db.execute("UPDATE runs SET trial_dir = replace(trial_dir, ?, '/home/ubuntu/jobs')", (str(jobs_mirror),))
    led.close()
    plan = {"budget_usd": 10, "jobs_root": "/home/ubuntu/jobs", "budget_counts_jobs": [],
            "batches": [{"name": "job1", "status": "running", "attempts": 2, "concurrency": 2}]}
    (state_root / "r1" / "schedule_ec2.json").write_text(json.dumps(plan))
    (state_root / "r1" / "sync.json").write_text(json.dumps({"ok": True, "errors": [],
                                                              "ec2": {"state": "running", "hours": 2.0, "cost_usd": 3.23}}))
    cfg = tmp_path / "remotes.json"
    cfg.write_text(json.dumps({"mirror_root": str(mirror_root), "state_root": str(state_root), "remotes": [
        {"name": "r1", "host": "h", "user": "u", "key": "k", "region": "us-east-2", "instance_id": "i-1",
         "instance_type": "m7i.8xlarge", "hourly_usd": 1.6128, "remote_harbor": "/home/ubuntu/geodesic/aees/harbor",
         "remote_jobs": "/home/ubuntu/jobs", "plan": "state/schedule_ec2.json"}]}))
    return cfg


def test_browser_lists_remote_source_and_opens_its_trials_from_the_mirror(tmp_path, monkeypatch):
    cfg = _remote_setup(tmp_path, monkeypatch)
    srcs = browse.sources(tmp_path / "local.sqlite", cfg)
    assert set(srcs) == {"local", "r1"} and srcs["r1"]["remote"]
    led = Ledger(srcs["r1"]["ledger"])
    key = led.scenarios()[0]["key"]
    stored = led.trial_facts(key)[0]["trial_dir"]
    assert stored.startswith("/home/ubuntu/jobs/")
    d = browse.trial_detail(led, key, 1, 1, srcs["r1"]["rewrite"])
    assert d is not None and d["segments"], "trajectory must be read from the local mirror"
    assert d["trial_dir"].startswith(str(tmp_path / "mirror" / "r1"))


def test_status_reports_remote_api_and_ec2_cost(tmp_path, monkeypatch):
    cfg = _remote_setup(tmp_path, monkeypatch)
    [r] = status.remote_status(cfg, dt.datetime.now(dt.timezone.utc))
    assert r["config"]["name"] == "r1" and r["sync"]["ec2"]["cost_usd"] == 3.23
    assert abs(r["aggregate_cost_usd"] - 0.02) < 1e-9  # two mirrored trials at $0.01
    assert r["ledger"] == {"scenarios": 1, "trials": 2}
    assert [j["name"] for j in r["jobs"]] == []  # fake job has no config.json, so no job card


def test_selected_tasks_reads_the_list_and_tolerates_a_missing_file(tmp_path):
    f = tmp_path / "sel.txt"
    f.write_text("mailman\n\ncrack-7z-hash\n")
    assert browse.selected_tasks(f) == ["mailman", "crack-7z-hash"]
    assert browse.selected_tasks(tmp_path / "missing.txt") == []


def test_stopped_instance_cost_freezes_at_the_stop_time():
    from aees_harbor.remote import stopped_time
    t = stopped_time("User initiated (2026-10-19 19:42:00 GMT)")
    assert t == dt.datetime(2026, 10, 19, 19, 42, tzinfo=dt.timezone.utc)
    assert stopped_time("") is None and stopped_time(None) is None


def test_job_status_counts_exceptions_by_type_and_message(tmp_path):
    job = _job(tmp_path)
    trial = next(p for p in job.iterdir() if (p / "result.json").exists())
    r = json.loads((trial / "result.json").read_text())
    r["exception_info"] = {"exception_type": "RuntimeError", "exception_message": "Command timed out after 120 seconds\nmore"}
    (trial / "result.json").write_text(json.dumps(r))
    s = status.job_status(job, dt.datetime(2026, 10, 13, 12))  # fixture timestamps are naive
    assert ["RuntimeError: Command timed out after 120 seconds", 1] in [list(x) for x in s["exceptions"]]


def test_trials_and_jobs_carry_their_model_for_the_model_filter(tmp_path):
    job = _job(tmp_path)
    (job / "config.json").write_text(json.dumps({"agents": [{"name": "terminus-2", "model_name": "hosted_vllm/qwen3-4b"}]}))
    assert browse.job_model(job) == "hosted_vllm/qwen3-4b"
    assert browse.job_model(tmp_path) is None  # not a job dir: no config.json
    led = Ledger(tmp_path / "l.sqlite")
    ingest_job(led, job)
    models = {t["model"] for t in browse.trials(led, None)}
    assert models == {led.scenarios()[0]["fields"]["model"]}
    assert [j["model"] for j in browse.jobs(led, tmp_path)] == ["hosted_vllm/qwen3-4b"]


def test_test_candidates_reads_its_own_list(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("fix-git\nregex-log\n")
    assert browse.test_candidates(f) == ["fix-git", "regex-log"]
    assert browse.test_candidates(tmp_path / "missing.txt") == []
