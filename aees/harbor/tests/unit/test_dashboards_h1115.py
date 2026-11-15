"""The three trackers use the 2026-11-14 split and count running trials under their task (h1115 run)."""
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

HARBOR = Path(__file__).resolve().parents[2]
STATE = HARBOR / "state"
sys.path.insert(0, str(HARBOR))

from aees_harbor import browse, status  # noqa: E402


def _phase1():
    spec = importlib.util.spec_from_file_location("phase1_tmax", HARBOR / "monitor_tmax" / "phase1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _split(name):
    return (STATE / f"tb21_split1114_{name}.txt").read_text().split()


def test_browser_and_tracker_lists_are_the_nov14_split():
    assert browse.selected_tasks() == _split("train")
    assert browse.test_candidates() == _split("test")
    assert browse.validation_tasks() == _split("val")
    assert len(set(_split("train")) | set(_split("test")) | set(_split("val"))) == 48  # disjoint, 20 + 20 + 8


def test_status_counts_running_trials_under_their_task(tmp_path):
    job = tmp_path / "tb21_h1115_i2_b1"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({"n_attempts": 50}))
    done = job / "polyglot-rust-c__20261115-081948__AbCdEfG"
    done.mkdir()
    (done / "result.json").write_text(json.dumps({
        "task_name": "terminal-bench/polyglot-rust-c", "started_at": "2026-11-15T08:20:00Z",
        "finished_at": "2026-11-15T08:25:00Z", "verifier_result": {"rewards": {"reward": 1.0}}}))
    (job / "count-dataset-tokens__20261115-081948__7ayN7YU").mkdir()  # running: no result.json yet
    (job / "count-dataset-tokens__20261115-081949__Jq449JN").mkdir()
    cells = status.job_status(job, dt.datetime(2026, 11, 15, 9, tzinfo=dt.timezone.utc))["cells"]
    names = list(cells) if isinstance(cells, dict) else [c.get("task") for c in cells]
    assert sorted({n.split("/")[-1] for n in names}) == ["count-dataset-tokens", "polyglot-rust-c"]


def test_tmax_reference_is_the_training_split():
    ref = json.loads((HARBOR / "monitor_tmax" / "h1115_reference.json").read_text())
    assert list(ref["tasks"]) == _split("train")
    assert "tmax_paper" not in ref["overall"]  # the paper's 43.6% is over all 89 tasks


def test_tmax_headline_is_records_over_these_20_tasks():
    p1 = _phase1()
    ref = json.loads((HARBOR / "monitor_tmax" / "h1115_reference.json").read_text())
    o = p1.overall([], ref["tasks"], ref["overall"], [])
    k = sum(v["tmax"]["passed"] for v in ref["tasks"].values())
    n = sum(v["tmax"]["n"] for v in ref["tasks"].values())
    assert (o["tmax"]["passed"], o["tmax"]["n"]) == (k, n)
    assert o["tmax"]["tasks"] == 20


def test_tmax_per_task_rows_follow_the_split():
    p1 = _phase1()
    ref = json.loads((HARBOR / "monitor_tmax" / "h1115_reference.json").read_text())
    rows = [{"task": "write-compressor", "reward": 1.0, "exit_status": "Completed"}]
    names = [r["task"] for r in p1.per_task(rows, [], ref["tasks"])]
    assert names == _split("train")


def test_tmax_tracker_takes_planned_from_the_feeder(tmp_path):
    p1 = _phase1()
    (tmp_path / "monitor" / "h1115").mkdir(parents=True)
    (tmp_path / "trials" / "h1115").mkdir(parents=True)
    (tmp_path / "monitor" / "h1115" / "settings.json").write_text(json.dumps({"planned": 8700, "attempts": 50}))
    t = p1.Tracker("h1115", tmp_path, p1.Live([], ""), 1000, HARBOR / "monitor_tmax" / "h1115_reference.json")
    assert t.snapshot()["planned"] == 8700
