"""Tests for monitor/server.py on fake run folders laid out as on the box ($DATA/ckpt, trials, monitor)."""
import json
import random
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from monitor import server as srv

T0 = 1_790_000_000.0
OUTCOME_FIELDS = {  # outcome -> (exception_type, reward) as Harbor writes them
    "passed": ("", 1.0), "failed": ("", 0.0), "timeout": ("AgentTimeoutError", 0.0),
    "passed_after_timeout": ("AgentTimeoutError", 1.0), "context_limit": ("ContextLengthExceededError", None),
    "error": ("RuntimeError", None), "infra": ("EnvironmentStartTimeoutError", None), "no_reward": ("", None),
}
EXPECTED = {"passed_after_timeout": "passed", "no_reward": "error"}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_event(f, t: float, metrics: dict):
    f.write(json.dumps({"timestamp": iso(t), "source": {}, "type": "metric", "metrics": metrics},
                       separators=(",", ":")) + "\n")


def write_trial(folder, task: str, kind: str, start: float, end: float, turns=7, parse_errors=0):
    exc, reward = OUTCOME_FIELDS[kind]
    (folder / "agent").mkdir(parents=True)
    (folder / "agent" / "trajectory.json").write_text(json.dumps(
        {"steps": [{"observation": srv.PARSE_ERROR_PREFIX + ": bad JSON"}] * parse_errors}))
    (folder / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{task}", "started_at": iso(start), "finished_at": iso(end),
        "exception_info": {"exception_type": exc} if exc else None,
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "agent_result": {"n_input_tokens": 9000, "n_output_tokens": 100 * turns, "metadata": {"n_episodes": turns}}}))


def make_run(data, rng: random.Random, run="r1", steps=4, tasks=("a", "b", "c"), n=4):
    """Fake run: per step, trials end inside the rollout window, then a rollout and a train event.
    Returns {trial folder name: true step} and the step times."""
    events = data / "ckpt" / run / "events"
    events.mkdir(parents=True)
    truth, times, t = {}, {}, T0
    with (events / "rollout.jsonl").open("w") as fr, (events / "train.jsonl").open("w") as ft:
        fr.write(json.dumps({"timestamp": iso(t), "type": "data_source_issued_samples", "x": 1}) + "\n")
        for k in range(steps):
            t_roll = t + rng.uniform(600, 3000)
            for task in tasks:
                for i in range(n):
                    end = rng.uniform(t + 1, t_roll)
                    name = f"{task}__s{k}_{i}"
                    write_trial(data / "trials" / run / name, task, rng.choice(list(OUTCOME_FIELDS)),
                                end - rng.uniform(60, 500), end, turns=rng.randint(1, 30), parse_errors=rng.randint(0, 2))
                    truth[name] = k
            write_event(fr, t_roll, {"rollout/step": k, "rollout/raw_reward": rng.random()})
            t = t_roll + rng.uniform(100, 900)
            write_event(ft, t, {"rollout/step": k, "train/step": k, "train/loss": rng.random(), "train/grad_norm": 1.0})
            times[k] = (t_roll, t)
    return truth, times


def test_event_log_reads_steps_and_only_new_bytes(tmp_path):
    truth, times = make_run(tmp_path, random.Random(0))
    log = srv.EventLog(tmp_path / "ckpt" / "r1" / "events")
    steps = log.refresh()
    assert sorted(steps) == [0, 1, 2, 3]
    for k, (t_roll, t_end) in times.items():
        assert steps[k]["t_rollout"] == pytest.approx(t_roll) and steps[k]["t_end"] == pytest.approx(t_end)
        assert {"train/loss", "rollout/raw_reward", "train/grad_norm"} <= set(steps[k])
    f = tmp_path / "ckpt" / "r1" / "events" / "train.jsonl"
    with f.open("a") as out:
        write_event(out, T0 + 1e6, {"rollout/step": 4, "train/loss": 0.5})
        out.write('{"timestamp":"' + iso(T0 + 2e6) + '","type":"metric","metrics":{"rollout/st')   # half-written
    steps = log.refresh()
    assert steps[4]["train/loss"] == 0.5 and "t_rollout" not in steps[4] and 5 not in steps
    with f.open("a") as out:
        out.write('ep":5,"train/loss":0.25}}\n')
    assert log.refresh()[5]["train/loss"] == 0.25
    assert log.refresh()[4]["train/loss"] == 0.5                       # nothing re-added


@pytest.mark.parametrize("kind", list(OUTCOME_FIELDS))
def test_trial_outcomes(tmp_path, kind):
    write_trial(tmp_path / "t", "mailman", kind, T0, T0 + 90, turns=5, parse_errors=2)
    row = srv.trial_row(tmp_path / "t")
    assert row["outcome"] == EXPECTED.get(kind, kind)
    assert (row["task"], row["turns"], row["tokens_out"], row["parse_errors"]) == ("mailman", 5, 500, 2)
    assert row["end"] - row["start"] == pytest.approx(90)


def test_unfinished_or_broken_trials_are_skipped(tmp_path):
    (tmp_path / "running").mkdir()
    assert srv.trial_row(tmp_path / "running") is None
    write_trial(tmp_path / "t", "a", "passed", T0, T0 + 1)
    r = json.loads((tmp_path / "t" / "result.json").read_text())
    (tmp_path / "t" / "result.json").write_text(json.dumps({**r, "finished_at": None}))
    assert srv.trial_row(tmp_path / "t") is None
    (tmp_path / "t" / "result.json").write_text("{not json")
    assert srv.trial_row(tmp_path / "t") is None


@pytest.mark.parametrize("seed", range(50))
def test_trials_land_in_their_step(tmp_path, seed):
    rng = random.Random(seed)
    truth, times = make_run(tmp_path, rng, steps=rng.randint(1, 6), n=rng.randint(1, 5))
    steps = srv.EventLog(tmp_path / "ckpt" / "r1" / "events").refresh()
    trials = srv.Trials(tmp_path / "trials" / "r1", "*/result.json").refresh()
    srv.assign_steps(trials, steps)
    assert {t["trial"]: t["step"] for t in trials} == truth
    late = {"end": max(t_end for _, t_end in times.values()) + 5}
    srv.assign_steps([late], steps)
    assert late["step"] == len(times)                                  # the step in progress


def test_trials_read_once_and_logs_ignored(tmp_path, monkeypatch):
    make_run(tmp_path, random.Random(1), steps=2)
    (tmp_path / "trials" / "r1" / "agent_server.log").write_text("log")
    trials = srv.Trials(tmp_path / "trials" / "r1", "*/result.json")
    first = trials.refresh()
    calls = []
    monkeypatch.setattr(srv, "trial_row", lambda p: calls.append(p))
    assert len(trials.refresh()) == len(first) == 2 * 3 * 4 and calls == []


def test_evals_and_checkpoints(tmp_path):
    ev = tmp_path / "eval" / "r1"
    for ck, sett, kind in [(9, "val", "passed"), (9, "test", "failed"), (19, "val", "timeout")]:
        write_trial(ev / f"iter_{ck:07d}" / sett / f"x__{ck}{sett}", "x", kind, T0, T0 + 1)
    rows = srv.Trials(ev, "iter_*/*/*/result.json").refresh()
    srv.label_evals(rows)
    assert sorted((r["checkpoint"], r["set"], r["outcome"]) for r in rows) == [
        (9, "test", "failed"), (9, "val", "passed"), (19, "val", "timeout")]
    for k in [19, 9]:
        (tmp_path / "ckpt" / f"iter_{k:07d}" / "adapter").mkdir(parents=True)
    (tmp_path / "ckpt" / "iter_0000029").mkdir()                        # no adapter yet: not a checkpoint
    assert [c["step"] for c in srv.checkpoints(tmp_path / "ckpt")] == [9, 19]


def test_tracker_status_and_http(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_run", lambda cmd: None)                 # no nvidia-smi / docker here
    truth, _ = make_run(tmp_path, random.Random(2), run="grpo_x")
    write_trial(tmp_path / "trials" / "eval" / "grpo_x" / "iter_0000003" / "val" / "a__v", "a", "passed", T0, T0 + 1)
    tracker = srv.Tracker("grpo_x", tmp_path, num_rollout=50)
    tracker.refresh()
    tracker.refresh()
    s = json.loads(tracker.body)
    assert (s["run"], s["num_rollout"], len(s["steps"]), len(s["trials"]), len(s["evals"])) == ("grpo_x", 50, 4, 48, 1)
    assert {t["trial"]: t["step"] for t in s["trials"]} == truth and s["evals"][0]["checkpoint"] == 3
    assert len(s["host"]) == 2 and s["host"][0]["gpu_mem_gb"] is None and s["host"][0]["disk_free_gb"] > 0
    assert len((tmp_path / "monitor" / "grpo_x" / "host.jsonl").read_text().splitlines()) == 2
    assert [len(s["sets"][k]) for k in ["train", "val", "test"]] == [20, 8, 20]
    assert srv.Tracker("grpo_x", tmp_path, 50).host.samples == s["host"]   # history survives a restart

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(tracker))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert b"Training Tracker" in urllib.request.urlopen(url + "/").read()
        assert json.loads(urllib.request.urlopen(url + "/status.json?t=1").read())["run"] == "grpo_x"
        r = urllib.request.urlopen(url + "/download")
        assert "grpo_x_status.json" in r.headers["Content-Disposition"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/etc/passwd")
    finally:
        httpd.shutdown()
