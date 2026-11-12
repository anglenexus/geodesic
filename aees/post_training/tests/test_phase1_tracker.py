"""Tests for monitor/phase1.py on a fake $DATA tree (rollouts.jsonl, dispatch.jsonl, trial folders) and fake
SGLang /metrics text."""
import json
import socket
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from monitor import phase1 as p1

T0 = 1_794_500_000.0
REFERENCE = json.loads((p1.HERE / "phase1_reference.json").read_text())["tasks"]
TASKS = list(REFERENCE)                                # reference order
A, B = TASKS[0], TASKS[1]


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def row(task, attempt, start, end, reward, status, **kw):
    return {"task": task, "attempt": attempt, "engine": "http://127.0.0.1:30001/v1", "start": start, "end": end,
            "reward": reward, "exit_status": status, "agent_metrics": {}, **kw}


def make_data(tmp_path, run="p1"):
    """A pass, a time-limit timeout, a failure and an infra error from the driver; the dispatcher also has one
    trial the driver hasn't written yet. Two trial folders: one finished (str-repr fields), one still running."""
    root = tmp_path / "trials" / run
    root.mkdir(parents=True)
    rollouts = [row(A, 0, T0, T0 + 600, 1.0, "Submitted"), row(A, 1, T0, T0 + 900, 0.0, "TimeLimitExceeded"),
                row(B, 0, T0, T0 + 300, 0.0, "AgentError"),
                row(B, 1, T0, T0 + 5, None, "Error: Driver: ConnectionResetError()")]
    (root / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts) + '{"task": "half-wr')
    dispatch = [{"task": A, "box": "b1:11001", "queued": T0, "start": T0 + 1, "end": T0 + 600, "reward": 1.0,
                 "exit_status": "Submitted"},
                {"task": A, "box": "b1:11001", "queued": T0, "start": T0 + 2, "end": T0 + 900, "reward": 0.0,
                 "exit_status": "TimeLimitExceeded"},
                {"task": A, "box": "b2:11001", "queued": T0, "start": T0 + 3, "end": T0 + 1000, "reward": 1.0,
                 "exit_status": "Submitted"}]                                   # not in rollouts.jsonl yet
    (root / "dispatch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dispatch))

    done = root / f"{A}__20261112-170000__abc"
    (done / "agent").mkdir(parents=True)
    (done / "result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{A}", "finished_at": iso(T0 + 600), "exception_info": None,
        "verifier_result": {"rewards": {"reward": 1.0}},
        "agent_execution": {"started_at": iso(T0 + 60), "finished_at": iso(T0 + 540)}}))
    note = p1.INTERRUPT
    steps = [{"source": "user", "message": "task"},
             {"source": "agent", "metrics": str({"prompt_tokens": 1000, "completion_tokens": 300}),
              "tool_calls": str([{"function_name": "bash", "arguments": {"keystrokes": "make\n", "duration": 1}},
                                 {"function_name": "bash", "arguments": {"keystrokes": "ls\n"}}]),
              "observation": str({"results": [{"content": f"...\n{note}\n$ "}]})},
             {"source": "agent", "metrics": {"prompt_tokens": 2000, "completion_tokens": 500},
              "tool_calls": [{"arguments": {"keystrokes": "sleep 200\n"}}],
              "observation": {"results": [{"content": f"{note}. {note}"}]}},
             {"source": "agent", "metrics": "{'prompt_tokens': 3000, 'completion_tok", "tool_calls": "[{broken",
              "observation": "plain text"}]
    (done / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))

    running = root / f"{B}__20261112-170100__def"
    running.mkdir()
    (running / "result.json").write_text('{"task_name": "' + B + '", "agent_exec')     # half-synced
    return root


METRICS = """# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="m"} 1000.0
sglang:prompt_tokens_total{model_name="m"} 50000
sglang:cached_tokens_total{model_name="m"} 40000
sglang:num_running_reqs{model_name="m",tp_rank="0"} 9
sglang:num_queue_reqs{model_name="m"} 2
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 50
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="m"} 90
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 100
sglang:time_to_first_token_seconds_sum{model_name="m"} 60
sglang:time_to_first_token_seconds_count{model_name="m"} 100
weird line without value
odd{label="a \\"quoted\\" b"} 3
"""


# ---- rows ----

def test_jsonl_rows_skips_half_written_lines(tmp_path):
    root = make_data(tmp_path)
    assert len(p1.jsonl_rows(root / "rollouts.jsonl")) == 4
    assert p1.jsonl_rows(root / "missing.jsonl") == []


def test_merge_adds_dispatch_rows_the_driver_has_not_written(tmp_path):
    root = make_data(tmp_path)
    rows = p1.merge_rows(p1.jsonl_rows(root / "rollouts.jsonl"), p1.jsonl_rows(root / "dispatch.jsonl"))
    assert len(rows) == 5
    extra = [r for r in rows if r["source"] == "dispatch"]
    assert len(extra) == 1 and extra[0]["end"] == T0 + 1000 and extra[0]["task"] == A


def test_pass_rates_exclude_infra_errors(tmp_path):
    root = make_data(tmp_path)
    rows = p1.jsonl_rows(root / "rollouts.jsonl")
    o = p1.overall(rows)
    assert (o["scored"], o["passed"], o["timeouts"]) == (3, 1, 1) and o["pass_rate"] == pytest.approx(1 / 3)
    assert o["tmax"]["rate"] == 0.24 and o["ours_nothink"]["rate"] == pytest.approx(117 / 480)
    out = p1.outcomes(rows)
    assert out["by_status"] == {"Submitted": 1, "TimeLimitExceeded": 1, "AgentError": 1}
    assert out["infra"] == {"Error: Driver: ConnectionResetError()": 1} and out["infra_total"] == 1
    assert out["passed_by_status"]["Submitted"] == 1


def test_progress_counts_and_naive_eta():
    rows = [{"start": T0, "end": T0 + 100}, {"start": T0 + 10, "end": T0 + 300}]
    live = {"boxes": {"x": {"running": 18, "free": 2}, "y": {"running": 20, "free": 0}}, "waiting": 100, "done": 2}
    p = p1.progress(rows, live, 160, T0 + 600)
    assert p["done"] == 2 and p["elapsed"] == 600 and p["eta"] == pytest.approx(600 / 2 * 158)
    assert p["dispatcher"]["running"] == 38 and p["dispatcher"]["free"] == 2 and p["dispatcher"]["waiting"] == 100
    assert p1.progress([], None, 160, T0) == {"planned": 160, "done": 0, "first_start": None, "elapsed": None,
                                              "eta": None, "last_end": None, "dispatcher": None}


# ---- trial folders ----

def test_trial_folder_parses_str_reprs_and_counts_interrupts(tmp_path):
    root = make_data(tmp_path)
    f = p1.trial_folder(root / f"{A}__20261112-170000__abc")
    assert f["task"] == A and f["finished"] and f["reward"] == 1.0 and f["agent_min"] == pytest.approx(8)
    assert f["calls"] == 3 and f["completion_tokens"] == 800 and f["prompt_tokens"] == 3000
    assert f["call_tokens"] == [300, 500] and f["commands"] == 3
    assert f["interrupts"] == 3                       # one in the str-repr observation, two in the dict one


def test_running_trial_folder_is_tolerated(tmp_path):
    root = make_data(tmp_path)
    f = p1.trial_folder(root / f"{B}__20261112-170100__def")
    assert f["task"] == B and not f["finished"] and f["calls"] == 0 and f["interrupts"] == 0


def test_folders_are_cached_by_mtime(tmp_path, monkeypatch):
    root = make_data(tmp_path)
    folders = p1.Folders(root)
    assert len(folders.refresh()) == 2
    calls = []
    monkeypatch.setattr(p1, "trial_folder", lambda f: calls.append(f) or {"task": "x"})
    folders.refresh()
    assert calls == []                                # nothing changed: nothing re-parsed
    import os
    res = root / f"{B}__20261112-170100__def" / "result.json"
    os.utime(res, (T0, T0))
    folders.refresh()
    assert [c.name for c in calls] == [res.parent.name]


def test_per_task_rows_in_reference_order(tmp_path):
    root = make_data(tmp_path)
    rows = p1.merge_rows(p1.jsonl_rows(root / "rollouts.jsonl"), p1.jsonl_rows(root / "dispatch.jsonl"))
    tasks = p1.per_task(rows, p1.Folders(root).refresh(), REFERENCE)
    assert [t["task"] for t in tasks] == TASKS and len(tasks) == 20
    a, b = tasks[0], tasks[1]
    assert (a["done"], a["scored"], a["passed"], a["timeouts"]) == (3, 3, 2, 1)
    assert a["results"] == ["pass", "timeout", "pass"]
    assert (a["trials_parsed"], a["calls_median"], a["interrupts"], a["stopped_64"]) == (1, 3, 3, 0)
    assert a["tmax"] == REFERENCE[A]["tmax"] and a["limit_min"] == REFERENCE[A]["limit_min"]
    assert (b["done"], b["scored"], b["infra"], b["passed"], b["pass_rate"]) == (2, 1, 1, 0, 0.0)
    assert b["trials_parsed"] == 0 and b["calls_median"] is None     # its folder is still running
    assert tasks[5]["done"] == 0 and tasks[5]["pass_rate"] is None


def test_behaviour_and_64_call_stops():
    fs = [{"task": "a", "finished": True, "calls": c, "completion_tokens": 10 * c, "call_tokens": [10] * c,
           "commands": c, "interrupts": i, "agent_min": 5.0, "reward": r}
          for c, i, r in [(64, 0, 0.0), (10, 2, 1.0), (20, 0, 0.0), (0, 0, None)]]    # the last: no model call
    fs.append({**fs[1], "finished": False})
    b = p1.behaviour(fs)
    assert b["trials"] == 3 and b["running"] == 1
    assert b["calls"] == {"n": 3, "median": 20, "mean": pytest.approx(94 / 3), "p90": pytest.approx(55.2), "max": 64}
    assert b["hit_64"] == pytest.approx(1 / 3) and b["any_interrupt"] == pytest.approx(1 / 3) and b["interrupts"] == 2
    assert b["call_output_tokens"]["median"] == 10 and len(b["per_trial"]) == 3


# ---- Prometheus ----

def test_parse_prometheus_and_lookups():
    m = p1.parse_prometheus(METRICS)
    assert m["sglang:num_running_reqs"] == [({"model_name": "m", "tp_rank": "0"}, 9.0)]
    assert m["odd"] == [({"label": 'a "quoted" b'}, 3.0)] and "weird" not in m
    assert p1.metric(m, p1.GEN_TOKENS) == 1000 and p1.metric(m, ("nope",)) is None
    assert p1.histogram(m, p1.TTFT) == [(0.1, 0), (0.5, 50), (1.0, 90), (float("inf"), 100)]


def test_histogram_quantile():
    b = [(0.1, 0), (0.5, 50), (1.0, 90), (float("inf"), 100)]
    assert p1.hist_quantile(b, 0.5) == pytest.approx(0.5)
    assert p1.hist_quantile(b, 0.25) == pytest.approx(0.3)            # halfway through the 0.1-0.5 bucket
    assert p1.hist_quantile(b, 0.7) == pytest.approx(0.75)
    assert p1.hist_quantile(b, 0.95) == 1.0                           # in +Inf: the top finite bound
    assert p1.hist_quantile([(1.0, 0), (float("inf"), 0)], 0.5) is None and p1.hist_quantile(None, 0.5) is None
    assert p1.bucket_delta([(1, 5), (2, 9)], [(1, 2), (2, 3)]) == [(1, 3), (2, 6)]
    assert p1.bucket_delta([(1, 1)], [(1, 2)]) is None                 # counter reset


def sample(t, gen, prompt=None, cached=None, up=True, ttft=None):
    return {"t": t, "up": up, "gen_tokens": gen, "prompt_tokens": prompt, "cached_tokens": cached,
            "cache_hit_rate": None, "running": 5, "queued": 1, "ttft": ttft}


def test_counter_rate_and_series():
    a, b = sample(T0, 1000), sample(T0 + 10, 3000)
    assert p1.counter_rate(a, b, "gen_tokens") == 200
    assert p1.counter_rate(b, sample(T0 + 20, 10), "gen_tokens") is None          # engine restarted
    assert p1.counter_rate(a, {"t": T0 + 10, "up": False}, "gen_tokens") is None
    ss = [sample(T0 + 10 * i, 1000 * i) for i in range(7)]
    assert p1.rate_series(ss, "gen_tokens", step=30) == [[T0 + 30, 100.0], [T0 + 60, 100.0]]


def test_engine_summary_recent_rates_cache_hit_and_ttft():
    old_b = [(0.5, 10), (1.0, 20), (float("inf"), 20)]
    new_b = [(0.5, 10), (1.0, 40), (float("inf"), 40)]                 # the last 20 observations all 0.5-1.0 s
    ss = [sample(T0, 0, 0, 0, ttft=old_b), sample(T0 + 100, 30000, 100000, 80000, ttft=new_b)]
    e = p1.engine_summary(30001, ss)
    assert e["up"] and e["tok_s"] == 300 and e["cache_hit"] == pytest.approx(0.8)
    assert e["ttft"]["p50"] == pytest.approx(0.75) and e["ttft"]["n"] == 20
    assert e["ttft"]["p50_all"] == pytest.approx(0.5 + 0.5 * 10 / 30) and e["itl"]["p50"] is None
    assert e["running"] == 5 and e["queued"] == 1
    down = p1.engine_summary(30002, [{"t": T0, "up": False}])
    assert not down["up"] and down["tok_s"] is None and down["series"] == []
    assert p1.engine_summary(30003, [])["up"] is False
    s = p1.serving({30001: ss, 30002: [{"t": T0, "up": False}]})
    assert s["up"] == 1 and s["tok_s"] == 300 and s["running"] == 5


def test_engine_sample_from_text():
    s = p1.engine_sample(METRICS, T0)
    assert s["up"] and s["gen_tokens"] == 1000 and s["cached_tokens"] == 40000 and s["queued"] == 2
    assert p1.engine_sample(None, T0) == {"t": T0, "up": False}


# ---- the server ----

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_serves_page_and_status_with_nothing_live(tmp_path):
    make_data(tmp_path, run="p1")
    live = p1.Live([free_port(), free_port()], f"http://127.0.0.1:{free_port()}")   # nothing listens there
    live.sample()
    live.sample()
    (tmp_path / "monitor" / "p1").mkdir(parents=True)
    (tmp_path / "monitor" / "p1" / "settings.json").write_text(json.dumps({"thinking": True, "max_model_calls": 64}))
    tracker = p1.Tracker("p1", tmp_path, live)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), p1.make_handler(tracker))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        r = urllib.request.urlopen(url + "/")
        assert r.status == 200 and b"<title>Rollout Baseline Tracker</title>" in r.read()
        r = urllib.request.urlopen(url + "/status.json?t=1")
        assert r.status == 200
        s = json.loads(r.read())
        assert set(s) == {"run", "generated", "planned", "sources", "progress", "outcomes", "overall", "tasks",
                          "behaviour", "serving", "reference", "settings", "attempts"}
        assert s["settings"] == {"thinking": True, "max_model_calls": 64}
        assert s["attempts"]["attempts"] is None and s["overall"]["tmax"]["label"] == "Tmax released eval, TB 2.0, our 20 tasks"
        assert s["behaviour"]["by_outcome"]["passed"] == {"n": 1, "calls": 3, "commands": 3, "agent_min": 8,
                                                         "output_tokens": 800}
        assert s["behaviour"]["by_outcome"]["failed"]["n"] == 0
        assert s["reference"]["tmax"]["tok_s_per_agent_during_calls"] == 47.8
        assert s["run"] == "p1" and s["planned"] == 160 and s["progress"]["done"] == 5
        assert s["progress"]["dispatcher"] is None and s["sources"]["folders"] == 2
        assert s["serving"]["up"] == 0 and s["serving"]["tok_s"] is None
        assert all(not e["up"] and e["series"] == [] for e in s["serving"]["engines"])
        assert len(s["tasks"]) == 20 and s["overall"]["passed"] == 2
        text = urllib.request.urlopen(url + "/summary.txt").read().decode()
        assert text.startswith("p1: 5/160 trials done (dispatcher not answering") and "vs Tmax 24.0%" in text
        assert "TTFT p50/p90 n/a/n/a" in text
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/etc/passwd")
    finally:
        httpd.shutdown()


def test_main_parses_arguments(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        monkeypatch.delenv("RUN", raising=False)
        p1.main(["--data", str(tmp_path)])                             # no run name


# ---- timing split, latency, persistence ----

def test_time_split_from_step_timestamps(tmp_path):
    """A Terminus-2 step is logged after its commands ran. Agent starts at 0 (a user step logged earlier doesn't move
    that back); step 1 is logged at 95 s and asked for 5 s + 100 s (capped at 60): commands 65 s, model 30 s. Step 2
    at 115 s with no usable duration: model 20 s. The agent ran 200 s: other 85 s."""
    steps = [{"source": "user", "timestamp": iso(T0 - 5)},
             {"source": "agent", "timestamp": iso(T0 + 95), "metrics": {"completion_tokens": 600},
              "tool_calls": str([{"arguments": {"keystrokes": "a", "duration": 5}},
                                 {"arguments": {"keystrokes": "b", "duration": 100}}])},
             {"source": "agent", "timestamp": iso(T0 + 115), "metrics": {"completion_tokens": 400},
              "tool_calls": [{"arguments": {"keystrokes": "c", "duration": "x"}}]}]
    st = p1.trajectory_stats(steps, start=T0)
    assert st["model_s"] == pytest.approx(50) and st["cmd_s"] == pytest.approx(65) and st["commands"] == 3
    assert p1.trajectory_stats([{"source": "agent"}])["model_s"] is None          # no timestamps: no split
    folder = tmp_path / "t__s__1"
    (folder / "agent").mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"task_name": "t", "finished_at": "x", "verifier_result": None,
        "agent_execution": {"started_at": iso(T0), "finished_at": iso(T0 + 200)}}))
    (folder / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))
    f = p1.trial_folder(folder)
    assert f["other_s"] == pytest.approx(85)
    r = p1.tok_rates([f])
    assert r["tok_s_overall"] == pytest.approx(1000 / 200) and r["tok_s_calls"] == pytest.approx(1000 / 50)
    assert (r["share_model"], r["share_cmd"], r["share_other"]) == pytest.approx((0.25, 0.325, 0.425))
    b = p1.behaviour([f])
    assert b["model_min"]["median"] == pytest.approx(50 / 60) and b["interrupt_rate_per_command"] == 0


def test_overall_counts_context_limit_stops(tmp_path):
    rows = p1.jsonl_rows(make_data(tmp_path) / "rollouts.jsonl") + [row(A, 2, T0, T0 + 1, 0.0, "SequenceLengthLimitExceeded")]
    o = p1.overall(rows)
    assert o["seqlen"] == 1 and o["seqlen_rate"] == 0.25


ITL_TEXT = """sglang:inter_token_latency_seconds_bucket{le="0.02"} 0
sglang:inter_token_latency_seconds_bucket{le="0.04"} 100
sglang:inter_token_latency_seconds_bucket{le="+Inf"} 100
sglang:generation_tokens_total 500
"""


def test_itl_histogram_and_combined_latency():
    s1, s2 = p1.engine_sample(ITL_TEXT, T0), p1.engine_sample(ITL_TEXT.replace(" 100", " 300"), T0 + 60)
    assert s1["itl"] == [(0.02, 0), (0.04, 100), (float("inf"), 100)] and s1["ttft"] is None
    e = p1.engine_summary(30001, [s1, s2])
    assert e["itl"]["p50"] == pytest.approx(0.03) and e["itl"]["n"] == 200
    c = p1.combined_latency({1: [s1, s2], 2: [s1, s2], 3: [{"t": T0, "up": False}]}, "itl")
    assert c["n"] == 400 and c["n_all"] == 600 and c["p90"] == pytest.approx(0.038)


def test_samples_are_logged_and_reloaded(tmp_path, monkeypatch):
    log = tmp_path / "monitor" / "r" / "serving.jsonl"
    texts = iter([ITL_TEXT, ITL_TEXT.replace(" 100", " 300").replace("500", "1500")])
    monkeypatch.setattr(p1, "fetch", lambda url, timeout=3: next(texts) if "/metrics" in url else None)
    live = p1.Live([30001], "http://127.0.0.1:1", log)
    monkeypatch.setattr(p1.time, "time", lambda: T0)
    live.sample()
    monkeypatch.setattr(p1.time, "time", lambda: T0 + 10)
    live.sample()
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["d"]["tok_s"] == 100
    assert json.loads(lines[0])["itl"][-1] == ["+Inf", 100]
    again = p1.Live([30001, 30002], "http://127.0.0.1:1", log)      # a restart keeps the history
    h, _, _ = again.view()
    assert [s["t"] for s in h[30001]] == [T0, T0 + 10] and h[30002] == []
    assert h[30001][1]["itl"][-1] == (float("inf"), 300) and h[30001][1]["d"]["itl_p50"] == pytest.approx(0.03)
    assert p1.latency_series(h[30001], step=5)[-1][3] == pytest.approx(0.03)
    assert p1.decode_sample("{broken") is None and p1.decode_sample('{"t": 1}') is None


def test_settings_missing_or_bad(tmp_path):
    assert p1.read_settings(tmp_path / "none.json") is None and p1.read_settings(None) is None
    (tmp_path / "bad.json").write_text("[1, 2")
    assert p1.read_settings(tmp_path / "bad.json") is None


def test_context_stops_include_engine_refusals():
    assert p1.context_stop({"exit_status": "SequenceLengthLimitExceeded"})
    assert p1.context_stop({"exit_status": "AgentError", "agent_metrics": {"exception_type": "ContextLengthExceededError"}})
    assert not p1.context_stop({"exit_status": "AgentError", "agent_metrics": {"exception_type": "RewardFileNotFoundError"}})


# ---- full TB 2.1 reference, attempts ----

TB21 = json.loads((p1.HERE / "tb21_reference.json").read_text())


def test_tb21_reference_covers_all_tasks_and_keeps_the_20():
    names = [t.strip() for t in (p1.HERE.parents[1] / "harbor/state/tb21_all_tasks.txt").read_text().splitlines()]
    limits = json.loads((p1.HERE.parent / "grpo/task_limits_tb21.json").read_text())
    assert len(names) == 89 and list(TB21["tasks"]) == names and list(limits) == names
    assert all(TB21["tasks"][t]["limit_min"] == limits[t] for t in names)
    old = json.loads((p1.HERE / "phase1_reference.json").read_text())
    for t, v in old["tasks"].items():                  # same derivation: the 20 tasks' entries are unchanged
        assert TB21["tasks"][t]["tmax"] == v["tmax"] and TB21["tasks"][t]["ours_nothink"] == v["ours_nothink"]
        assert TB21["tasks"][t]["limit_min"] == v["limit_min"]
    assert sum("ours_nothink" in v for v in TB21["tasks"].values()) == 20
    assert TB21["overall"]["tmax_paper"]["pass_rate"] == 0.436


def test_overall_headline_is_the_paper_number_with_tb21_reference():
    o = p1.overall([row(A, 0, T0, T0 + 1, 1.0, "Submitted")], TB21["tasks"], TB21["overall"])
    assert o["tmax"]["rate"] == 0.436 and o["tmax"]["n"] is None and o["tmax"]["label"] == "Tmax paper, TB 2.1, all tasks"
    rec = o["tmax_records"]
    assert rec["n"] == sum(v["tmax"]["n"] for v in TB21["tasks"].values()) and rec["tasks"] == 89
    assert o["ours_nothink"]["rate"] == pytest.approx(117 / 480) and "20 tasks only" in o["ours_nothink"]["label"]
    o20 = p1.overall([], REFERENCE, {})                # the 20-task reference: records are the headline, as before
    assert o20["tmax"]["rate"] == 0.24 and o20["tmax"]["n"] == 100 and o20["ours_nothink"]["label"] == "thinking off"


def test_attempts_and_extension(tmp_path):
    ext = tmp_path / "extend"
    a = p1.attempts({"attempts": 5, "extend_attempts": 8}, ext, 445)
    assert a == {"attempts": 5, "extend_attempts": 8, "extend_possible": True, "extend_requested": False, "planned": 445}
    ext.touch()
    a = p1.attempts({"attempts": 5, "extend_attempts": 8}, ext, 445)
    assert a["extend_requested"] and a["planned"] == 712
    assert p1.attempts({"attempts": 8, "extend_attempts": 8}, ext, 160)["extend_possible"] is False
    assert p1.attempts(None, ext, 160) == {"attempts": None, "extend_attempts": None, "extend_possible": False,
                                           "extend_requested": False, "planned": 160}


def test_server_with_tb21_reference_and_extension(tmp_path):
    root = make_data(tmp_path, run="full")
    (tmp_path / "monitor" / "full").mkdir(parents=True)
    (tmp_path / "monitor" / "full" / "settings.json").write_text(json.dumps({"tasks": 89, "attempts": 5,
                                                                            "extend_attempts": 8}))
    live = p1.Live([], f"http://127.0.0.1:{free_port()}")
    tracker = p1.Tracker("full", tmp_path, live, 445, p1.HERE / "tb21_reference.json")
    s = tracker.snapshot()
    assert len(s["tasks"]) == 89 and s["planned"] == 445 and not s["attempts"]["extend_requested"]
    assert s["sources"]["reference"] == "tb21_reference.json" and s["reference"]["tmax_paper"]["pass_rate"] == 0.436
    (root / "extend").touch()
    s = tracker.snapshot()
    assert s["planned"] == 712 and s["progress"]["planned"] == 712
    text = p1.summary_text(s)
    assert "vs Tmax paper, TB 2.1, all tasks 43.6% ± 1.9 (released records" in text
    assert "extension to 8 attempts requested" in text
