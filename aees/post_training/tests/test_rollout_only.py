"""grpo/rollout_only.py: the trial plan, the request each trial sends, and a run against a fake dispatcher."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from grpo import rollout_only

HERE = Path(__file__).resolve().parents[1]
ENGINES = [f"http://127.0.0.1:3000{i}/v1" for i in range(1, 5)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plan_is_longest_first_and_spreads_expected_minutes_over_engines():
    minutes = json.loads((HERE / "grpo/task_limits.json").read_text())
    tasks = [t.strip() for t in (HERE / "../harbor/state/tb21_selected_tasks.txt").read_text().splitlines() if t.strip()]
    trials = rollout_only.plan(tasks, 8, ENGINES, minutes)
    assert len(trials) == 160 and len({(t["task"], t["attempt"]) for t in trials}) == 160
    order = [minutes[t["task"]] for t in trials]
    assert order == sorted(order, reverse=True)                    # 30-min tasks first
    load = {e: sum(minutes[t["task"]] for t in trials if t["engine"] == e) for e in ENGINES}
    assert max(load.values()) - min(load.values()) <= max(minutes.values())
    assert {e: sum(t["engine"] == e for t in trials) for e in ENGINES} == {e: 40 for e in ENGINES}


def test_plan_without_minutes_round_robins():
    trials = rollout_only.plan(["a", "b"], 2, ENGINES[:2], {})
    assert [t["engine"] for t in trials] == [ENGINES[0], ENGINES[1], ENGINES[0], ENGINES[1]]


def test_task_limits_cover_the_training_tasks():
    limits = json.loads((HERE / "grpo/task_limits.json").read_text())
    tasks = {t.strip() for t in (HERE / "../harbor/state/tb21_selected_tasks.txt").read_text().splitlines() if t.strip()}
    assert set(limits) == tasks
    assert limits["mailman"] == 30.0 and limits["overfull-hbox"] == 12.5 and limits["password-recovery"] == 15.0


def test_request_carries_thinking_and_the_tmax_limits():
    params = rollout_only.sampling_params(16384, 1.0, 1.0, thinking=True)
    body = rollout_only.request_body({"task": "mailman", "attempt": 0, "engine": ENGINES[2]}, "openai/model", 65536, params)
    assert body["base_url"] == ENGINES[2] and body["instance_id"] == "mailman" and body["agent_name"] == "terminus-2"
    assert body["max_seq_len"] == 65536 and body["sampling_params"]["max_tokens"] == 16384
    assert body["sampling_params"]["temperature"] == 1.0 and body["sampling_params"]["top_p"] == 1.0
    assert body["sampling_params"]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    off = rollout_only.sampling_params(8192, 1.0, 1.0, thinking=False)["extra_body"]["chat_template_kwargs"]
    assert off == {"enable_thinking": False, "preserve_thinking": False}


class FakeDispatcher:
    """/run replies like the agent server: tasks named fail* score 0, err* is an infra error, the rest pass."""

    def __init__(self):
        self.bodies, self.lock = [], threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.bodies.append(body)
                task = body["instance_id"]
                reply = ({"reward": 0.0, "exit_status": "Error: Dispatch: box down"} if task.startswith("err") else
                         {"reward": 0.0, "exit_status": "TimeLimitExceeded"} if task.startswith("fail") else
                         {"reward": 1.0, "exit_status": "Submitted", "agent_metrics": {"exception_type": ""}})
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        class Server(ThreadingHTTPServer):
            request_queue_size = 64                      # every trial connects at once (the real dispatcher: 1024)

        self.port = free_port()
        self.server = Server(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_run_sends_every_trial_and_writes_one_row_each(tmp_path, capsys):
    fake = FakeDispatcher()
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("ok1\nfail1\nerr1\n")
    out = tmp_path / "rollouts.jsonl"
    rc = rollout_only.main(["--dispatcher", f"http://127.0.0.1:{fake.port}", "--engines", ",".join(ENGINES[:2]),
                            "--tasks", str(tasks), "--attempts", "2", "--out", str(out)])
    fake.server.shutdown()
    assert rc == 0 and len(fake.bodies) == 6
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 6 and all(r["start"] <= r["end"] for r in rows)
    assert {r["engine"] for r in rows} == set(ENGINES[:2])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["trials"] == 6 and summary["scored"] == 4 and summary["pass_rate"] == 0.5   # infra errors excluded
    assert summary["per_task"] == {"err1": None, "fail1": 0.0, "ok1": 1.0}


def test_summary_uses_the_grpo_infra_exclusions():
    rows = [{"task": "a", "reward": 0.0, "exit_status": "Flushed"},
            {"task": "a", "reward": 0.0, "exit_status": "TimeLimitExceeded",
             "agent_metrics": {"exception_type": "EnvironmentStartTimeoutError"}},
            {"task": "a", "reward": 0.0, "exit_status": "TimeLimitExceeded", "agent_metrics": {"exception_type": "AgentTimeoutError"}},
            {"task": "a", "reward": 1.0, "exit_status": "Submitted"}]
    s = rollout_only.summary(rows)
    assert s["scored"] == 2 and s["pass_rate"] == 0.5 and s["per_task"] == {"a": 0.5}


def test_unreachable_dispatcher_is_an_infra_error_not_a_zero(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("t\n")
    out = tmp_path / "r.jsonl"
    rollout_only.main(["--dispatcher", f"http://127.0.0.1:{free_port()}", "--engines", ENGINES[0], "--tasks", str(tasks),
                       "--attempts", "1", "--out", str(out)])
    row = json.loads(out.read_text())
    assert row["reward"] is None and row["exit_status"].startswith("Error: Driver:")


def test_dry_run_sends_nothing(tmp_path, capsys):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("a\nb\n")
    assert rollout_only.main(["--engines", ",".join(ENGINES), "--tasks", str(tasks), "--attempts", "2",
                              "--out", str(tmp_path / "x.jsonl"), "--dry-run"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["trials"] == 4 and sum(shown["per_engine"].values()) == 4 and not (tmp_path / "x.jsonl").exists()
