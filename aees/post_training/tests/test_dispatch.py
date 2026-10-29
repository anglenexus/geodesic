"""grpo/dispatch.py against fake agent servers: order, per-box limits, failures, broadcast, logging."""
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler

import pytest

from grpo import dispatch
from grpo.infra_filter import infra_cause


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeBox:
    """An agent server: /run sleeps `delay` s (per task) and records who ran when; /health; /flush."""

    def __init__(self, delay=None, reward=1.0, drop=False):
        self.delay, self.reward, self.drop = delay or {}, reward, drop
        self.started, self.running, self.peak, self.flushed = [], 0, 0, []
        self.lock = threading.Lock()
        box = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                self._send({"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/run":
                    box.flushed.append((self.path, body, self.headers.get("Authorization")))
                    return self._send({"flushed": 1})
                with box.lock:
                    box.started.append(body["instance_id"])
                    box.running += 1
                    box.peak = max(box.peak, box.running)
                time.sleep(box.delay.get(body["instance_id"], 0.05))
                with box.lock:
                    box.running -= 1
                if box.drop:
                    self.connection.close()            # dies mid-trial
                    return
                self._send({"reward": box.reward, "exit_status": "Submitted", "eval_report": {}, "agent_metrics": {}})

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.port = free_port()
        self.server = dispatch.Server(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def addr(self):
        return f"127.0.0.1:{self.port}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def start(tmp_path):
    """start(boxes, slots, minutes) -> (base_url, log path); the dispatcher runs in a thread."""
    servers = []

    def _start(boxes, slots=2, minutes=None):
        port = free_port()
        log = tmp_path / "dispatch.jsonl"
        pool = dispatch.Pool([b if isinstance(b, str) else b.addr for b in boxes], slots, minutes or {})
        srv = dispatch.Server(("127.0.0.1", port), dispatch.make_handler(dispatch.Dispatcher(pool, str(log))))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{port}", log

    yield _start
    for s in servers:
        s.shutdown()


def run_trial(base, task, out=None):
    body = json.dumps({"instance_id": task, "base_url": "http://127.0.0.1:30000/v1"}).encode()
    reply = json.loads(dispatch.post(f"{base}/run", body))
    if out is not None:
        out.append((task, reply))
    return reply


def run_all(base, tasks, gap=0.0):
    out, threads = [], []
    for t in tasks:
        th = threading.Thread(target=run_trial, args=(base, t, out))
        th.start()
        threads.append(th)
        time.sleep(gap)
    for th in threads:
        th.join(30)
    return out


def test_longest_expected_task_starts_first_once_slots_free(start):
    box = FakeBox(delay={"blocker": 0.4})
    base, _ = start([box], slots=1, minutes={"short": 3.0, "mid": 10.0, "long": 30.0})
    th = threading.Thread(target=run_trial, args=(base, "blocker"))
    th.start()
    time.sleep(0.1)                                    # the blocker holds the only slot; the rest queue
    out = run_all(base, ["short", "mid", "long"], gap=0.02)
    th.join()
    assert box.started == ["blocker", "long", "mid", "short"]
    assert all(r["reward"] == 1.0 for _, r in out)


def test_unknown_task_gets_the_mean_and_observed_minutes_take_over():
    pool = dispatch.Pool(["a:1"], 1, {"x": 10.0, "y": 20.0})
    assert pool.expected("new") == 15.0
    pool.acquire("x")
    pool.release("a:1", "x", 2.0)
    pool.acquire("x")
    pool.release("a:1", "x", 4.0)
    assert pool.expected("x") == 3.0 and pool.expected("y") == 20.0
    assert dispatch.Pool(["a:1"], 1, {}).expected("anything") == 0.0


def test_never_more_than_slots_per_box_and_load_is_spread(start):
    boxes = [FakeBox(delay={f"t{i}": 0.2 for i in range(12)}) for _ in range(3)]
    base, log = start(boxes, slots=2)
    out = run_all(base, [f"t{i}" for i in range(12)])
    assert len(out) == 12
    assert all(b.peak <= 2 for b in boxes)
    assert all(len(b.started) == 4 for b in boxes)    # least-loaded box first
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 12 and {r["box"] for r in rows} == {b.addr for b in boxes}
    assert all(r["queued"] <= r["start"] <= r["end"] and r["reward"] == 1.0 for r in rows)


def test_a_box_that_refuses_is_skipped(start):
    good = FakeBox()
    dead = f"127.0.0.1:{free_port()}"                  # nothing listens
    base, log = start([dead, good], slots=4)
    out = run_all(base, ["a", "b", "c"])
    assert all(r["exit_status"] == "Submitted" for _, r in out)
    assert sorted(good.started) == ["a", "b", "c"]
    assert {json.loads(l)["box"] for l in log.read_text().splitlines()} == {good.addr}


def test_all_boxes_down_or_dropped_mid_trial_is_an_infra_error(start):
    base, _ = start([f"127.0.0.1:{free_port()}"], slots=1)
    r = run_trial(base, "a")
    assert r["exit_status"].startswith("Error: Dispatch") and infra_cause(r) is not None
    dropper = FakeBox(drop=True)
    base2, _ = start([dropper], slots=1)
    r = run_trial(base2, "a")
    assert r["exit_status"].startswith("Error: Dispatch") and infra_cause(r) is not None
    r = run_trial(base2, "b")                          # the slot was released
    assert r["exit_status"].startswith("Error: Dispatch")


def test_flush_goes_to_every_box_with_the_admin_header(start):
    boxes = [FakeBox(), FakeBox()]
    base, _ = start(boxes)
    req = urllib.request.Request(f"{base}/flush", data=json.dumps({"session_server_instance_id": "s1"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer k"},
                                 method="POST")
    out = json.loads(urllib.request.urlopen(req).read())
    assert set(out) == {b.addr for b in boxes}
    assert all(b.flushed == [("/flush", {"session_server_instance_id": "s1"}, "Bearer k")] for b in boxes)


def test_health_and_status(start):
    up = FakeBox()
    down = f"127.0.0.1:{free_port()}"
    base, _ = start([up, down], slots=3)
    h = dispatch.get(f"{base}/health")
    assert h["status"] == "ok" and h["boxes_up"] == {up.addr: True, down: False}
    assert h["boxes"][up.addr] == {"running": 0, "free": 3} and h["waiting"] == 0
    run_trial(base, "a")
    assert dispatch.get(f"{base}/status")["done"] == 1


def test_160_trials_at_once_none_dropped(start):
    boxes = [FakeBox() for _ in range(3)]
    base, log = start(boxes, slots=32)
    out = run_all(base, [f"t{i % 20}" for i in range(160)])
    assert len(out) == 160 and all(r["exit_status"] == "Submitted" for _, r in out)
    assert sum(len(b.started) for b in boxes) == 160 and len(log.read_text().splitlines()) == 160


def test_seed_minutes_cover_the_training_tasks():
    from pathlib import Path
    from grpo.split_tasks import LISTS, STATE, read_list
    minutes = json.loads((Path(__file__).resolve().parents[1] / "grpo/task_minutes.json").read_text())
    tasks = read_list(STATE / LISTS["train"])
    assert sorted(minutes) == sorted(tasks) and all(m > 0 for m in minutes.values())
