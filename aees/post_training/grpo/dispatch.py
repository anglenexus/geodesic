"""Agent-server dispatcher for GRPO with sandboxes on several EC2 boxes (dist_setup.md).

Miles posts every trial to one URL. This server holds them, starts the longest expected task first, and forwards
each to a box with a free slot (through the H200's ssh tunnels). One JSONL row per trial for the tracker.

  python3 grpo/dispatch.py --backends 127.0.0.1:11001,127.0.0.1:11002,127.0.0.1:11003 --slots 32 \
      --minutes grpo/task_minutes.json --log $DATA/trials/$RUN/dispatch.jsonl
Standard library only (host python3), like monitor/server.py.
"""
import argparse
import heapq
import itertools
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 11000                                           # what Miles' AGENT_SERVER_URL points at (scripts/run.sh)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024                          # Miles opens all 160 trial connections at once (default 5)


class Pool:
    """Free slots per box and the waiting trials, longest expected first."""

    def __init__(self, backends: list[str], slots: int, minutes: dict[str, float]):
        self.free = {b: slots for b in backends}
        self.slots, self.seed, self.seen = slots, dict(minutes), {}
        self.waiting, self.seq = [], itertools.count()
        self.cv = threading.Condition()

    def expected(self, task: str) -> float:
        """This run's mean minutes for the task once it has one, else the seed, else the seeds' mean."""
        if self.seen.get(task):
            return statistics.mean(self.seen[task])
        return self.seed.get(task, statistics.mean(self.seed.values()) if self.seed else 0.0)

    def acquire(self, task: str, avoid: set[str] = frozenset()) -> str:
        """Block until this trial is the longest waiting one and a box has a slot; returns the box."""
        with self.cv:
            entry = (-self.expected(task), next(self.seq))
            heapq.heappush(self.waiting, entry)
            while self.waiting[0] != entry or not self._open(avoid):
                self.cv.wait()
            heapq.heappop(self.waiting)
            box = max(self._open(avoid), key=self.free.get)   # least-loaded box
            self.free[box] -= 1
            self.cv.notify_all()
            return box

    def _open(self, avoid: set[str]) -> list[str]:
        return [b for b, n in self.free.items() if n > 0 and b not in avoid]

    def release(self, box: str, task: str | None = None, minutes: float | None = None) -> None:
        with self.cv:
            self.free[box] += 1
            if task is not None and minutes is not None:
                self.seen.setdefault(task, []).append(minutes)
            self.cv.notify_all()

    def status(self) -> dict:
        with self.cv:
            return {"boxes": {b: {"running": self.slots - n, "free": n} for b, n in self.free.items()},
                    "waiting": len(self.waiting)}


def error_reply(msg: str) -> dict:
    """An agent-server style reply that grpo/infra_filter.py excludes from training ("Error: ...")."""
    return {"reward": 0.0, "exit_status": f"Error: Dispatch: {msg}", "eval_report": {}, "agent_metrics": {}}


def post(url: str, body: bytes, timeout: float | None = None) -> bytes:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Dispatcher:
    """Runs one trial per /run call: queue, forward, log. A box that refuses the connection is skipped and
    the trial retried on another; a box that fails mid-trial returns an infra error (excluded, not 0)."""

    def __init__(self, pool: Pool, log_path: str | None):
        self.pool, self.log_path, self.lock = pool, log_path, threading.Lock()
        self.done = 0

    def run(self, body: bytes) -> dict:
        task = json.loads(body).get("instance_id", "?")
        queued, tried = time.time(), set()
        while True:
            box = self.pool.acquire(task, tried)
            start = time.time()
            try:
                reply = json.loads(post(f"http://{box}/run", body))
            except (ConnectionRefusedError, urllib.error.URLError) as e:
                refused = isinstance(getattr(e, "reason", e), ConnectionRefusedError)
                self.pool.release(box)
                if refused and len(tried) + 1 < len(self.pool.free):
                    tried.add(box)
                    continue                           # never started: try another box
                reply = error_reply(f"{box}: {e}")
            except Exception as e:                     # dropped mid-trial, bad JSON, ...
                self.pool.release(box)
                reply = error_reply(f"{box}: {e!r}")
            else:
                self.pool.release(box, task, (time.time() - start) / 60)
            self.record(task, box, queued, start, reply)
            return reply

    def record(self, task: str, box: str, queued: float, start: float, reply: dict) -> None:
        row = {"task": task, "box": box, "queued": round(queued, 3), "start": round(start, 3),
               "end": round(time.time(), 3), "reward": reply.get("reward"), "exit_status": reply.get("exit_status")}
        with self.lock:
            self.done += 1
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    def broadcast(self, path: str, body: bytes, headers: dict) -> dict:
        """/flush and /flush_all go to every box (Miles' abort hook sends them once)."""
        out = {}
        for box in self.pool.free:
            req = urllib.request.Request(f"http://{box}{path}", data=body, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    out[box] = json.loads(r.read())
            except Exception as e:
                out[box] = {"error": repr(e)}
        return out

    def health(self) -> dict:
        boxes = {}
        for box in self.pool.free:
            try:
                boxes[box] = get(f"http://{box}/health").get("status") == "ok"
            except Exception:
                boxes[box] = False
        st = self.pool.status()
        return {"status": "ok" if any(boxes.values()) else "down", "boxes_up": boxes, "done": self.done, **st}


def make_handler(d: Dispatcher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                  # keep-alive: Miles holds one connection per trial

        def _send(self, obj: dict, code: int = 200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(d.health())
            elif self.path == "/status":
                self._send({**d.pool.status(), "done": d.done})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/run":
                self._send(d.run(body))
            elif self.path in ("/flush", "/flush_all"):
                auth = {"Authorization": self.headers["Authorization"]} if self.headers.get("Authorization") else {}
                self._send(d.broadcast(self.path, body, auth))
            else:
                self._send({"error": "not found"}, 404)

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backends", required=True, help="host:port of each box's agent server, comma-separated")
    p.add_argument("--slots", type=int, default=32, help="trials at once per box (= its --max-concurrent)")
    p.add_argument("--minutes", help="JSON {task: expected minutes}; longest first")
    p.add_argument("--log", help="JSONL, one row per finished trial")
    p.add_argument("--port", type=int, default=PORT)
    a = p.parse_args(argv)
    minutes = json.load(open(a.minutes)) if a.minutes else {}
    pool = Pool(a.backends.split(","), a.slots, minutes)
    server = Server(("127.0.0.1", a.port), make_handler(Dispatcher(pool, a.log)))
    print(f"dispatching to {a.backends} ({a.slots} each) on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
