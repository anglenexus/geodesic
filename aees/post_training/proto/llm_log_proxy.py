"""Logging pass-through between Harbor (on EC2) and the H200's SGLang, for the overlap prototype (PLAN.md §5b).

Harbor's terminus-2 calls http://127.0.0.1:<port>/v1/...; this forwards every request to --upstream (the SSH
tunnel to the H200) and logs each chat call, so real trials give per-turn model latency and environment
gaps even when a trial is cut off. Turns are chained into trials by message history: a call whose messages
minus the last two (assistant reply + new observation) equal an earlier call's messages continues that trial.

  python3 llm_log_proxy.py --port 8001 --upstream http://127.0.0.1:30001 --out calls.jsonl
Output: JSONL, one row per chat call: t_send, t_end, status, trial, turn, prompt_tokens, completion_tokens, error.
Standard library only (runs with the EC2 system python).
"""
import argparse
import hashlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def digest(messages) -> str:
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()


class Trials:
    """Assigns each chat call to a trial and a turn number from its message history."""

    def __init__(self):
        self.lock, self.seen, self.count = threading.Lock(), {}, 0     # digest(messages) -> [(trial, turn), ...]

    def assign(self, messages) -> tuple[int, int]:
        """Two trials of one task send the same first call, so each digest keeps a queue: the first
        continuation takes the oldest waiting trial."""
        with self.lock:
            waiting = self.seen.get(digest(messages[:-2])) if len(messages) > 2 else None
            if waiting:
                trial, turn = waiting.pop(0)
                turn += 1
            else:
                self.count += 1
                trial, turn = self.count, 0
            self.seen.setdefault(digest(messages), []).append((trial, turn))
            return trial, turn


def make_handler(upstream: str, out, trials: Trials, lock: threading.Lock):
    u = urlparse(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _forward(self, body: bytes | None):
            t_send, row = time.time(), None
            if self.command == "POST" and self.path.endswith("/chat/completions"):
                try:
                    trial, turn = trials.assign(json.loads(body).get("messages") or [])
                except ValueError:
                    trial, turn = 0, 0
                row = {"t_send": t_send, "trial": trial, "turn": turn}
            try:
                c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
                c.request(self.command, self.path, body=body, headers=headers)
                r = c.getresponse()
                data, status = r.read(), r.status
                ctype = r.getheader("Content-Type", "application/json")
                c.close()
                err = None if status == 200 else data[:200].decode(errors="replace")
            except OSError as e:
                data, status, ctype, err = json.dumps({"error": str(e)}).encode(), 502, "application/json", str(e)
            if row is not None:
                usage = {}
                if status == 200:
                    try:
                        usage = json.loads(data).get("usage") or {}
                    except ValueError:
                        pass
                row.update({"t_end": time.time(), "status": status, "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"), "error": err})
                with lock:
                    out.write(json.dumps(row) + "\n")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._forward(None)

        def do_POST(self):
            self._forward(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass                                     # clients closing kept-alive connections; calls are logged above


def serve(port: int, upstream: str, out_path: str) -> ThreadingHTTPServer:
    out = open(out_path, "a", buffering=1)
    return Server(("127.0.0.1", port), make_handler(upstream, out, Trials(), threading.Lock()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--upstream", default="http://127.0.0.1:30001")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    serve(a.port, a.upstream, a.out).serve_forever()
