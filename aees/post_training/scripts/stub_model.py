"""A stand-in for Miles' session server in the EC2 rehearsal (scripts/rehearse.sh): an OpenAI-compatible
/v1/chat/completions that runs one `ls` and then declares the task complete, so a real Harbor trial on a box
finishes in seconds with the real verifier. GET / answers "aees-stub" (the reverse-tunnel probe).

  python3 scripts/stub_model.py --port 30000          # standard library only
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LS = {"analysis": "Rehearsal stub.", "plan": "List files, then finish.",
      "commands": [{"keystrokes": "ls\n", "duration": 0.1}], "task_complete": False}
DONE = {"analysis": "Rehearsal stub.", "plan": "Finish.", "commands": [], "task_complete": True}


def reply(messages: list[dict]) -> dict:
    """First turn: run ls. Every later turn: task_complete (terminus-2 asks to confirm once)."""
    turns = sum(1 for m in messages if m.get("role") == "assistant")
    return LS if turns == 0 else DONE


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, kind: str = "application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(b"aees-stub", "text/plain")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = json.dumps(reply(req.get("messages", [])))
        self._send(json.dumps({
            "id": f"stub-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=30000)
    a = p.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
