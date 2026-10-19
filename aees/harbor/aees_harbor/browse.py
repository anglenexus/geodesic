"""Local trajectory browser: a small read-only web server over the ledger and Harbor trial dirs.

    python -m aees_harbor.browse --ledger state/ledger.sqlite --jobs-root ~/code/jobs
    # then open http://127.0.0.1:8765

Binds to 127.0.0.1 only. Trial files are located through the ledger (never from a URL path),
so the server can only read trials that were ingested. Ingest accepts only the name of a
direct sub-directory of --jobs-root.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aees_harbor.ingest import ingest_job
from aees_harbor.ledger import Ledger

PAGE = Path(__file__).parent / "browser" / "index.html"
SELECTED_TASKS = Path(__file__).resolve().parent.parent / "state" / "tb21_selected_tasks.txt"


def selected_tasks(path: Path = SELECTED_TASKS) -> list[str]:
    """The tasks chosen for collection (bare names, e.g. 'mailman'); [] if the file is missing."""
    try:
        return [t.strip() for t in path.read_text().split() if t.strip()]
    except FileNotFoundError:
        return []
TEST_LOG_TAIL = 20_000  # characters of verifier/test-stdout.txt to send


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def derive_actions(message: str) -> dict | None:
    """terminus-2 logs the model's raw JSON reply (raw_content=True) instead of tool_calls.

    Parse that reply into analysis / plan / actions so each step's commands can be shown as
    tool calls. Returns None if the message is not terminus-style JSON.
    """
    for text in (message, message[message.find("{"): message.rfind("}") + 1]):
        try:
            obj = json.loads(text)
            break
        except (json.JSONDecodeError, ValueError):
            obj = None
    if not isinstance(obj, dict) or "commands" not in obj:
        return None
    actions = [{"name": "bash_keystrokes", "arguments": {"keystrokes": c.get("keystrokes", ""),
                                                          "duration": c.get("duration")}}
               for c in obj.get("commands") or [] if isinstance(c, dict)]
    if obj.get("task_complete"):
        actions.append({"name": "mark_task_complete", "arguments": {}})
    return {"analysis": obj.get("analysis"), "plan": obj.get("plan"), "actions": actions}


def scenarios(ledger: Ledger) -> list[dict]:
    out = []
    for s in ledger.scenarios():
        rewards = [t["facts"].get("reward") for t in ledger.trial_facts(s["key"])]
        out.append({**s, "n_trials": len(rewards),
                    "raw_reward_1": sum(r == 1.0 for r in rewards),
                    "raw_reward_0": sum(r == 0.0 for r in rewards),
                    "raw_no_reward": sum(r is None for r in rewards)})
    return out


def trials(ledger: Ledger, key: str | None) -> list[dict]:
    keys = [key] if key else [s["key"] for s in ledger.scenarios()]
    out = []
    for k in keys:
        s = ledger.summary(k)
        status = {**{i: "held" for i in s["held"]}, **{i: "pass" for i in s["passed"]},
                  **{i: "fail" for i in s["failed"]}}
        fields = ledger.scenario_fields(k)
        task, model = fields["task"], fields.get("model")
        for t in ledger.trial_facts(k):
            out.append({"key": k, "task": task, "model": model, "run_index": t["run_index"], "attempt": t["attempt"],
                        "status": status.get(t["run_index"], "released"), **t["facts"]})
    return out


def sources(ledger_path: Path, remotes_path: Path | None) -> dict[str, dict]:
    """Ledgers the browser can show: this machine, plus each EC2 remote's mirrored snapshot.

    Remote trial dirs (/home/ubuntu/jobs/...) are rewritten to the local mirror (aees_harbor.remote).
    """
    out = {"local": {"label": "This Mac", "ledger": ledger_path, "rewrite": [], "remote": False}}
    if remotes_path and remotes_path.exists():
        from aees_harbor.remote import load_config, mirror_dirs
        cfg = load_config(remotes_path)
        for r in cfg["remotes"]:
            jobs_mirror, state = mirror_dirs(cfg, r)
            out[r["name"]] = {"label": f"EC2 {r['name']} ({r.get('instance_type', '')})",
                              "ledger": state / "ledger.sqlite", "remote": True,
                              "rewrite": [(r["remote_jobs"].rstrip("/"), str(jobs_mirror))]}
    return out


def _rewrite(path: str, rules) -> str:
    for old, new in rules:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def trial_detail(ledger: Ledger, key: str, run_index: int, attempt: int, rewrite=()) -> dict | None:
    match = [t for t in ledger.trial_facts(key, run_index) if t["attempt"] == attempt]
    if not match:
        return None
    d = Path(_rewrite(match[0]["trial_dir"], rewrite))
    segments = []  # trajectory.json plus any continuation files written after summarization
    for p in sorted((d / "agent").glob("trajectory*.json")):
        traj = _read_json(p)
        if traj is None:
            continue
        for step in traj.get("steps", []):
            if step.get("source") == "agent" and not step.get("tool_calls"):
                msg = step.get("message")
                step["derived"] = derive_actions(msg) if isinstance(msg, str) else None
        segments.append({"file": p.name, "trajectory": traj})
    ctrf = _read_json(d / "verifier" / "ctrf.json")
    log = d / "verifier" / "test-stdout.txt"
    return {
        "key": key, "run_index": run_index, "attempt": attempt, "trial_dir": str(d),
        "facts": match[0]["facts"], "scenario": ledger.scenario_fields(key),
        "segments": segments,
        "verifier": {
            "reward_txt": (d / "verifier" / "reward.txt").read_text().strip()
            if (d / "verifier" / "reward.txt").exists() else None,
            "ctrf": (ctrf or {}).get("results"),
            "report": _read_json(d / "verifier" / "report.json"),
            "test_log_tail": log.read_text(errors="replace")[-TEST_LOG_TAIL:] if log.exists() else None,
        },
        "exception": (_read_json(d / "result.json") or {}).get("exception_info"),
    }


def job_dir(jobs_root: Path, name: str) -> Path:
    """Resolve a job name to a direct child of jobs_root that looks like a Harbor job, or raise."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"invalid job name {name!r}")
    d = (jobs_root / name).resolve()
    if d.parent != jobs_root.resolve() or not (d / "lock.json").exists():
        raise ValueError(f"{name!r} is not a Harbor job directory in {jobs_root}")
    return d


def job_model(job: Path) -> str | None:
    """The model a Harbor job ran (its first agent's model_name in config.json), e.g. 'openai/gpt-5.6-luna'."""
    agents = (_read_json(job / "config.json") or {}).get("agents") or [{}]
    return agents[0].get("model_name")


def jobs(ledger: Ledger, jobs_root: Path) -> list[dict]:
    """Harbor job dirs under jobs_root with finished / already-ingested trial counts, newest first."""
    out = []
    for d in sorted(jobs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (d / "lock.json").exists():
            continue
        trial_dirs = [t for t in d.iterdir() if (t / "result.json").exists()]
        finished = [t for t in trial_dirs if (_read_json(t / "result.json") or {}).get("finished_at")]
        out.append({"name": d.name, "model": job_model(d), "trials": len(trial_dirs), "finished": len(finished),
                    "in_ledger": sum(1 for t in finished if ledger.find_trial(str(t)))})
    return out


class Handler(BaseHTTPRequestHandler):
    ledger_path: Path
    jobs_root: Path
    remotes_path: Path | None = None

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        srcs = sources(self.ledger_path, self.remotes_path)
        if url.path == "/api/selected":
            return self._json(selected_tasks())
        if url.path == "/api/sources":
            return self._json([{"name": k, "label": v["label"], "remote": v["remote"], "available": v["ledger"].exists()}
                               for k, v in srcs.items()])
        src = srcs.get(q.get("source", "local"))
        if src is None or not src["ledger"].exists():
            return self._json({"error": f"unknown or not-yet-synced source {q.get('source')!r}"}, 404)
        ledger = Ledger(src["ledger"])
        try:
            if url.path == "/api/scenarios":
                return self._json(scenarios(ledger))
            if url.path == "/api/trials":
                return self._json(trials(ledger, q.get("key")))
            if url.path == "/api/jobs":
                return self._json([] if src["remote"] else jobs(ledger, self.jobs_root))
            if url.path == "/api/trial":
                detail = trial_detail(ledger, q["key"], int(q["index"]), int(q.get("attempt", 1)), src["rewrite"])
                return self._json(detail) if detail else self._json({"error": "not found"}, 404)
            return self._json({"error": "unknown endpoint"}, 404)
        except (KeyError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        finally:
            ledger.close()

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if urlparse(self.path).path != "/api/ingest":
            return self._json({"error": "unknown endpoint"}, 404)
        ledger = Ledger(self.ledger_path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            d = job_dir(self.jobs_root, str(body.get("job", "")).strip())
            return self._json({"job": d.name, **ingest_job(ledger, d)})
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"error": str(e)}, 400)
        finally:
            ledger.close()

    def log_message(self, fmt, *args) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="Browse the trajectory ledger in a web browser")
    p.add_argument("--ledger", type=Path, default=Path("state/ledger.sqlite"))
    p.add_argument("--remotes", type=Path, default=Path("state/remotes.json"),
                   help="EC2 remotes config; their mirrored ledgers become extra sources")
    p.add_argument("--jobs-root", type=Path, default=Path("~/code/jobs"))
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    Handler.ledger_path = args.ledger.resolve()
    Handler.jobs_root = args.jobs_root.expanduser().resolve()
    Handler.remotes_path = args.remotes.resolve() if args.remotes.exists() else None
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Trajectory browser: http://127.0.0.1:{args.port}  (ledger: {Handler.ledger_path}, "
          f"jobs: {Handler.jobs_root})")
    server.serve_forever()


if __name__ == "__main__":
    main()
