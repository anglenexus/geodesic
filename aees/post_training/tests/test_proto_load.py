"""Tests for proto/make_profile.py, proto/agent_load.py (against a fake SGLang) and proto/llm_log_proxy.py."""
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from proto import agent_load as al, llm_log_proxy as lp, make_profile as mp

HERE = Path(__file__).resolve().parents[1]


class Quiet(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass                                                   # clients closing kept-alive sockets


# --- make_profile -------------------------------------------------------------------------------------
def agent_step(prompt, out, message):
    return {"source": "agent", "message": message, "metrics": {"prompt_tokens": prompt, "completion_tokens": out}}


def test_profile_turns_from_trajectories(tmp_path):
    cmds = lambda *d: json.dumps({"analysis": "", "commands": [{"keystrokes": "ls\n", "duration": x} for x in d]})
    trials = {
        "mailman__abc": [{"source": "user", "message": "task"}, agent_step(800, 100, cmds(0.1, 2.0)),
                         agent_step(1000, 50, "not json"), agent_step(1300, 30, cmds(15))],
        "dna-insert__x": [agent_step(900, 0, cmds())],
        "empty__y": [{"source": "user", "message": "task"}],
    }
    for name, steps in trials.items():
        (tmp_path / "job" / name / "agent").mkdir(parents=True)
        (tmp_path / "job" / name / "agent" / "trajectory.json").write_text(json.dumps({"steps": steps}))
    (tmp_path / "job" / "broken__z" / "agent").mkdir(parents=True)
    (tmp_path / "job" / "broken__z" / "agent" / "trajectory.json").write_text("{")
    prof = mp.build([tmp_path / "job"])
    got = {t["task"]: (t["prompt0"], t["turns"]) for t in prof["trials"]}
    oh = mp.ENV_OVERHEAD_S
    assert got == {"mailman": (800, [[100, 100, 2.1 + oh], [50, 250, oh], [30, 0, 15 + oh]]),
                   "dna-insert": (900, [[1, 0, oh]])}
    assert "2 trials, 4 turns" in mp.summary(prof)


def test_committed_profile_is_usable():
    prof = json.loads((HERE / "proto" / "load_profile.json").read_text())
    assert len(prof["trials"]) > 500
    assert all(t["prompt0"] > 0 and t["turns"] and all(len(x) == 3 and x[0] >= 1 for x in t["turns"])
               for t in prof["trials"])


# --- agent_load against a fake SGLang ------------------------------------------------------------------
class FakeSGLang:
    """/generate streams SSE chunks of 3 tokens; output ids are 5, 6, 7, ... (cumulative, incremental or absent)."""

    def __init__(self, ids_mode="cumulative", fail=False):
        self.requests, self.adapters, self.lock = [], [], threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path in ("/load_lora_adapter", "/unload_lora_adapter"):
                    outer.adapters.append((self.path, body))
                    return self.reply(200, b"{}")
                if fail:
                    return self.reply(500, b"boom")
                with outer.lock:
                    outer.requests.append(body)
                n = body["sampling_params"]["max_new_tokens"]
                out = list(range(5, 5 + n))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for k in range(0, n, 3):
                    chunk = {"text": "x", "meta_info": {"completion_tokens": min(n, k + 3), "cached_tokens": 1}}
                    if ids_mode == "cumulative":
                        chunk["output_ids"] = out[:k + 3]
                    elif ids_mode == "incremental":
                        chunk["output_ids"] = out[k:k + 3]
                    self.chunk(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.chunk(b"data: [DONE]\n\n")
                self.chunk(b"")

            def chunk(self, data):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

            def reply(self, code, data):
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = Quiet(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


PROFILE = {"trials": [{"task": "a", "prompt0": 800, "turns": [[4, 11, 0.01], [7, 13, 0.01], [5, 0, 0.01]]},
                      {"task": "b", "prompt0": 900, "turns": [[8, 17, 0.01], [3, 19, 0.01]]},
                      {"task": "long", "prompt0": 2950, "turns": [[40, 10, 0.01], [40, 10, 0.01]]}]}


def run_load(tmp_path, server, *extra):
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps(PROFILE))
    out = tmp_path / "agents.jsonl"
    al.main(["--url", server.url, "--profile", str(prof), "--agents", "3", "--minutes", "0.02", "--stagger", "0",
             "--max-ctx", "3000", "--out", str(out), *extra])
    return [json.loads(line) for line in out.read_text().splitlines()]


@pytest.mark.parametrize("ids_mode", ["cumulative", "incremental", "absent"])
def test_agents_replay_turns_on_a_growing_context(tmp_path, ids_mode):
    srv = FakeSGLang(ids_mode)
    rows = run_load(tmp_path, srv, "--lora", "proto")
    reqs = srv.requests
    assert rows and len(rows) == len(reqs) and all(r["error"] is None for r in rows)
    turns = {(t["prompt0"], tuple(map(tuple, t["turns"]))) for t in PROFILE["trials"]}
    obs_by_out = {x[0]: x[1] for t in PROFILE["trials"] for x in t["turns"]}
    prefix = reqs[0]["input_ids"][:al.PREFIX_TOKENS]
    for r in reqs:
        assert r["input_ids"][:al.PREFIX_TOKENS] == prefix                       # shared system prompt
        assert r["sampling_params"]["ignore_eos"] and r["stream"] and r["lora_path"] == "proto"
        assert len(r["input_ids"]) + r["sampling_params"]["max_new_tokens"] <= 3000   # --max-ctx
    firsts = [r for r in reqs if len(r["input_ids"]) in (800, 900)]
    assert firsts and len({p for p, _ in turns}) == 3
    for r in reqs:                                       # every later turn extends an earlier call exactly
        if len(r["input_ids"]) in (800, 900, 2950):
            continue
        parents = [p for p in reqs if len(p["input_ids"]) < len(r["input_ids"])
                   and r["input_ids"][:len(p["input_ids"])] == p["input_ids"]
                   and len(r["input_ids"]) - len(p["input_ids"]) == (n := p["sampling_params"]["max_new_tokens"])
                   + obs_by_out[n]]
        assert parents, "turn does not extend a previous call by output + observation"
        p = parents[-1]
        out = r["input_ids"][len(p["input_ids"]):len(p["input_ids"]) + p["sampling_params"]["max_new_tokens"]]
        if ids_mode != "absent":
            assert out == list(range(5, 5 + len(out)))                          # the server's own tokens
    assert all(len(r["input_ids"]) != 2950 + 40 + 10 for r in reqs)            # long trial stops at the cap
    for row in rows:
        assert row["t_send"] <= row["t_first"] <= row["t_end"] and row["out"] >= 1 and row["max_gap"] >= 0


def test_agents_log_errors_and_keep_going(tmp_path):
    rows = run_load(tmp_path, FakeSGLang(fail=True))
    assert rows and all("HTTP 500" in r["error"] for r in rows)


def test_hot_load_times_adapter_pushes(tmp_path):
    srv = FakeSGLang()
    out = tmp_path / "hot.jsonl"
    al.main(["--url", srv.url, "--hot-load", "/ad", "--times", "2", "--gap", "0", "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["ok"] for r in rows] == [True, True] and all(r["load_s"] >= 0 for r in rows)
    assert srv.adapters == [("/load_lora_adapter", {"lora_name": "swap0", "lora_path": "/ad"}),
                            ("/unload_lora_adapter", {"lora_name": "swap0"}),
                            ("/load_lora_adapter", {"lora_name": "swap1", "lora_path": "/ad"}),
                            ("/unload_lora_adapter", {"lora_name": "swap1"})]


# --- llm_log_proxy ------------------------------------------------------------------------------------
class FakeChat(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.reply(200, b"ok")

    def do_POST(self):
        msgs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"]
        if msgs[-1]["content"] == "fail":
            return self.reply(500, b"upstream broke")
        self.reply(200, json.dumps({"choices": [{"message": {"content": f"reply to {msgs[-1]['content']}"}}],
                                    "usage": {"prompt_tokens": 10 * len(msgs), "completion_tokens": 5}}).encode())

    def reply(self, code, data):
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_log_proxy_chains_turns_into_trials(tmp_path):
    up = Quiet(("127.0.0.1", 0), FakeChat)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    out = tmp_path / "calls.jsonl"
    proxy = lp.serve(0, f"http://127.0.0.1:{up.server_address[1]}", str(out))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    def call(msgs):
        c = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
        c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": msgs}),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        body = r.read()
        return r.status, (json.loads(body)["choices"][0]["message"]["content"] if r.status == 200 else body)

    sys_msg = {"role": "system", "content": "you are terminus"}
    hist = {k: [sys_msg, {"role": "user", "content": task}] for k, task in
            [("t1", "task A"), ("t2", "task B"), ("t3", "task A")]}            # t3: same task as t1
    for _ in range(3):                                                       # 3 turns, interleaved trials
        for k in ("t1", "t2", "t3"):
            st, reply = call(hist[k])
            assert st == 200
            hist[k] += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"obs {k} {len(hist[k])}"}]
    assert call([sys_msg, {"role": "user", "content": "fail"}])[0] == 500
    c = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
    c.request("GET", "/health")
    assert c.getresponse().status == 200                                     # passed through, not logged
    time.sleep(0.1)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 10
    by_trial = {}
    for r in rows[:9]:
        by_trial.setdefault(r["trial"], []).append(r["turn"])
        assert r["status"] == 200 and r["completion_tokens"] == 5 and r["t_send"] <= r["t_end"]
    assert sorted(by_trial.values()) == [[0, 1, 2]] * 3                      # three trials, turns in order
    assert rows[9]["status"] == 500 and "upstream broke" in rows[9]["error"] and rows[9]["turn"] == 0
    proxy.shutdown()
    up.shutdown()
