"""Agent-like load on an SGLang server, replaying real trials' turn sizes and wait times (PLAN.md §5b).

N agent threads. Each picks a trial from the profile (make_profile.py) and plays it: per turn, one streamed
/generate call on the growing context (a shared system prefix + this trial's tokens) that must produce the
real turn's output length (ignore_eos); then the real observation length is appended and the agent sleeps
the real environment time. A trial stops at --max-ctx tokens (Miles' 48k), as Harbor would.

  python -m proto.agent_load --profile proto/load_profile.json --agents 16 --minutes 5 --out agents.jsonl --lora proto
  python -m proto.agent_load --hot-load /root/proto_out/RUN/adapter --times 5 --out hotload.jsonl
Output: JSONL, one row per call: agent, trial, turn, t_send, t_first, t_end, prompt, out, cached,
max_gap (longest pause between streamed chunks, s), sleep (env seconds after the call), error.
"""
import argparse
import http.client
import json
import random
import threading
import time
from urllib.parse import urlparse

PREFIX_TOKENS = 700                  # terminus-2's system prompt, shared by every trial (prefix-cache hits)
TOKEN_RANGE = (1000, 150000)         # ordinary vocabulary ids (special tokens start at 248044)


def _conn(url: str, timeout: float) -> http.client.HTTPConnection:
    u = urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def post_json(url: str, path: str, payload: dict, timeout: float = 600) -> tuple[int, str]:
    c = _conn(url, timeout)
    try:
        c.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read().decode(errors="replace")
    finally:
        c.close()


def generate(url: str, payload: dict, timeout: float) -> dict:
    """One streamed /generate call. Returns t_first, max_gap, output ids (if the server sends them), meta."""
    c = _conn(url, timeout)
    try:
        c.request("POST", "/generate", body=json.dumps(payload), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:200]!r}")
        t_first, last_t, max_gap, meta, cum, inc, mode = None, None, 0.0, {}, [], [], None
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk, now = json.loads(data), time.time()
            meta = chunk.get("meta_info") or meta
            ids = chunk.get("output_ids") or []
            if ids:                                       # servers send ids cumulatively or incrementally
                if mode is None and cum and len(ids) > len(cum):
                    mode = "cumulative" if ids[:len(cum)] == cum else "incremental"
                if mode != "cumulative":
                    inc += ids
                cum = ids
            if t_first is None and meta.get("completion_tokens", 0) > 0:
                t_first = now
            if last_t is not None:
                max_gap = max(max_gap, now - last_t)
            last_t = now
        n = meta.get("completion_tokens", 0)
        ids = cum if len(cum) == n else inc if len(inc) == n else None
        return {"t_first": t_first, "max_gap": max_gap, "ids": ids, "meta": meta}
    finally:
        c.close()


class Load:
    def __init__(self, args, profile: dict):
        self.a, self.trials = args, profile["trials"]
        self.end = time.time() + 60 * args.minutes
        self.prefix = [random.Random(0).randrange(*TOKEN_RANGE) for _ in range(PREFIX_TOKENS)]
        self.lock, self.out = threading.Lock(), open(args.out, "a", buffering=1)

    def write(self, row: dict) -> None:
        with self.lock:
            self.out.write(json.dumps(row) + "\n")

    def agent(self, i: int) -> None:
        a, rng = self.a, random.Random(1000 + i)
        rand = lambda n: [rng.randrange(*TOKEN_RANGE) for _ in range(n)]
        time.sleep(i * a.stagger)
        trial_no = 0
        while time.time() < self.end:
            tr, trial_no = rng.choice(self.trials), trial_no + 1
            ctx = self.prefix + rand(max(0, tr["prompt0"] - PREFIX_TOKENS))
            for turn, (out, obs, env) in enumerate(tr["turns"]):
                if time.time() >= self.end or len(ctx) + out > a.max_ctx:
                    break
                payload = {"input_ids": ctx, "stream": True,
                           "sampling_params": {"max_new_tokens": out, "ignore_eos": True, "temperature": 1.0}}
                if a.lora:
                    payload["lora_path"] = a.lora
                row = {"agent": i, "trial": f"{i}.{trial_no}", "turn": turn, "prompt": len(ctx), "t_send": time.time()}
                try:
                    g = generate(a.url, payload, a.timeout)
                except Exception as e:                    # noqa: BLE001 (server restart, timeout: log it)
                    self.write({**row, "t_end": time.time(), "error": str(e)[:200]})
                    time.sleep(1)
                    break
                meta = g["meta"]
                n = meta.get("completion_tokens", 0)
                sleep = env * a.env_scale
                self.write({**row, "t_first": g["t_first"], "t_end": time.time(), "out": n,
                            "cached": meta.get("cached_tokens"), "max_gap": round(g["max_gap"], 3),
                            "sleep": round(sleep, 3), "error": None})
                ctx = ctx + (g["ids"] if g["ids"] is not None else rand(n)) + rand(obs)
                time.sleep(max(0.0, min(sleep, self.end - time.time())))

    def run(self) -> None:
        threads = [threading.Thread(target=self.agent, args=(i,), daemon=True) for i in range(self.a.agents)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def hot_load(url: str, adapter: str, times: int, gap: float, out_path: str) -> None:
    """Load + unload a copy of the adapter `times` times, as an overlapped trainer would push each step."""
    with open(out_path, "a", buffering=1) as out:
        for k in range(times):
            name, t0 = f"swap{k}", time.time()
            st, body = post_json(url, "/load_lora_adapter", {"lora_name": name, "lora_path": adapter})
            t1 = time.time()
            st2, _ = post_json(url, "/unload_lora_adapter", {"lora_name": name})
            out.write(json.dumps({"type": "hotload", "t": t0, "load_s": round(t1 - t0, 3),
                                  "unload_s": round(time.time() - t1, 3), "ok": st == 200 and st2 == 200,
                                  "error": None if st == 200 else body[:200]}) + "\n")
            time.sleep(gap)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--out", required=True)
    p.add_argument("--profile")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--max-ctx", type=int, default=49152)
    p.add_argument("--lora", default="", help="adapter name loaded in the server (--lora-paths name=...)")
    p.add_argument("--stagger", type=float, default=2.0, help="seconds between agent starts")
    p.add_argument("--env-scale", type=float, default=1.0, help="multiplies the environment waits")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--hot-load", default="", help="adapter folder: time load/unload instead of running agents")
    p.add_argument("--times", type=int, default=5)
    p.add_argument("--gap", type=float, default=30.0)
    a = p.parse_args(argv)
    if a.hot_load:
        hot_load(a.url, a.hot_load, a.times, a.gap, a.out)
        return
    with open(a.profile) as f:
        Load(a, json.load(f)).run()


if __name__ == "__main__":
    main()
