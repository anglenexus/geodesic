"""Serving check before Experiment 1's rollouts (align_tmax_plan.md): runs on the 8xH100 host against the SGLang
engines, ~2 min. Standard library only.

  1. thinking: a reply carries reasoning_content; with preserve_thinking an earlier turn's thinking reaches the prompt
  2. latency: streamed requests at 1 per engine, then at the rollout load (10 per engine): time to first token (TTFT),
     inter-token latency (ITL, time between streamed chunks), output tokens/s per request and in total

  python3 scripts/phase1_probe.py --ports 30001,30002,30003,30004 --out $DATA/monitor/$RUN/probe.json
"""
import argparse
import json
import random
import statistics
import string
import threading
import time
import urllib.request

WORDS = ["terminal", "process", "kernel", "buffer", "socket", "verify", "compile", "archive", "schema", "packet"]


def prompt(seed: int, words: int) -> str:
    """A distinct ~words-long task text per seed, so requests don't share a cached prefix beyond the template."""
    rnd = random.Random(seed)
    body = " ".join(rnd.choice(WORDS) + rnd.choice(string.ascii_lowercase) for _ in range(words))
    return f"Task {seed}: summarize the following log in three bullet points.\n{body}"


def chat(port: int, messages: list, max_tokens: int, thinking: dict, stream: bool) -> urllib.request.Request:
    body = {"model": "model", "messages": messages, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0,
            "stream": stream, "chat_template_kwargs": thinking}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")


def thinking_checks(port: int) -> dict:
    on = {"enable_thinking": True, "preserve_thinking": True}
    with urllib.request.urlopen(chat(port, [{"role": "user", "content": "What is 17*23? Answer briefly."}], 2048, on, False),
                                timeout=300) as r:
        first = json.loads(r.read())
    msg = first["choices"][0]["message"]
    history = [{"role": "user", "content": "What is 17*23? Answer briefly."},
               {"role": "assistant", "content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content")},
               {"role": "user", "content": "And 19*23?"}]
    tokens = {}
    for keep in (True, False):
        kw = {"enable_thinking": True, "preserve_thinking": keep}
        with urllib.request.urlopen(chat(port, history, 1, kw, False), timeout=300) as r:
            tokens[keep] = json.loads(r.read())["usage"]["prompt_tokens"]
    return {"reasoning_content_chars": len(msg.get("reasoning_content") or ""), "content": (msg.get("content") or "")[:80],
            "prompt_tokens_preserve": tokens[True], "prompt_tokens_drop": tokens[False],
            "preserve_thinking_works": tokens[True] > tokens[False]}


def stream_one(port: int, seed: int, words: int, max_tokens: int, out: list, lock: threading.Lock) -> None:
    on = {"enable_thinking": True, "preserve_thinking": True}
    t0 = time.perf_counter()
    stamps, completion = [], None
    try:
        with urllib.request.urlopen(chat(port, [{"role": "user", "content": prompt(seed, words)}], max_tokens, on, True),
                                    timeout=600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    completion = d["usage"].get("completion_tokens")
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    stamps.append(time.perf_counter())
        row = {"port": port, "ttft": stamps[0] - t0 if stamps else None,
               "itl": [b - a for a, b in zip(stamps, stamps[1:])], "tokens": completion or len(stamps),
               "seconds": time.perf_counter() - t0}
    except Exception as e:
        row = {"port": port, "error": repr(e)}
    with lock:
        out.append(row)


def load(ports: list[int], per_engine: int, words: int, max_tokens: int, seed0: int) -> dict:
    rows, lock, threads = [], threading.Lock(), []
    t0 = time.perf_counter()
    for i, port in enumerate(ports):
        for j in range(per_engine):
            t = threading.Thread(target=stream_one, args=(port, seed0 + i * 100 + j, words, max_tokens, rows, lock))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in rows if "error" not in r and r["ttft"] is not None]
    ttft = sorted(r["ttft"] for r in ok)
    itl = sorted(x for r in ok for x in r["itl"])
    q = lambda xs, p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 4) if xs else None
    return {"per_engine": per_engine, "requests": len(rows), "errors": len(rows) - len(ok),
            "ttft_p50_s": q(ttft, .5), "ttft_p90_s": q(ttft, .9), "itl_p50_ms": q([x * 1e3 for x in itl], .5),
            "itl_p90_ms": q([x * 1e3 for x in itl], .9),
            "tok_s_per_request_p50": q(sorted(r["tokens"] / r["seconds"] for r in ok), .5),
            "tok_s_total": round(sum(r["tokens"] for r in ok) / wall, 1), "wall_s": round(wall, 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ports", default="30001,30002,30003,30004")
    p.add_argument("--levels", default="1,10", help="concurrent requests per engine, one round each")
    p.add_argument("--words", type=int, default=3000, help="prompt length (roughly tokens)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    ports = [int(x) for x in a.ports.split(",")]
    result = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thinking": {}, "load": []}
    for port in ports:
        result["thinking"][port] = thinking_checks(port)
        print(port, result["thinking"][port], flush=True)
    for k, n in enumerate(int(x) for x in a.levels.split(",")):
        result["load"].append(load(ports, n, a.words, a.max_tokens, 1000 * (k + 1)))
        print(json.dumps(result["load"][-1]), flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    return 0 if all(t["preserve_thinking_works"] and t["reasoning_content_chars"] for t in result["thinking"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
