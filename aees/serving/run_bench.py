#!/usr/bin/env python3
"""
run_bench.py - Experiment 1 (fast track): latency, throughput, QPS and stress tests for the
Qwen3-4B SGLang server, repeated across KV-cache and attention-backend ("arm") settings.

Run on the GPU host, in the same directory as serve.sh and make_dashboard.py:
    source ~/qwen-serve/.venv/bin/activate
    python run_bench.py --dry-run          # print the plan and every command, run nothing
    python run_bench.py --quick            # ~15 min end-to-end check of the whole pipeline
    python run_bench.py                    # full matrix, ~1 hour
    python run_bench.py --arms baseline fp8_kv
    python run_bench.py --resume results/20261005-141200   # continue an interrupted run

Each arm restarts the server through serve.sh with extra flags (last flag wins), so the
server must be launched only through serve.sh. results/<run>/dashboard.html is rebuilt after
every arm, so it can be opened while the rest of the matrix is still running.
"""
import argparse
import datetime
import json
import os
import pathlib
import shlex
import shutil
import re
import subprocess
import sys
import threading
import time
import urllib.request

ARMS = {  # name: (extra sglang flags appended to serve.sh's baseline command, description)
    "baseline": ("", "FlashInfer, radix cache on, BF16 KV, page size 1"),
    "no_radix": ("--disable-radix-cache", "Prefix (radix) cache off"),
    "fp8_kv": ("--kv-cache-dtype fp8_e5m2", "FP8 e5m2 KV cache"),
    "page16": ("--page-size 16", "KV page size 16"),
    "triton": ("--attention-backend triton --sampling-backend pytorch", "Triton attention, no FlashInfer"),
}
WORKLOADS = {
    # agent: 16 shared 4K-token system prompts (prompt templates + tool schemas), short question
    "agent": dict(kind="gsp", system=4096, question=128, output=256, groups=16),
    "chat": dict(kind="random", input=1024, output=256),
    "long": dict(kind="random", input=8192, output=256),
}
FULL = {  # concurrency sweeps per arm; "_other" applies to arms not listed
    "baseline": {"agent": [1, 4, 16, 32, 64, 128, 256], "chat": [1, 4, 16, 32, 64, 128, 256],
                 "long": [1, 4, 8, 16, 32, 48, 64]},
    "fp8_kv": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128], "long": [8, 16, 32, 48, 64]},
    "_other": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128]},
}
QUICK = {
    "baseline": {"agent": [1, 16, 64], "chat": [16, 64], "long": [8]},
    "_other": {"agent": [16, 64]},
}
RATES_FULL, RATES_QUICK = [1, 2, 4, 8, 16, 32], [2, 8]  # offered req/s for the QPS sweep
COLD_FULL, COLD_QUICK = [16, 64], [16]  # agent points measured from an empty prefix cache (baseline only)
SERVED_NAME = os.environ.get("SERVED_NAME", "qwen3-4b")
METRIC_RE = re.compile(r"^(sglang:[A-Za-z0-9_:]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)\s*$")

# Tool-call quality check: same prompts against every server setting, temperature 0.
Q_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "ISO code"},
                    "to_currency": {"type": "string", "description": "ISO code"}},
                    "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "Add an event to the user's calendar",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM, 24-hour"}},
                    "required": ["title", "date", "time"]}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Search the company documentation",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def quality_cases():
    cases = []
    for i, c in enumerate(["Paris", "Tokyo", "Austin", "Nairobi", "Lima", "Oslo", "Seoul", "Cairo"]):
        u = ["celsius", "fahrenheit"][i % 2]
        cases.append((f"What's the weather in {c} right now, in {u}?", "get_weather", {"city": c, "unit": u}))
    for amt, a, b in [(100, "USD", "EUR"), (250, "GBP", "JPY"), (75.5, "EUR", "USD"), (1200, "JPY", "USD"),
                      (60, "CAD", "EUR"), (9, "AUD", "GBP"), (500, "CHF", "USD"), (42, "USD", "INR")]:
        cases.append((f"Convert {amt} {a} to {b}.", "convert_currency",
                      {"amount": amt, "from_currency": a, "to_currency": b}))
    for t, d, tm in [("Team sync", "2026-10-12", "09:30"), ("Dentist", "2026-11-03", "14:00"),
                     ("Quarterly review", "2026-12-01", "10:00"), ("Lunch with Ana", "2026-10-20", "12:15"),
                     ("Flight to Denver", "2026-11-18", "07:45"), ("Gym", "2026-10-15", "18:00"),
                     ("Board meeting", "2027-01-09", "16:30"), ("Call with vendor", "2026-10-29", "11:00")]:
        cases.append((f"Put '{t}' on my calendar for {d} at {tm}.", "create_event", {"title": t, "date": d, "time": tm}))
    for q in ["rate limits", "refund policy", "SSO setup", "data retention", "API keys", "webhooks",
              "billing cycle", "export to CSV"]:
        cases.append((f"Search our docs for '{q}'.", "search_docs", {"query": q}))
    for q in ["Say hi in five words.", "What is 2+2? Answer with just the number.", "Name three primary colors.",
              "Translate 'good morning' to Spanish.", "Is a tomato a fruit? One sentence."]:
        cases.append((q, None, None))
    return cases


def args_match(expected, got):
    for k, v in expected.items():
        g = got.get(k)
        if isinstance(v, (int, float)):
            try:
                if abs(float(g) - float(v)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(g).strip().lower() != str(v).strip().lower():
            return False
    return True


class MetricsSampler(threading.Thread):
    """Samples SGLang's Prometheus gauges once a second into a CSV (time, metric, value)."""

    def __init__(self, url, path):
        super().__init__(daemon=True)
        self.url, self.path, self.stop_evt = url, path, threading.Event()

    def run(self):
        with open(self.path, "a") as f:
            while not self.stop_evt.is_set():
                t = time.time()
                try:
                    with urllib.request.urlopen(self.url, timeout=5) as r:
                        for line in r.read().decode().splitlines():
                            m = METRIC_RE.match(line)
                            if m and not m.group(1).endswith(("_bucket", "_sum", "_count", "_created", "_total")):
                                f.write(f"{t:.2f},{m.group(1)},{m.group(2)}\n")
                    f.flush()
                except Exception:
                    pass
                self.stop_evt.wait(1.0)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=5)
REQUIRED_FLAGS = ["--backend", "--dataset-name", "--num-prompts", "--max-concurrency",
                  "--request-rate", "--output-file", "--output-details", "--seed",
                  "--random-input-len", "--random-output-len", "--random-range-ratio",
                  "--gsp-num-groups", "--gsp-prompts-per-group", "--gsp-system-prompt-len",
                  "--gsp-question-len", "--gsp-output-len"]


def n_prompts(wl, conc, quick):
    if wl == "long":
        n = max(20, 4 * conc)
    elif conc == 1:
        n = 40
    elif conc <= 4:
        n = 80
    else:
        n = max(200, 6 * conc)
    return max(16, n // 3) if quick else n


def last_json(path):
    try:
        lines = [ln for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1])
    except Exception:
        return None


class Runner:
    def __init__(self, a):
        self.a = a
        self.here = pathlib.Path(__file__).resolve().parent
        self.serve = self.here / "serve.sh"
        self.dir = a.resume or a.out / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.manifest = self.dir / "manifest.jsonl"
        self.random_ds = "random-ids"
        self.done = set()
        if self.manifest.exists():
            for line in self.manifest.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("rc") == 0:
                        self.done.add((r["arm"], r["workload"], r["mode"], str(r["value"])))

    # ---------- environment ----------
    def check_bench(self):
        p = subprocess.run([sys.executable, "-m", "sglang.bench_serving", "--help"],
                           capture_output=True, text=True)
        text = p.stdout + p.stderr
        if p.returncode != 0:
            sys.exit("Cannot run sglang.bench_serving; activate the venv first.\n" + text[-2000:])
        missing = [f for f in REQUIRED_FLAGS if f not in text]
        if missing:
            sys.exit(f"Installed bench_serving lacks flags {missing}. "
                     "Share `python -m sglang.bench_serving --help` to adapt the script.")
        if "random-ids" not in text:
            self.random_ds = "random"
            print("NOTE: no 'random-ids' dataset; using 'random' (downloads ShareGPT once).")

    def http(self, path, method="GET"):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.a.port}{path}", method=method)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        except Exception:
            return None

    def write_meta(self):
        path = self.dir / "meta.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        meta.setdefault("runs", []).append(dict(
            started=datetime.datetime.now().isoformat(timespec="seconds"),
            arms=self.a.arms, quick=self.a.quick, host=os.uname().nodename))
        meta["arm_flags"] = {k: v[0] or "(baseline)" for k, v in ARMS.items()}
        meta["arm_labels"] = {k: v[1] for k, v in ARMS.items()}
        meta["workloads"] = WORKLOADS
        if not self.a.dry_run:
            v = subprocess.run([sys.executable, "-c",
                                "import torch, flashinfer, sglang; print(torch.__version__, "
                                "torch.version.cuda, flashinfer.__version__, sglang.__version__)"],
                               capture_output=True, text=True).stdout.split()
            if len(v) == 4:
                meta["versions"] = dict(torch=v[0], cuda=v[1], flashinfer=v[2], sglang=v[3])
            g = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout
            meta["gpu"] = g.strip()
        path.write_text(json.dumps(meta, indent=2))

    # ---------- server ----------
    def restart(self, arm, note=""):
        extra = ARMS[arm][0]
        print(f"\n=== {note or 'arm ' + arm}: {ARMS[arm][1]} ===\n  serve.sh restart  EXTRA_ARGS='{extra}'")
        if self.a.dry_run:
            return True
        p = subprocess.run(["bash", str(self.serve), "restart"], env=dict(os.environ, EXTRA_ARGS=extra),
                           capture_output=True, text=True)
        (self.dir / arm).mkdir(parents=True, exist_ok=True)
        (self.dir / arm / "server_start.txt").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            print(f"  server failed to start; skipping arm. See {self.dir / arm / 'server_start.txt'}")
            return False
        return True

    def collect_server_files(self, arm):
        latest = self.a.work_dir / "logs" / "latest.log"
        try:
            log = latest.resolve()
            shutil.copy(log, self.dir / arm / "server.log")
            cmd = log.with_suffix(".cmd")
            if cmd.exists():
                shutil.copy(cmd, self.dir / arm / "server.cmd")
        except Exception as e:
            print(f"  (could not copy server log: {e})")

    # ---------- benchmark points ----------
    def bench_cmd(self, wl, out, n, conc, rate):
        w = WORKLOADS[wl]
        cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang",
               "--host", "127.0.0.1", "--port", str(self.a.port), "--model", self.a.model_dir,
               "--output-file", str(out), "--output-details", "--seed", "1"]
        if w["kind"] == "gsp":
            per = -(-n // w["groups"])
            n = per * w["groups"]
            cmd += ["--dataset-name", "generated-shared-prefix",
                    "--gsp-num-groups", str(w["groups"]), "--gsp-prompts-per-group", str(per),
                    "--gsp-system-prompt-len", str(w["system"]),
                    "--gsp-question-len", str(w["question"]), "--gsp-output-len", str(w["output"])]
        else:
            cmd += ["--dataset-name", self.random_ds, "--random-input-len", str(w["input"]),
                    "--random-output-len", str(w["output"]), "--random-range-ratio", "1.0"]
        cmd += ["--num-prompts", str(n), "--request-rate", str(rate) if rate else "inf"]
        if conc:
            cmd += ["--max-concurrency", str(conc)]
        return cmd, n

    def run_point(self, arm, wl, mode, value, n, conc=None, rate=None, record=True, warm=None):
        """warm=True: flush, then send the identical prompts once (unrecorded) so the measured pass sees
        the steady state of a warm prefix cache. Defaults to True for agent points except mode 'cold'."""
        if warm is None:
            warm = record and wl == "agent" and mode != "cold"
        key = (arm, wl, mode, str(value))
        if record and key in self.done:
            print(f"  skip (already done): {arm} {wl} {mode}={value}")
            return
        arm_dir = self.dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        name = f"{wl}_{mode}{value}" if record else "_warmup"
        out = arm_dir / f"{name}.jsonl"
        cmd, n = self.bench_cmd(wl, out, n, conc, rate)
        if self.a.dry_run:
            print("   ", shlex.join(cmd))
            return
        out.unlink(missing_ok=True)
        self.http("/flush_cache", "POST")  # every point starts from an empty prefix cache...
        if warm:  # ...then agent points warm it with the same prompts (dataset depends only on seed/size)
            wcmd, _ = self.bench_cmd(wl, arm_dir / "_warm.jsonl", n, 64, None)
            with open(arm_dir / "_warm.log", "w") as f:
                subprocess.run(wcmd, stdout=f, stderr=subprocess.STDOUT, timeout=self.a.point_timeout)
        t0 = time.time()
        with open(arm_dir / f"{name}.log", "w") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=self.a.point_timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        dt = time.time() - t0
        res = last_json(out)
        if rc == 0 and res is None:
            rc = -1
        if record:
            rec = dict(arm=arm, workload=wl, mode=mode, value=value, num_prompts=n, warm=bool(warm),
                       max_concurrency=conc, request_rate=rate, rc=rc,
                       file=str(out.relative_to(self.dir)), start=round(t0, 1), seconds=round(dt, 1))
            with open(self.manifest, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if res:
            nan = float("nan")
            print(f"  {arm:9s} {wl:6s} {mode + '=' + str(value):10s} n={n:<5d}"
                  f"{res.get('output_throughput', nan):8.0f} out tok/s"
                  f"  TTFT p99 {res.get('p99_ttft_ms', nan):8.1f} ms"
                  f"  ITL p99 {res.get('p99_itl_ms', nan):6.1f} ms  [{dt:.0f}s]")
        else:
            print(f"  {arm} {wl} {mode}={value}: FAILED rc={rc}; see {arm_dir / (name + '.log')}")

    def quality_check(self, arm):
        path = self.dir / arm / "quality.json"
        if self.a.dry_run:
            print(f"    quality check: {len(quality_cases())} tool-call prompts at temperature 0")
            return
        if path.exists():
            print("  skip (already done): quality check")
            return
        rows = []
        for prompt, tool, expected in quality_cases():
            body = json.dumps({"model": SERVED_NAME, "messages": [{"role": "user", "content": prompt}],
                               "tools": Q_TOOLS, "temperature": 0, "max_tokens": 256}).encode()
            row = dict(prompt=prompt, expected_tool=tool, expected_args=expected)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{self.a.port}/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    msg = json.loads(r.read())["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                row["content"] = (msg.get("content") or "")[:300]
                row["tool"] = calls[0]["function"]["name"] if calls else None
                raw = calls[0]["function"]["arguments"] if calls else None
                row["raw_args"] = raw
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    row["valid_json"] = calls != [] and isinstance(args, dict)
                except Exception:
                    args, row["valid_json"] = {}, False
                row["args"] = args
                row["tool_ok"] = row["tool"] == tool
                row["args_ok"] = (tool is None and not calls) or (row["tool_ok"] and args_match(expected or {}, args))
            except Exception as e:
                row.update(error=str(e)[:200], tool_ok=False, args_ok=False, valid_json=False)
            rows.append(row)
        n = len(rows)
        summ = dict(n=n, tool_ok=sum(r["tool_ok"] for r in rows) / n, args_ok=sum(r["args_ok"] for r in rows) / n)
        path.write_text(json.dumps(dict(summary=summ, cases=rows), indent=1))
        print(f"  quality: correct tool {summ['tool_ok']:.0%}, correct tool + arguments {summ['args_ok']:.0%} ({n} prompts)")

    def run_arm(self, arm):
        plan = QUICK if self.a.quick else FULL
        sweeps = plan.get(arm, plan["_other"])
        if not self.restart(arm):
            return
        gpu = sampler = None
        if not self.a.dry_run:
            sampler = MetricsSampler(f"http://127.0.0.1:{self.a.port}/metrics", self.dir / arm / "metrics_ts.csv")
            sampler.start()
            gpu = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits", "-l", "1"],
                stdout=open(self.dir / arm / "gpu.csv", "a"), stderr=subprocess.DEVNULL)
        try:
            self.run_point(arm, "agent", "warmup", 0, 64, conc=16, record=False)
            for wl, levels in sweeps.items():
                for c in levels:
                    self.run_point(arm, wl, "conc", c, n_prompts(wl, c, self.a.quick), conc=c)
                if not self.a.dry_run:
                    m = self.http("/metrics")
                    if m:
                        (self.dir / arm / f"metrics_{wl}.txt").write_text(m)
            self.quality_check(arm)
            if arm == "baseline":
                q = self.a.quick
                for c in (COLD_QUICK if q else COLD_FULL):  # cold start: empty cache, everyone arrives at once
                    self.run_point(arm, "agent", "cold", c, n_prompts("agent", c, q), conc=c, warm=False)
                for r in (RATES_QUICK if q else RATES_FULL):
                    n = max(30, r * 10) if q else max(60, r * 30)
                    self.run_point(arm, "agent", "rate", r, n, rate=r)
                n = 400 if q else 3000
                self.run_point(arm, "agent", "p999", 64, n, conc=64)  # long run for p99.9 tails
                n = 300 if q else 1500
                self.run_point(arm, "agent", "burst", n, n)  # every request sent at once
        finally:
            if gpu:
                gpu.terminate()
            if sampler:
                sampler.stop()
            if not self.a.dry_run:
                self.collect_server_files(arm)
        self.dashboard()

    def dashboard(self):
        if self.a.dry_run:
            return
        p = subprocess.run([sys.executable, str(self.here / "make_dashboard.py"), str(self.dir)])
        if p.returncode == 0:
            print(f"  dashboard updated: {self.dir / 'dashboard.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--quick", action="store_true", help="small version of every test (~15 min)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands only")
    ap.add_argument("--resume", type=pathlib.Path, help="results dir of an interrupted run")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(os.environ.get(
        "RESULTS_DIR", pathlib.Path(os.environ.get("WORK_DIR", "~/qwen-serve")).expanduser() / "results")),
        help="results folder, the one ./serve.sh dashboard serves (default ~/qwen-serve/results)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    ap.add_argument("--work-dir", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("WORK_DIR", "~/qwen-serve")).expanduser())
    ap.add_argument("--model-dir", default=os.environ.get(
        "MODEL_DIR", str(pathlib.Path("~/models/Qwen3-4B-Instruct-2507").expanduser())))
    ap.add_argument("--point-timeout", type=int, default=1200, help="seconds per test point")
    ap.add_argument("--no-restore", action="store_true", help="leave the last arm's server running")
    a = ap.parse_args()

    r = Runner(a)
    if not a.dry_run:
        if not r.serve.exists():
            sys.exit(f"serve.sh not found next to {__file__}")
        r.check_bench()
    r.dir.mkdir(parents=True, exist_ok=True)
    r.write_meta()
    print(f"Results: {r.dir}")
    arms = [x for x in ARMS if x in a.arms]  # canonical order, baseline first
    for arm in arms:
        r.run_arm(arm)
    if not a.dry_run and not a.no_restore and arms and arms[-1] != "baseline":
        r.restart("baseline", note="restoring baseline server")
    print(f"\nDone. Dashboard: {r.dir / 'dashboard.html'}")
    print(f"View it: ./serve.sh dashboard, then http://127.0.0.1:{os.environ.get('DASH_PORT', '8000')}/{r.dir.name}/dashboard.html")


if __name__ == "__main__":
    main()
