#!/usr/bin/env python3
"""
run_bench.py - Phase 3: latency, throughput, QPS and stress tests for an SGLang server,
repeated across weight-precision, KV-cache, attention-backend and decoding ("arm") settings.

Run on the GPU host, in the same directory as the serve scripts and make_dashboard.py:
    source ~/qwen-serve/.venv/bin/activate
    python run_bench.py --model qwen3.5-27b --dry-run    # print the plan and every command, run nothing
    python run_bench.py --model qwen3.5-27b --quick      # ~10 min pipeline check (one setting)
    python run_bench.py --model qwen3.5-27b              # Phase 3, capped at 30 minutes
    python run_bench.py --model qwen3.5-27b --arms baseline fa3 triton     # other settings, own 30 min
    python run_bench.py                                  # Qwen3-4B, the original ~1.5 h matrix
    python run_bench.py --resume results/<run-id>        # continue an interrupted run

--model picks the serve script, settings and test sizes (MODELS below); the GPU is detected (--gpu
overrides). Qwen3.5-27B default settings:
    A100-40GB   AWQ 4-bit baseline, prefix cache off, MTP speculative decoding
    H100/H200   the three weight precisions: AWQ 4-bit (baseline), FP8, BF16
Every 27B run stops at --budget-min (default 30): points that don't fit are skipped and logged, and the
baseline's extra tests (cold start, QPS, p99.9, burst) only run if the remaining settings still fit.

Each arm restarts the server through its serve script with extra flags (last flag wins), so the server
must be launched only through that script. Results go to results/<model>_<gpu>_<time>/:
run_log.md (readable progress log), dashboard.html (rebuilt after every setting), and the raw data.
To check results into git: python export_results.py --phase3 results/<run-id>
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

# Every arm: extra sglang flags appended to the serve script's command and a description; optionally
# env vars for the serve script ("env"), GPU classes it runs on ("gpus"), minimum GPU memory in GB
# ("min_gb") and a checkpoint folder it needs ("needs").
ARMS = {
    "baseline": dict(flags="", label="Default settings: FlashInfer, prefix cache on, BF16 KV cache"),
    "no_radix": dict(flags="--disable-radix-cache", label="Prefix (radix) cache off"),
    "fp8_kv": dict(flags="--kv-cache-dtype fp8_e5m2", label="FP8 e5m2 KV cache", gpus=("ampere", "other")),
    "fp8_kv_e4m3": dict(flags="--kv-cache-dtype fp8_e4m3", label="FP8 e4m3 KV cache (native FP8 on Hopper)",
                        gpus=("hopper",)),
    "page16": dict(flags="--page-size 16", label="KV page size 16"),
    "triton": dict(flags="--attention-backend triton --sampling-backend pytorch", label="Triton attention, no FlashInfer"),
    "fa3": dict(flags="--attention-backend fa3", label="FlashAttention-3 attention (Hopper)", gpus=("hopper",)),
    # Synthetic prompts are random tokens, so the draft-acceptance rate here is not that of real text.
    "mtp": dict(flags="--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 "
                      "--speculative-num-draft-tokens 4",
                label="MTP speculative decoding (3 steps, 4 draft tokens; random-token prompts)"),
    "fp8_weights": dict(flags="", label="FP8 weights (official Qwen FP8: W8A8, 128x128 block scales)",
                        env={"WEIGHTS": "fp8"}, needs="~/models/Qwen3.5-27B-FP8", min_gb=70),
    "bf16_weights": dict(flags="", label="BF16 weights (official Qwen checkpoint, full precision)",
                         env={"WEIGHTS": "bf16"}, needs="~/models/Qwen3.5-27B", min_gb=70),
}
WORKLOADS = {
    # agent: 16 shared 4K-token system prompts (prompt templates + tool schemas), short question
    "agent": dict(kind="gsp", system=4096, question=128, output=256, groups=16),
    "chat": dict(kind="random", input=1024, output=256),
    "long": dict(kind="random", input=8192, output=256),
}
WL_TEXT = {"agent": "agent (4K shared system prompt + 128-token question)", "chat": "chat (1K in)",
           "long": "long context (8K in)"}


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


def sweep(spec, quick=False):
    """{workload: [concurrency, ...]} -> {workload: [(concurrency, requests), ...]} with the 4B sizing."""
    return {wl: [(c, n_prompts(wl, c, quick)) for c in cs] for wl, cs in spec.items()}


# Plans. sweeps: per arm ("_other" = arms not listed) {workload: [(concurrency, requests)]}.
# extras (baseline only): cold [(concurrency, requests)] from an empty prefix cache, rates [(req/s, requests)]
# with Poisson arrivals, p999 (concurrency, requests) for tails, burst = requests sent at once.
# budget_min caps the wall time; reserve_min is what one more setting needs (restart + sweep), used to
# decide whether the baseline extras still fit. warm_output=1 warms the prefix cache with 1-token answers.
_4B_FULL = {"baseline": {"agent": [1, 4, 16, 32, 64, 128, 256], "chat": [1, 4, 16, 32, 64, 128, 256],
                         "long": [1, 4, 8, 16, 32, 48, 64]},
            "fp8_kv": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128], "long": [8, 16, 32, 48, 64]},
            "_other": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128]}}
_4B_QUICK = {"baseline": {"agent": [1, 16, 64], "chat": [16, 64], "long": [8]}, "_other": {"agent": [16, 64]}}
PLAN_4B = {
    "full": dict(sweeps={k: sweep(v) for k, v in _4B_FULL.items()},
                 extras=dict(cold=[(c, n_prompts("agent", c, False)) for c in (16, 64)],
                             rates=[(r, max(60, r * 30)) for r in (1, 2, 4, 8, 16, 32)], p999=(64, 3000), burst=1500)),
    "quick": dict(sweeps={k: sweep(v, True) for k, v in _4B_QUICK.items()},
                  extras=dict(cold=[(16, n_prompts("agent", 16, True))], rates=[(r, max(30, r * 10)) for r in (2, 8)],
                              p999=(64, 400), burst=300)),
}
PLAN_4B["trace"] = dict(PLAN_4B["quick"], arms=["baseline"])

_A100_SWEEP = {"agent": [(1, 4), (16, 32), (32, 64)], "chat": [(32, 64)], "long": [(8, 16)]}
PLAN_27B_A100 = {
    "full": dict(arms=["baseline", "no_radix", "mtp"], sweeps={"_other": _A100_SWEEP},
                 extras=dict(cold=[(16, 32)], rates=[(1, 20), (4, 60)], p999=(32, 600), burst=150),
                 budget_min=30, reserve_min=8, warm_output=1),
    "quick": dict(arms=["baseline"], sweeps={"_other": {"agent": [(16, 32)], "chat": [(16, 32)]}},
                  budget_min=12, reserve_min=0, warm_output=1),
    "trace": dict(arms=["baseline"], sweeps={"_other": {"agent": [(16, 48), (32, 64)], "chat": [(32, 64)]}},
                  budget_min=8, reserve_min=0, warm_output=1, quality=False),
}
_HOPPER_SWEEP = {"agent": [(1, 8), (16, 64), (64, 192)], "chat": [(64, 128)], "long": [(16, 32)]}
PLAN_27B_HOPPER = {
    "full": dict(arms=["baseline", "fp8_weights", "bf16_weights"], sweeps={"_other": _HOPPER_SWEEP},
                 extras=dict(cold=[(16, 48)], rates=[(4, 60), (16, 160)], p999=(64, 1000), burst=300),
                 budget_min=30, reserve_min=7, warm_output=1),
    "quick": dict(arms=["baseline"], sweeps={"_other": {"agent": [(16, 64)], "chat": [(64, 128)]}},
                  budget_min=12, reserve_min=0, warm_output=1),
    "trace": dict(arms=["baseline"], sweeps={"_other": {"agent": [(16, 64), (64, 128)], "chat": [(64, 128)]}},
                  budget_min=8, reserve_min=0, warm_output=1, quality=False),
}

MODELS = {
    "qwen3-4b": dict(
        label="Qwen3-4B-Instruct-2507", serve="serve.sh", served="qwen3-4b",
        model_dir="~/models/Qwen3-4B-Instruct-2507",
        arms=["baseline", "no_radix", "fp8_kv", "page16", "triton"],
        plans={"_any": PLAN_4B}, serve_env={"_any": {}}, base_flags={"_any": ""}, point_timeout=1200,
        quality_extra={}, restore=True),
    "qwen3.5-27b": dict(
        label="Qwen3.5-27B", serve="serve_qwen3_27B.sh", served="qwen3.5-27b",
        model_dir="~/models/Qwen3.5-27B-AWQ-4bit",
        baseline_label="AWQ 4-bit weights (W4A16, Marlin), FlashInfer, prefix cache on, BF16 KV cache",
        arms=["baseline", "fp8_weights", "bf16_weights", "no_radix", "fp8_kv", "fp8_kv_e4m3", "triton", "fa3", "mtp"],
        plans={"ampere": PLAN_27B_A100, "_any": PLAN_27B_HOPPER},
        # Above Harbor's 16 so saturation shows; each running request holds ~150 MiB of DeltaNet state.
        # CUDA graphs only up to the largest tested batch keep restarts short. Env vars override.
        serve_env={"ampere": {"MAX_RUNNING": "32"}, "_any": {"MAX_RUNNING": "64"}},
        # {cg}: --cuda-graph-max-bs, or --cuda-graph-max-bs-decode in SGLang versions that split it.
        base_flags={"ampere": "{cg} 32", "_any": "{cg} 64"},
        point_timeout=900,
        # The quality check measures tool-call formatting across settings, not reasoning: thinking off
        # keeps it fast and deterministic (thinking stays on for Harbor runs).
        quality_extra={"chat_template_kwargs": {"enable_thinking": False}}, restore=False),
}

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




def last_json(path):
    try:
        lines = [ln for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1])
    except Exception:
        return None


def gpu_info():
    """(name, memory GB, class) of GPU 0; class is ampere / hopper / other."""
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()[0]
        name, mem, cap = [x.strip() for x in line.split(",")]
    except Exception:
        return None, None, "other"
    return name, int(float(mem)) // 1024, {"8": "ampere", "9": "hopper"}.get(cap.split(".")[0], "other")


def slug(text):
    return re.sub(r"[^a-z0-9.]+", "-", text.lower().replace("nvidia", "")).strip("-")


def pick(table, gpu_class):
    return table.get(gpu_class, table["_any"])


def server_flags():
    """Every --flag the installed SGLang server accepts (None if it can't be listed). Tries the `sglang serve`
    CLI the serve scripts use, then the older `python -m sglang.launch_server`."""
    clis = [str(pathlib.Path(sys.executable).with_name("sglang")), shutil.which("sglang")]
    cmds = [[c, "serve", "--help"] for c in clis if c and pathlib.Path(c).exists()]
    for cmd in cmds + [[sys.executable, "-m", "sglang.launch_server", "--help"]]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception:
            continue
        flags = set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", p.stdout + p.stderr))
        if "--attention-backend" in flags:  # a real server help text, not an error message
            return flags
    return None


def fmt(x, nd=0):
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "–"


class Runner:
    def __init__(self, a):
        self.a = a
        self.here = pathlib.Path(__file__).resolve().parent
        self.prof = MODELS[a.model]
        self.serve = self.here / self.prof["serve"]
        self.plan = a.plan
        self.serve_env = {k: v for k, v in pick(self.prof["serve_env"], a.gpu).items() if k not in os.environ}
        self.base_flags = pick(self.prof["base_flags"], a.gpu)
        run_id = f"{a.model}_{slug(a.gpu_name or a.gpu)}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
        self.dir = a.resume or a.out / run_id
        self.manifest = self.dir / "manifest.jsonl"
        self.log_md = self.dir / "run_log.md"
        self.random_ds = "random-ids"
        self.t0 = time.time()
        self.budget = (a.budget_min if a.budget_min is not None else self.plan.get("budget_min")) or 0
        self.done = set()
        if self.manifest.exists():
            for line in self.manifest.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("rc") == 0:
                        self.done.add((r["arm"], r["workload"], r["mode"], str(r["value"])))

    # ---------- time budget and readable log ----------
    def elapsed_min(self):
        return (time.time() - self.t0) / 60

    def left_min(self):
        return self.budget - self.elapsed_min() if self.budget else float("inf")

    def md(self, text):
        print(text if not text.startswith("|") else "  " + text)
        if not self.a.dry_run:
            with open(self.log_md, "a") as f:
                f.write(text + "\n")

    def label(self, arm):
        return self.prof.get("baseline_label", ARMS[arm]["label"]) if arm == "baseline" else ARMS[arm]["label"]

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
            arms=self.a.arms, quick=self.a.quick, mode=self.a.mode, budget_min=self.budget,
            host=os.uname().nodename))
        meta.update(model=self.a.model, model_label=self.prof["label"], served_name=self.prof["served"],
                    serve_script=self.prof["serve"], serve_env=self.serve_env, base_flags=self.base_flags,
                    gpu_class=self.a.gpu)
        meta.setdefault("arm_flags", {}).update(
            {k: (ARMS[k]["flags"] + " " + " ".join(f"{e}={v}" for e, v in ARMS[k].get("env", {}).items())).strip()
             or "(baseline)" for k in self.a.arms})
        meta.setdefault("arm_labels", {}).update({k: self.label(k) for k in self.a.arms})
        meta["workloads"] = WORKLOADS
        meta["plan"] = {k: v for k, v in self.plan.items()}
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
        return meta

    def log_header(self, meta):
        v = meta.get("versions", {})
        new = not self.log_md.exists()
        self.md(("" if new else "\n---\n") + f"# Phase 3 run log: {self.prof['label']} on {self.a.gpu_name or self.a.gpu}\n")
        self.md(f"- Started {datetime.datetime.now():%Y-%m-%d %H:%M}, plan `{self.a.mode}`, time budget "
                f"{f'{self.budget:g} min' if self.budget else 'none'}")
        if v:
            self.md(f"- SGLang {v.get('sglang')}, FlashInfer {v.get('flashinfer')}, torch {v.get('torch')} (CUDA {v.get('cuda')})")
        self.md(f"- Server: `{self.prof['serve']}` with {' '.join(f'{k}={x}' for k, x in self.serve_env.items()) or 'defaults'}"
                f"{f', flags `{self.base_flags}`' if self.base_flags else ''}")
        self.md("- Workloads: " + "; ".join(WL_TEXT[w] for w in WORKLOADS) + "; 256 output tokens each")
        self.md("- Settings: " + "; ".join(f"**{k}**: {self.label(k)}" for k in self.a.arms))
        self.md("- Columns: concurrency = requests in flight; tok/s = output tokens per second, all requests; "
                "TTFT = time to first token; ITL = time between output tokens; p50 / p99 = median / 99th percentile\n")

    # ---------- server ----------
    def restart(self, arm, note=""):
        spec = ARMS[arm]
        extra = (self.base_flags + " " + spec["flags"]).strip()
        env = dict(self.serve_env, **{k: os.path.expanduser(v) for k, v in spec.get("env", {}).items()})
        shown = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"\n=== {note or 'setting ' + arm}: {self.label(arm)} ===\n  {shown} EXTRA_ARGS='{extra}' {self.serve.name} restart")
        need = spec.get("needs")
        if need and not pathlib.Path(need).expanduser().joinpath("config.json").exists():
            self.md(f"\n## {arm}: skipped\n\nNo checkpoint at `{need}`; run setup_host_qwen3_27B.sh to download it.")
            return False
        if self.a.dry_run:
            return True
        t = time.time()
        p = subprocess.run(["bash", str(self.serve), "restart"], env=dict(os.environ, EXTRA_ARGS=extra, **env),
                           capture_output=True, text=True)
        (self.dir / arm).mkdir(parents=True, exist_ok=True)
        (self.dir / arm / "server_start.txt").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            self.md(f"\n## {arm}: server failed to start\n\nSee `{arm}/server_start.txt`. Last lines:\n\n```\n"
                    + "\n".join((p.stdout + p.stderr).strip().splitlines()[-12:]) + "\n```")
            return False
        facts = self.startup_facts()
        self.md(f"\n## {arm}: {self.label(arm)}\n\nServer ready in {time.time() - t:.0f} s (restart included)"
                + "".join(f"; {k} {v}" for k, v in facts.items()) + f". Elapsed {self.elapsed_min():.1f} min.\n")
        self.md("| workload | test | concurrency | requests | tok/s | req/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms "
                "| ITL p99 ms | seconds |")
        self.md("|---|---|---|---|---|---|---|---|---|---|---|")
        return True

    def startup_facts(self):
        facts = {}
        try:
            text = (self.a.work_dir / "logs" / "latest.log").resolve().read_text(errors="ignore")
        except Exception:
            return facts
        m = re.search(r"max_total_num_tokens=(\d+)", text)
        if m:
            facts["KV pool"] = f"{int(m.group(1)):,} tokens"
        m = re.search(r"Load weight end.*?mem usage=([\d.]+)\s*GB", text)
        if m:
            facts["weights"] = f"{float(m.group(1)):.1f} GB"
        m = re.search(r"Load weight end.*?avail mem=([\d.]+)\s*GB", text)
        if m:
            facts["free after load"] = f"{float(m.group(1)):.1f} GB"
        return facts

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
    def bench_cmd(self, wl, out, n, conc, rate, output=None):
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
                    "--gsp-question-len", str(w["question"]), "--gsp-output-len", str(output or w["output"])]
        else:
            cmd += ["--dataset-name", self.random_ds, "--random-input-len", str(w["input"]),
                    "--random-output-len", str(output or w["output"]), "--random-range-ratio", "1.0"]
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
        test = {"conc": "sweep", "cold": "cold start", "rate": f"{value} req/s", "p999": "p99.9 run",
                "burst": "burst"}.get(mode, mode)
        conc_txt = str(conc) if conc else ("all" if mode == "burst" else "open")
        if record and self.left_min() <= 0.5:
            self.md(f"| {wl} | {test} | {conc_txt} | {n} | skipped: time budget used | | | | | | |")
            return
        arm_dir = self.dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        name = f"{wl}_{mode}{value}" if record else "_warmup"
        out = arm_dir / f"{name}.jsonl"
        cmd, n = self.bench_cmd(wl, out, n, conc, rate)
        if self.a.dry_run:
            print("   ", shlex.join(cmd))
            return
        timeout = self.a.point_timeout
        if self.budget:
            timeout = max(60, min(timeout, int(self.left_min() * 60) + 60))
        out.unlink(missing_ok=True)
        self.http("/flush_cache", "POST")  # every point starts from an empty prefix cache...
        if warm:  # ...then agent points warm it with the same prompts (dataset depends only on seed/size)
            wcmd, _ = self.bench_cmd(wl, arm_dir / "_warm.jsonl", n, 64, None, output=self.plan.get("warm_output"))
            with open(arm_dir / "_warm.log", "w") as f:
                try:
                    subprocess.run(wcmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
        t0 = time.time()
        with open(arm_dir / f"{name}.log", "w") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
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
                g = res.get
                self.md(f"| {wl} | {test} | {conc_txt} | {n} | {fmt(g('output_throughput'))} | {fmt(g('request_throughput'), 2)} "
                        f"| {fmt(g('median_ttft_ms'))} | {fmt(g('p99_ttft_ms'))} | {fmt(g('median_itl_ms'), 1)} "
                        f"| {fmt(g('p99_itl_ms'), 1)} | {dt:.0f} |")
            else:
                why = "timed out" if rc == -9 else f"failed (rc={rc}), see {arm}/{name}.log"
                self.md(f"| {wl} | {test} | {conc_txt} | {n} | {why} | | | | | | {dt:.0f} |")

    def quality_check(self, arm):
        path = self.dir / arm / "quality.json"
        if self.a.dry_run:
            print(f"    quality check: {len(quality_cases())} tool-call prompts at temperature 0")
            return
        if path.exists():
            print("  skip (already done): quality check")
            return

        def one(case):
            prompt, tool, expected = case
            body = json.dumps({"model": self.prof["served"], "messages": [{"role": "user", "content": prompt}],
                               "tools": Q_TOOLS, "temperature": 0, "max_tokens": 512,
                               **self.prof["quality_extra"]}).encode()
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
            return row

        with concurrent.futures.ThreadPoolExecutor(8) as ex:  # 8 at a time: same answers, a fraction of the time
            rows = list(ex.map(one, quality_cases()))
        n = len(rows)
        summ = dict(n=n, tool_ok=sum(r["tool_ok"] for r in rows) / n, args_ok=sum(r["args_ok"] for r in rows) / n)
        path.write_text(json.dumps(dict(summary=summ, cases=rows), indent=1))
        self.md(f"\nTool-call quality: right tool {summ['tool_ok']:.0%}, right tool and arguments {summ['args_ok']:.0%} "
                f"({n} prompts, temperature 0).")

    def run_arm(self, arm, arms_left):
        sweeps = self.plan["sweeps"].get(arm, self.plan["sweeps"]["_other"])
        if self.budget and self.left_min() < 3:
            self.md(f"\n## {arm}: skipped\n\nTime budget used ({self.elapsed_min():.0f} of {self.budget:g} min).")
            return
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
            self.run_point(arm, "agent", "warmup", 0, 32, conc=16, record=False)
            for wl, points in sweeps.items():
                for c, n in points:
                    self.run_point(arm, wl, "conc", c, n, conc=c)
                if not self.a.dry_run:
                    m = self.http("/metrics")
                    if m:
                        (self.dir / arm / f"metrics_{wl}.txt").write_text(m)
            if self.plan.get("quality", True):
                self.quality_check(arm)
            ex = self.plan.get("extras")
            if arm == "baseline" and ex:
                need = self.plan.get("reserve_min", 0) * arms_left
                if self.budget and self.left_min() < need + 3:
                    self.md(f"\nSkipped the cold-start, QPS, p99.9 and burst tests: {self.left_min():.0f} min left, "
                            f"the other {arms_left} setting(s) need ~{need:g}.")
                else:
                    for c, n in ex["cold"]:  # cold start: empty cache, everyone arrives at once
                        self.run_point(arm, "agent", "cold", c, n, conc=c, warm=False)
                    for r, n in ex["rates"]:
                        self.run_point(arm, "agent", "rate", r, n, rate=r)
                    c, n = ex["p999"]
                    self.run_point(arm, "agent", "p999", c, n, conc=c)  # long run for p99.9 tails
                    self.run_point(arm, "agent", "burst", ex["burst"], ex["burst"])  # every request sent at once
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
        p = subprocess.run([sys.executable, str(self.here / "make_dashboard.py"), str(self.dir)],
                           capture_output=True, text=True)
        if p.returncode == 0:
            print(f"  dashboard updated: {self.dir / 'dashboard.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "qwen3-4b"), choices=list(MODELS),
                    help="model profile: serve script, settings and test sizes (default qwen3-4b)")
    ap.add_argument("--gpu", choices=["ampere", "hopper", "other"],
                    help="GPU class; detected with nvidia-smi when omitted (set it for --dry-run off the GPU host)")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS),
                    help="settings to test (default: the plan's list for this model and GPU)")
    ap.add_argument("--quick", action="store_true", help="small pipeline check")
    ap.add_argument("--trace", action="store_true", help="short baseline traffic for run_phase4.sh's trace step")
    ap.add_argument("--budget-min", type=float, help="wall-time cap in minutes (default: the plan's; 0 = none)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands only")
    ap.add_argument("--resume", type=pathlib.Path, help="results dir of an interrupted run")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(os.environ.get(
        "RESULTS_DIR", pathlib.Path(os.environ.get("WORK_DIR", "~/qwen-serve")).expanduser() / "results")),
        help="results folder, the one ./serve.sh dashboard serves (default ~/qwen-serve/results)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 30000)))
    ap.add_argument("--work-dir", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("WORK_DIR", "~/qwen-serve")).expanduser())
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"),
                    help="tokenizer folder for the load generator (default: the profile's checkpoint)")
    ap.add_argument("--point-timeout", type=int, help="seconds per test point (default: per model)")
    ap.add_argument("--no-restore", action="store_true", help="leave the last setting's server running")
    a = ap.parse_args()
    prof = MODELS[a.model]
    a.gpu_name, a.gpu_gb, detected = gpu_info()
    a.gpu = a.gpu or detected
    a.mode = "trace" if a.trace else "quick" if a.quick else "full"
    a.plan = pick(prof["plans"], a.gpu)[a.mode]
    a.model_dir = a.model_dir or str(pathlib.Path(prof["model_dir"]).expanduser())
    a.point_timeout = a.point_timeout or prof["point_timeout"]

    def usable(x):
        s = ARMS[x]
        return a.gpu in s.get("gpus", (a.gpu,)) and (a.gpu_gb is None or a.gpu_gb >= s.get("min_gb", 0))
    # Resolve version-dependent flag names and drop settings this SGLang can't run (argparse prefix
    # matching otherwise turns a renamed flag into an "ambiguous option" crash at server start).
    known = server_flags()
    if known is None and not a.dry_run:
        print("NOTE: couldn't list the SGLang server's flags; running without the CUDA-graph size cap.")
    cg = next((f for f in ("--cuda-graph-max-bs-decode", "--cuda-graph-max-bs") if known and f in known), None)
    prof["base_flags"] = {k: (v.replace("{cg}", cg) if cg else re.sub(r"\{cg\} \d+ ?", "", v)).strip()
                          for k, v in prof["base_flags"].items()}
    if known:
        for name, spec in ARMS.items():
            missing = [f for f in spec["flags"].split() if f.startswith("--") and f not in known]
            if missing:
                spec["gpus"] = ()
                print(f"NOTE: setting {name} needs {missing}, which this SGLang doesn't have; skipping it.")
        bad = [f for f in pick(prof["base_flags"], a.gpu).split() if f.startswith("--") and f not in known]
        if bad:
            sys.exit(f"This SGLang doesn't accept {bad}; tell the maintainer (run_bench.py MODELS base_flags).")
    wanted = a.arms or a.plan.get("arms") or prof["arms"]
    dropped = [x for x in wanted if x not in prof["arms"] or not usable(x)]
    if dropped:
        print(f"NOTE: settings {dropped} don't apply to {a.model} on this GPU ({a.gpu_name or a.gpu}); skipping them.")
    a.arms = [x for x in prof["arms"] if x in wanted and x not in dropped]  # profile order, baseline first

    r = Runner(a)
    if not a.dry_run:
        if not r.serve.exists():
            sys.exit(f"{r.serve.name} not found next to {__file__}")
        r.check_bench()
    r.dir.mkdir(parents=True, exist_ok=True)
    meta = r.write_meta()
    print(f"Model: {prof['label']} via {prof['serve']} · GPU: {a.gpu_name or '?'} ({a.gpu}) · plan: {a.mode} · "
          f"settings: {', '.join(a.arms)} · budget: {r.budget or 'none'} min")
    print(f"Results: {r.dir}")
    r.log_header(meta)
    for i, arm in enumerate(a.arms):
        r.run_arm(arm, arms_left=len(a.arms) - i - 1)
    r.md(f"\nFinished after {r.elapsed_min():.1f} min.")
    if prof["restore"] and not a.dry_run and not a.no_restore and a.arms and a.arms[-1] != "baseline":
        r.restart("baseline", note="restoring baseline server")
    print(f"\nDone. Readable log: {r.log_md}\nDashboard: {r.dir / 'dashboard.html'}")
    print(f"View it: ./serve.sh dashboard, then http://127.0.0.1:{os.environ.get('DASH_PORT', '8000')}/{r.dir.name}/dashboard.html")
    if not prof["restore"]:
        print(f"The server is still running the last setting. For Harbor: ./{prof['serve']} restart (defaults)")
    print(f"For git: python export_results.py --phase3 {r.dir}")


if __name__ == "__main__":
    main()
