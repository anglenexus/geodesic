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
import subprocess
import sys
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
                 "long": [1, 4, 8, 16, 32]},
    "fp8_kv": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128], "long": [8, 16, 32]},
    "_other": {"agent": [1, 16, 64, 128], "chat": [1, 16, 64, 128]},
}
QUICK = {
    "baseline": {"agent": [1, 16, 64], "chat": [16, 64], "long": [8]},
    "_other": {"agent": [16, 64]},
}
RATES_FULL, RATES_QUICK = [1, 2, 4, 8, 16, 32], [2, 8]  # offered req/s for the QPS sweep
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
        p = subprocess.run([str(self.serve), "restart"], env=dict(os.environ, EXTRA_ARGS=extra),
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

    def run_point(self, arm, wl, mode, value, n, conc=None, rate=None, record=True):
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
        self.http("/flush_cache", "POST")  # every point starts from an empty prefix cache
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
            rec = dict(arm=arm, workload=wl, mode=mode, value=value, num_prompts=n,
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

    def run_arm(self, arm):
        plan = QUICK if self.a.quick else FULL
        sweeps = plan.get(arm, plan["_other"])
        if not self.restart(arm):
            return
        gpu = None
        if not self.a.dry_run:
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
            if arm == "baseline":
                q = self.a.quick
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
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
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


if __name__ == "__main__":
    main()
