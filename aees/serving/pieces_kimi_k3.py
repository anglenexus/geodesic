#!/usr/bin/env python3
"""
pieces_kimi_k3.py - the Kimi K3 profiling run as small pieces (5 min or less), checkpointed to S3 after each one,
so a preempted or restarted GPU machine picks up at the next unfinished piece.

    python pieces_kimi_k3.py run    --run-id kimi-20261103a   # recover_and_check_status, then the pending pieces
    python pieces_kimi_k3.py status --run-id kimi-20261103a   # what is done / next (works from the laptop too)
    python pieces_kimi_k3.py list                            # the pieces and stages

S3 layout under s3://<bucket>/serving/kimi-k3/runs/<run-id>/:
    files/...                     a mirror of the local results folder (Phase 4 json, Phase 3 run folder, export)
    done/<piece>.json             written once a piece has finished and its files are uploaded
    attempts/<piece>-<time>.json  every try, with exit code and seconds (failed twice -> skipped)
The bucket allows writes but not deletes, so nothing is ever removed: a new run gets a new run id.

Options (env): RESULTS_DIR (local results, default /results/results), S3_BUCKET, PIECE_MIN=5 (time limit per piece),
DEADLINE (unix time: start no piece that could run past it), HOLD_MIN=0 (keep the server up for smoke tests at the end),
SIMULATE=1 (pieces only write a small file and sleep: tests the bookkeeping without a GPU).
"""
import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(os.environ.get("RESULTS_DIR", "/results/results"))
BUCKET = os.environ.get("S3_BUCKET", "aees-grpo-backups-611728564241")
PREFIX = "serving/kimi-k3/runs"
PIECE_MIN = float(os.environ.get("PIECE_MIN", "5"))
SIMULATE = os.environ.get("SIMULATE") == "1"
PORT = int(os.environ.get("PORT", "30000"))

# (piece, stage, what). Stage "p3" pieces need the server; "what" is a run_bench --points value.
PIECES = [
    ("p4_mla", "phase4", "fi_bench_kimi.py mla: MLA decode, TP8 vs DCP8, BF16 vs FP8 cache"),
    ("p4_fp4", "phase4", "fi_bench_kimi.py fp4: expert/attention GEMMs in BF16, FP8, MXFP8, MXFP4, NVFP4"),
    ("p4_report", "phase4", "fi_bench_kimi.py report: fi_report.html"),
    ("p3_agent_c1", "phase3", "agent:conc:1"),
    ("p3_agent_c16", "phase3", "agent:conc:16"),
    ("p3_agent_c64", "phase3", "agent:conc:64"),
    ("p3_chat_c64", "phase3", "chat:conc:64"),
    ("p3_long_c16", "phase3", "long:conc:16"),
    ("p3_quality", "phase3", "quality"),
    ("p3_cold16", "phase3", "agent:cold:16"),
    ("p3_rate4", "phase3", "agent:rate:4"),
    ("p3_p999", "phase3", "agent:p999:64"),
    ("p3_burst", "phase3", "agent:burst:256"),
    ("export", "export", "export_results.py: small, committable folder in bench_results format"),
]


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------- S3 (boto3, or the aws CLI)
class S3:
    def __init__(self, run_id):
        self.root = f"{PREFIX}/{run_id}"
        try:
            import boto3
            self.c = boto3.client("s3")
        except ImportError:
            self.c = None

    def keys(self, sub):
        pre = f"{self.root}/{sub}"
        if self.c:
            out, tok = [], None
            while True:
                kw = dict(Bucket=BUCKET, Prefix=pre, **({"ContinuationToken": tok} if tok else {}))
                r = self.c.list_objects_v2(**kw)
                out += [(o["Key"][len(pre):], o["Size"]) for o in r.get("Contents", [])]
                if not r.get("IsTruncated"):
                    return out
                tok = r["NextContinuationToken"]
        p = subprocess.run(["aws", "s3", "ls", "--recursive", f"s3://{BUCKET}/{pre}"], capture_output=True, text=True)
        rows = [ln.split(None, 3) for ln in p.stdout.splitlines() if ln.strip()]
        return [(r[3][len(pre):], int(r[2])) for r in rows if len(r) == 4]

    def put(self, key, path=None, body=None):
        full = f"{self.root}/{key}"
        if self.c:
            if path:
                self.c.upload_file(str(path), BUCKET, full)
            else:
                self.c.put_object(Bucket=BUCKET, Key=full, Body=body.encode())
            return
        if body is not None:
            tmp = pathlib.Path("/tmp") / f"s3put_{os.getpid()}"
            tmp.write_text(body)
            path = tmp
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(path), f"s3://{BUCKET}/{full}"], check=True)

    def get(self, key, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        full = f"{self.root}/{key}"
        if self.c:
            self.c.download_file(BUCKET, full, str(path))
        else:
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{full}", str(path)], check=True)


def upload_changed(s3, remote):
    """Mirror the local results folder to files/: only new or resized files (results only grow)."""
    n = 0
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(RESULTS))
            if remote.get(rel) != p.stat().st_size:
                s3.put(f"files/{rel}", path=p)
                remote[rel] = p.stat().st_size
                n += 1
    return n


# ----------------------------------------------------------------------------- recover and check status
def recover_and_check_status(run_id, restore=True):
    """Restore the run's files from S3 and return (s3, remote file sizes, done set, failures per piece)."""
    s3 = S3(run_id)
    remote = dict(s3.keys("files/"))
    done = {k[:-5] for k, _ in s3.keys("done/") if k.endswith(".json")}
    fails = {}
    for k, _ in s3.keys("attempts/"):  # <piece>-<time>-rc<exit code>.json; rc99 = server not ready (not the piece's fault)
        if not k.endswith(("-rc0.json", "-rc99.json")):
            fails[k.split("-")[0]] = fails.get(k.split("-")[0], 0) + 1
    if restore:
        RESULTS.mkdir(parents=True, exist_ok=True)
        got = 0
        for rel, size in remote.items():
            local = RESULTS / rel
            if not local.exists() or local.stat().st_size != size:
                s3.get(f"files/{rel}", local)
                got += 1
        log(f"recovered {got} file(s) from s3://{BUCKET}/{s3.root}/files/ ({len(remote)} there)")
    nxt = next((p for p, *_ in PIECES if p not in done and fails.get(p, 0) < 2), None)
    log(f"run {run_id}: {len(done)}/{len(PIECES)} pieces done; next: {nxt or 'nothing (complete)'}")
    for p, stage, what in PIECES:
        mark = "done" if p in done else ("SKIP (failed twice)" if fails.get(p, 0) >= 2 else
                                          f"pending{f' ({fails[p]} failed try)' if fails.get(p) else ''}")
        print(f"   {p:14s} {stage:7s} {mark:22s} {what}", flush=True)
    return s3, remote, done, fails


# ----------------------------------------------------------------------------- pieces
def server_ready():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
        return True
    except Exception:
        return False


def p3_dir(run_id):
    return RESULTS / f"kimi-k3_b300_{run_id}"


def fi_dir():
    return RESULTS / "fi_kimi-k3"


def piece_cmd(run_id, piece, stage, what):
    env = dict(os.environ, FI_OUT=str(fi_dir()))
    if stage == "phase4":
        return [sys.executable, "fi_bench_kimi.py", piece.split("_", 1)[1]], env
    if stage == "phase3":
        d = p3_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        return [sys.executable, "run_bench.py", "--model", "kimi-k3", "--reuse-server", "--no-restore",
                "--budget-min", "0", "--resume", str(d), "--points", what], env
    args = [sys.executable, "export_results.py", "--dest", str(RESULTS / "export")]
    if p3_dir(run_id).exists():
        args += ["--phase3", str(p3_dir(run_id))]
    if fi_dir().exists():
        args += ["--phase4", str(fi_dir())]
    return args, env


def run_piece(run_id, piece, stage, what):
    if SIMULATE:
        out = RESULTS / "simulate" / f"{piece}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"{piece} {datetime.datetime.now().isoformat()}\n")
        time.sleep(float(os.environ.get("SIMULATE_SEC", "1")))
        return 0
    cmd, env = piece_cmd(run_id, piece, stage, what)
    try:
        return subprocess.run(cmd, cwd=HERE, env=env, timeout=PIECE_MIN * 60).returncode
    except subprocess.TimeoutExpired:
        log(f"{piece}: stopped at its {PIECE_MIN:g}-minute limit")
        return 124


def ensure_server():
    if SIMULATE or server_ready():
        return True
    log("server not running: starting it (serve_kimi_k3.sh start)")
    work = pathlib.Path(os.environ.get("WORK_DIR", "/root/kimi-serve"))
    (work / "logs").mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(work / "logs" / "server_start.txt", "w") as f:
        rc = subprocess.run(["bash", "serve_kimi_k3.sh", "start"], cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"server start: exit {rc} after {time.time() - t:.0f} s")
    # Keep a copy of each start's log with the results, so it reaches S3 with the next piece.
    starts = RESULTS / "server_starts"
    starts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy(work / "logs" / "server_start.txt", starts / f"server_start_{stamp}.txt")
    latest = work / "logs" / "latest.log"
    if latest.exists():
        shutil.copy(latest.resolve(), starts / f"server_{stamp}.log")
    return rc == 0 and server_ready()


def set_status(s3, piece, stage, note=""):
    """status/current.json: what the machine is doing now (the progress dashboard shows it)."""
    s3.put("status/current.json", body=json.dumps({"piece": piece, "stage": stage, "note": note, "time": time.time(),
                                                    "host": os.uname().nodename}))


def run(run_id):
    s3, remote, done, fails = recover_and_check_status(run_id)
    set_status(s3, "recover", "start", f"{len(done)}/{len(PIECES)} pieces already done")
    deadline = float(os.environ.get("DEADLINE", "0")) or None
    for piece, stage, what in PIECES:
        if piece in done or fails.get(piece, 0) >= 2:
            continue
        if deadline and time.time() + PIECE_MIN * 60 > deadline:
            log(f"stopping before {piece}: it could run past the deadline")
            break
        if stage == "phase3" and not SIMULATE and not server_ready():
            set_status(s3, "server", "server", "loading 1.56 TB of weights, then CUDA graphs (~13 min)")
        if stage == "phase3" and not ensure_server():
            s3.put(f"attempts/{piece}-{int(time.time())}-rc99.json", body=json.dumps({"error": "server not ready"}))
            upload_changed(s3, remote)
            log("server failed to start; stopping (see server_starts/ in the results)")
            break
        log(f"piece {piece} ({stage}): {what}")
        set_status(s3, piece, stage, what)
        t = time.time()
        rc = run_piece(run_id, piece, stage, what)
        secs = round(time.time() - t)
        n = upload_changed(s3, remote)  # results first, then the marker: a done marker always has its files
        s3.put(f"attempts/{piece}-{int(t)}-rc{rc}.json", body=json.dumps({"rc": rc, "seconds": secs}))
        if rc == 0:
            s3.put(f"done/{piece}.json", body=json.dumps({"seconds": secs, "files_uploaded": n,
                                                          "time": datetime.datetime.now().isoformat()}))
            log(f"piece {piece} done in {secs} s; {n} file(s) uploaded")
        else:
            fails[piece] = fails.get(piece, 0) + 1
            log(f"piece {piece} failed (exit {rc}, try {fails[piece]}); {n} file(s) uploaded")
    hold = float(os.environ.get("HOLD_MIN", "0"))
    set_status(s3, "finished", "end", "all pieces done or skipped" + (f"; hold window {hold:g} min" if hold else ""))
    if hold > 0 and not SIMULATE:
        log(f"hold window: {hold:g} min for smoke tests and dashboards (touch /tmp/STOP to end it)")
        end = time.time() + hold * 60
        while time.time() < end and not os.path.exists("/tmp/STOP"):
            time.sleep(15)
    recover_and_check_status(run_id, restore=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "list"])
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    a = ap.parse_args()
    if a.cmd == "list":
        for i, (p, stage, what) in enumerate(PIECES, 1):
            print(f"{i:2d}. {p:14s} {stage:7s} {what}")
        return
    if not a.run_id:
        sys.exit("--run-id (or RUN_ID) is required")
    if a.cmd == "status":
        recover_and_check_status(a.run_id, restore=False)
    else:
        os.chdir(HERE)
        run(a.run_id)


if __name__ == "__main__":
    main()
