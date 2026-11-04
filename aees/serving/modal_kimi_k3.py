"""
modal_kimi_k3.py - Kimi K3 serving tests (Phase 3 + Phase 4) on one 8x B300 node on Modal.

Run from geodesic/aees/serving/ on the laptop (Modal CLI: `uv tool install modal && modal setup`):

    modal run modal_kimi_k3.py::download        # once: 1.56 TB of weights into the kimi-k3-weights Volume (CPU only)
    modal run modal_kimi_k3.py::dev             # ~3 min on 1x B300 (~$0.40): checks the image and the Phase 4 kernels
    modal run --detach modal_kimi_k3.py::run    # the test run on 8x B300 as 5-min pieces checkpointed to S3
                                                # (pieces_kimi_k3.py); --run-id <id> continues an earlier run
    modal run modal_kimi_k3.py::fetch           # copy the exported results into bench_results/
    modal run modal_kimi_k3.py::teardown --confirm   # back up all results to S3, verify, delete both Volumes

The run, in order (see kimi_k3_plan.md): Phase 4 kernels (15 min cap, GPUs to themselves) -> server start
(~13 min, mostly reading 1.56 TB from the Volume) -> Phase 3 (15 min cap) -> export -> a hold window for
smoke tests and dashboards through the tunnel -> exit.

Kill switches: the GPU function's timeout (KILL_MIN, default 120) makes Modal stop the container whatever
happens; the driver exits by itself after the hold window; `touch /tmp/STOP` over SSH (or `modal app stop`)
ends it early. Cost: 8 x $7.10/h for the GPUs plus ~$5/h of CPU and memory, ~$62/h.
"""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
MODEL_ID = "moonshotai/Kimi-K3"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.23-cu130"  # CUDA 13 build; B300 needs CUDA 13.1+ drivers (Modal: 13.3)
WEIGHTS, RESULTS = "/models", "/results"
MODEL_DIR = f"{WEIGHTS}/Kimi-K3"
KILL_MIN = int(os.environ.get("KILL_MIN", "120"))  # hard stop for the GPU container, minutes

app = modal.App("kimi-k3-serving")
weights = modal.Volume.from_name("kimi-k3-weights", create_if_missing=True, version=2)
results = modal.Volume.from_name("kimi-k3-results", create_if_missing=True, version=2)

dl_image = (modal.Image.debian_slim(python_version="3.12")
            .pip_install("huggingface_hub[hf_xet]>=0.35")
            .env({"HF_XET_HIGH_PERFORMANCE": "1"}))
gpu_image = (modal.Image.from_registry(SGLANG_IMAGE)
             .apt_install("openssh-server", "tmux", "curl")
             .pip_install("pandas", "plotly", "boto3")
             .env({"MODEL_DIR": MODEL_DIR, "RESULTS_DIR": f"{RESULTS}/results", "WORK_DIR": "/root/kimi-serve",
                   "PYTHONUNBUFFERED": "1"})
             .add_local_dir(HERE, "/root/serving",
                            ignore=["bench_results", "*.pdf", "__pycache__", ".DS_Store"]))

# AWS credentials for the S3 checkpoints (pieces_kimi_k3.py), as a Modal secret named kimi-k3-aws.
# KIMI_S3=0 at launch leaves it out (download, dev, or a run that checkpoints to the Volume only).
aws = [modal.Secret.from_name("kimi-k3-aws")] if os.environ.get("KIMI_S3", "1") == "1" else []

# ----------------------------------------------------------------------------- weights
@app.function(image=dl_image, volumes={WEIGHTS: weights}, cpu=8, memory=16384, timeout=4 * 3600)
def download_part(files: list[str]) -> float:
    """Download some of the checkpoint's files into the Volume; returns GB written."""
    from huggingface_hub import hf_hub_download
    t, gb = time.time(), 0.0
    for f in files:
        p = pathlib.Path(hf_hub_download(MODEL_ID, f, local_dir=MODEL_DIR))
        gb += p.stat().st_size / 1e9
        print(f"{f}: {p.stat().st_size / 1e9:.1f} GB, {gb / (time.time() - t):.2f} GB/s so far", flush=True)
    weights.commit()
    return gb


@app.local_entrypoint()
def download(parts: int = 12):
    """Split the repo's files over `parts` CPU containers (shards are ~16 GB each)."""
    import json
    import urllib.request
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true", timeout=60) as r:
        siblings = json.load(r)["siblings"]
    files = sorted(((s["rfilename"], s.get("size") or 0) for s in siblings
                    if not s["rfilename"].startswith((".eval_results/", "assets/"))), key=lambda x: -x[1])
    groups = [[] for _ in range(parts)]
    sizes = [0] * parts
    for name, size in files:  # largest first into the emptiest group
        i = sizes.index(min(sizes))
        groups[i].append(name)
        sizes[i] += size
    print(f"{len(files)} files, {sum(sizes) / 1e9:,.0f} GB over {parts} containers")
    t = time.time()
    total = sum(download_part.map(groups))
    print(f"Done: {total:,.0f} GB in {(time.time() - t) / 60:.1f} min -> Volume kimi-k3-weights:/Kimi-K3")


# ----------------------------------------------------------------------------- SSH tunnel
def start_sshd(pubkey: str) -> None:
    ssh = pathlib.Path("/root/.ssh")
    ssh.mkdir(mode=0o700, exist_ok=True)
    (ssh / "authorized_keys").write_text(pubkey.strip() + "\n")
    (ssh / "authorized_keys").chmod(0o600)
    pathlib.Path("/run/sshd").mkdir(exist_ok=True)
    subprocess.run(["ssh-keygen", "-A"], check=True, capture_output=True)
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e", "-o", "PasswordAuthentication=no",
                      "-o", "PermitRootLogin=prohibit-password", "-o", "AllowTcpForwarding=yes"])


def commit_every(vol, seconds, stop):
    while not stop.wait(seconds):
        try:
            vol.commit()
        except Exception as e:
            print(f"(volume commit failed: {e})", flush=True)


# ----------------------------------------------------------------------------- preflight (CPU only)
PREFLIGHT = r'''
import json, pathlib, subprocess, sys
d = pathlib.Path("/models/Kimi-K3")
idx = json.loads((d / "model.safetensors.index.json").read_text())
need = sorted(set(idx["weight_map"].values()))
have = [f for f in need if (d / f).exists()]
gb = sum((d / f).stat().st_size for f in have) / 1e9
print(f"weights: {len(have)}/{len(need)} shards, {gb:,.0f} GB" + ("" if len(have) == len(need) else "  MISSING: " + ", ".join(sorted(set(need) - set(have))[:10])))
sys.path.insert(0, "/root/serving")
import run_bench
h = subprocess.run([sys.executable, "-m", "sglang.bench_serving", "--help"], capture_output=True, text=True)
missing = [f for f in run_bench.REQUIRED_FLAGS if f not in h.stdout + h.stderr]
print("bench_serving flags:", "all present" if not missing else f"MISSING {missing}", "| random-ids:", "random-ids" in h.stdout)
from sglang.benchmark.utils import get_tokenizer
tok = get_tokenizer(str(d))
print("tokenizer:", type(tok).__name__, "vocab", len(tok), "| 'Hello world' ->", tok.encode("Hello world"))
'''


@app.function(image=gpu_image, volumes={WEIGHTS: weights}, cpu=4, memory=16384, timeout=900)
def preflight_check() -> None:
    p = subprocess.run([sys.executable, "-c", PREFLIGHT], capture_output=True, text=True)
    print(p.stdout, p.stderr[-3000:] if p.returncode else "", f"(exit {p.returncode})", flush=True)


@app.local_entrypoint()
def preflight():
    preflight_check.remote()


# ----------------------------------------------------------------------------- S3 self-test (CPU only)
@app.function(image=gpu_image, secrets=aws, cpu=2, timeout=600)
def s3_selftest_remote(run_id: str) -> None:
    """The pieces runner in SIMULATE mode with boto3 and the kimi-k3-aws secret: a run, then a restart from an
    empty folder (as after a preemption), which must restore the files and find all pieces done."""
    for attempt, folder in ((1, "/tmp/a"), (2, "/tmp/b")):
        env = dict(os.environ, SIMULATE="1", SIMULATE_SEC="0.2", RESULTS_DIR=folder)
        p = subprocess.run([sys.executable, "/root/serving/pieces_kimi_k3.py", "run", "--run-id", run_id],
                           env=env, capture_output=True, text=True)
        lines = [ln for ln in (p.stdout + p.stderr).splitlines() if "recovered" in ln or "pieces done" in ln
                 or "Error" in ln or "rror:" in ln]
        print(f"attempt {attempt} (exit {p.returncode}):\n  " + "\n  ".join(lines[:4] + lines[-1:]), flush=True)


@app.local_entrypoint()
def s3_selftest():
    s3_selftest_remote.remote("selftest-modal-" + time.strftime("%m%d%H%M%S"))


# ----------------------------------------------------------------------------- dev check (1x B300)
@app.function(image=gpu_image, gpu="B300", timeout=15 * 60, volumes={RESULTS: results})
def dev_check() -> str:
    """Image versions plus the Phase 4 microbenchmarks in --quick form, on one GPU."""
    cmds = [
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        "python -c 'import torch, flashinfer, sglang; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda,"
        " \"flashinfer\", flashinfer.__version__, \"sglang\", sglang.__version__)'",
        "cd /root/serving && python fi_bench_kimi.py mla --quick",
        "cd /root/serving && python fi_bench_kimi.py fp4 --quick",
        "cd /root/serving && python fi_bench_kimi.py report",
    ]
    out = []
    for c in cmds:
        p = subprocess.run(["bash", "-lc", c], capture_output=True, text=True,
                           env=dict(os.environ, FI_OUT="/tmp/fi_dev"))
        out.append(f"$ {c}\n{p.stdout[-6000:]}{p.stderr[-3000:]}(exit {p.returncode})")
        print(out[-1], flush=True)
    return "\n\n".join(out)


@app.local_entrypoint()
def dev():
    dev_check.remote()


# ----------------------------------------------------------------------------- the run (8x B300)
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))  # first start + 2 restarts after preemption




@app.function(image=gpu_image, gpu="B300:8", cpu=48, memory=400 * 1024, timeout=KILL_MIN * 60,
              volumes={WEIGHTS: weights, RESULTS: results}, secrets=aws)
def run_remote(pubkey: str, run_id: str, hold_min: int = 15) -> None:
    # Modal GPU functions are preemptible and restart with the same input, and the function timeout restarts
    # with them. So the budget lives on the results Volume: a deadline set by the first attempt and an attempt
    # count; a restarted attempt past either exits at once instead of paying for another weight load.
    results.reload()
    guard = pathlib.Path(RESULTS) / "guard.json"
    g = json.loads(guard.read_text()) if guard.exists() else {"deadline": time.time() + KILL_MIN * 60, "attempts": 0}
    g["attempts"] += 1
    guard.write_text(json.dumps(g))
    results.commit()
    left = g["deadline"] - time.time()
    print(f"Attempt {g['attempts']} of {MAX_ATTEMPTS}; {left / 60:.0f} min left of the {KILL_MIN}-min total budget",
          flush=True)
    if g["attempts"] > MAX_ATTEMPTS or left < 5 * 60:
        print("STOPPING: total budget or attempt limit reached (preempted too often); nothing started.", flush=True)
        return
    if g["attempts"] > 1:
        print("PREEMPTED earlier: resuming at the next unfinished piece (state from S3)", flush=True)
    start_sshd(pubkey)
    stop = threading.Event()
    threading.Thread(target=commit_every, args=(results, 60, stop), daemon=True).start()
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print("\n" + "=" * 100 + f"\nSSH tunnel (from geodesic/aees/serving on the laptop; leave it running):\n"
              f"  ssh -N -p {port} -L 30000:127.0.0.1:30000 -L 8000:127.0.0.1:8000 "
              f"-o StrictHostKeyChecking=no root@{host}\n"
              f"Shell:  ssh -p {port} -o StrictHostKeyChecking=no root@{host}     "
              f"(end early: touch /tmp/STOP)\n" + "=" * 100 + "\n", flush=True)
        env = dict(os.environ, HOLD_MIN=str(hold_min), DEADLINE=str(g["deadline"] - 120), RUN_ID=run_id)
        os.makedirs(f"{RESULTS}/results", exist_ok=True)
        subprocess.Popen([sys.executable, "-m", "http.server", "8000", "--bind", "127.0.0.1", "--directory",
                          f"{RESULTS}/results"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            p = subprocess.run([sys.executable, "/root/serving/pieces_kimi_k3.py", "run", "--run-id", run_id],
                               cwd="/root/serving", env=env, timeout=max(60, left - 120))
        except subprocess.TimeoutExpired:
            print("STOPPING: total budget reached", flush=True)
            p = subprocess.CompletedProcess([], 124)
        print(f"driver exited with {p.returncode}", flush=True)
    stop.set()
    results.commit()


@app.local_entrypoint()
def sync(every: int = 120, s3: str = "s3://aees-grpo-backups-611728564241/serving/kimi-k3/modal-" + time.strftime("%Y%m%d")):
    """Laptop-side checkpoint loop: results Volume -> .nebius/modal_raw -> S3 every `every` s (Ctrl-C to stop)."""
    raw = HERE / ".nebius" / "modal_raw"
    raw.mkdir(parents=True, exist_ok=True)
    while True:
        subprocess.run(["modal", "volume", "get", "--force", "kimi-k3-results", "/", str(raw)], capture_output=True)
        subprocess.run(["aws", "s3", "sync", str(raw), s3, "--only-show-errors"])
        n = sum(1 for p in raw.rglob("*") if p.is_file())
        print(f"[{time.strftime('%H:%M:%S')}] synced {n} files -> {s3}", flush=True)
        time.sleep(every)


@app.local_entrypoint()
def run(run_id: str = "", hold_min: int = 15, key: str = "~/.ssh/id_rsa.pub"):
    """New run (new run id, new budget), or `--run-id <id>` to continue an earlier run at its next piece."""
    pubkey = pathlib.Path(key).expanduser().read_text()
    run_id = run_id or "kimi-" + time.strftime("%Y%m%d-%H%M%S")
    # A new launch gets a new budget; restarts after preemption (same input, inside Modal) keep the old one.
    subprocess.run(["modal", "volume", "rm", "kimi-k3-results", "guard.json"], capture_output=True)
    print(f"Run {run_id}. Kill switch: {KILL_MIN} min of GPU time in total across restarts (~${KILL_MIN / 60 * 62:.0f}), "
          f"at most {MAX_ATTEMPTS} attempts. Pieces of <=5 min, each saved to "
          f"s3://aees-grpo-backups-611728564241/serving/kimi-k3/runs/{run_id}/ when it finishes. "
          f"Status: python pieces_kimi_k3.py status --run-id {run_id}")
    run_remote.remote(pubkey, run_id, hold_min)


# ----------------------------------------------------------------------------- results back to git
@app.local_entrypoint()
def fetch(dest: str = str(HERE / "bench_results")):
    """Copy the exported (small, committable) result folders from the Volume into bench_results/."""
    subprocess.run(["modal", "volume", "get", "--force", "kimi-k3-results", "export/", dest], check=True)
    print(f"Copied into {dest}; check `git status` and commit what's new.")


@app.local_entrypoint()
def teardown(confirm: bool = False, bucket: str = "s3://aees-grpo-backups-611728564241/serving/kimi-k3"):
    """Back up everything in kimi-k3-results to S3, check the upload, then delete both Volumes.
    The weights are not backed up: they are the public moonshotai/Kimi-K3 files (re-download: ~12 min, ~$1).
    Without --confirm it backs up and checks but deletes nothing."""
    import json
    import tempfile
    stamp = time.strftime("%Y%m%d-%H%M%S")
    local = pathlib.Path(tempfile.mkdtemp(prefix="kimi-k3-results-"))
    subprocess.run(["modal", "volume", "get", "kimi-k3-results", "/", str(local)], check=True)
    files = [p for p in local.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"Downloaded {len(files)} files, {size / 1e6:,.1f} MB to {local}")
    dest = f"{bucket}/{stamp}"
    subprocess.run(["aws", "s3", "sync", str(local), dest, "--only-show-errors"], check=True)
    listed = subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", dest + "/"],
                            capture_output=True, text=True, check=True).stdout
    n = int(next(ln.split(":")[1] for ln in listed.splitlines() if "Total Objects" in ln))
    s3_size = int(next(ln.split(":")[1] for ln in listed.splitlines() if "Total Size" in ln))
    ok = n == len(files) and s3_size == size
    print(f"S3 {dest}: {n} objects, {s3_size / 1e6:,.1f} MB -> {'matches' if ok else 'MISMATCH'}")
    (HERE / "bench_results" / "kimi_k3_backup.json").write_text(json.dumps(
        dict(s3=dest, objects=n, bytes=s3_size, time=stamp), indent=1) + "\n")
    if not ok:
        sys.exit("Backup check failed; nothing deleted.")
    if not confirm:
        print("Backup verified. Rerun with --confirm to delete the kimi-k3-weights and kimi-k3-results Volumes.")
        return
    for vol in ("kimi-k3-weights", "kimi-k3-results"):
        subprocess.run(["modal", "volume", "delete", "--yes", vol], check=True)
        print(f"Deleted Volume {vol}")
    print(subprocess.run(["modal", "volume", "list"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    sys.exit("Run with the Modal CLI, e.g. `modal run modal_kimi_k3.py::dev` (see the docstring).")
