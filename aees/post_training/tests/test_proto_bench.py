"""Dry run of proto/bench.sh with stub docker / nvidia-smi / sudo / curl: phase order, MPS on/off, the
trainer's flags and SM cap, the LoRA fallback, phase.json contents, and that report.py reads the result."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from proto import report as rp
from tests.test_proto_train import REAL

HERE = Path(__file__).resolve().parents[1]

DOCKER = r'''#!/usr/bin/env bash
echo "docker $*" >> "$STUB/log"
case "$1" in
  run) [[ "$*" == *proto_dcgm* ]] && exit 1; echo cid ;;
  exec)
    shift; envs=""
    while [[ "$1" == -* ]]; do
      case "$1" in -e) envs="$envs $2"; shift 2 ;; *) shift ;; esac
    done
    shift                                                    # container
    cmd="$*"
    out=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p' | sed "s#/root/proto_out#$DATA/proto#")
    case "$cmd" in
      pkill*) rm -f "$STUB/server" ;;
      pgrep*) [ -f "$STUB/server" ]; exit $? ;;
      *sglang.launch_server*)
        if [ -n "${STUB_LORA_FAILS:-}" ] && [[ "$cmd" == *--enable-lora* ]]; then exit 0; fi
        if [ -n "${STUB_KV8_FAILS:-}" ] && [[ "$cmd" == *fp8_e4m3* ]]; then exit 0; fi
        touch "$STUB/server" ;;
      *proto.adapter*) mkdir -p "$DATA/proto/$RUN/adapter"; echo '{}' > "$DATA/proto/$RUN/adapter/adapter_config.json" ;;
      *--pcie-only*) mkdir -p "$(dirname "$out")"; echo '{"t": 1, "type": "pcie", "h2d_gbps": 50, "d2h_gbps": 50}' >> "$out" ;;
      *proto.train_proxy*)
        mkdir -p "$(dirname "$out")"
        seq=49152; [[ "$cmd" == *"--seq 32768"* ]] && seq=32768
        off=false; [[ "$cmd" == *--offload* ]] && off=true
        if [ -n "${STUB_OOM:-}" ] && [ "$seq" = 49152 ] && [[ "$cmd" == *--offload* ]]; then
          echo '{"t": 1, "type": "oom"}' >> "$out"; exit 3
        fi
        t=$(date +%s)
        echo "{\"t\": $t, \"type\": \"start\", \"seq\": $seq, \"offload\": $off}" >> "$out"
        for k in 1 2 3; do
          echo "{\"t\": $((t + k)), \"type\": \"step\", \"step\": $k, \"seconds\": 1.0, \"tokens\": $seq, \"tflops\": 400, \"peak_gb\": 80}" >> "$out"
        done ;;
      *proto.agent_load*)
        mkdir -p "$(dirname "$out")"; t=$(date +%s)
        echo "{\"agent\": 0, \"t_send\": $t, \"t_first\": $t, \"t_end\": $t, \"out\": 1, \"sleep\": 0, \"error\": null}" >> "$out" ;;
    esac ;;
esac
exit 0
'''
STUBS = {
    "docker": DOCKER,
    "nvidia-smi": '#!/usr/bin/env bash\nif [[ "$*" == *utilization* ]]; then echo "50, 60, 1000, 143771, 500"; else echo 1000; fi\n',
    "curl": '#!/usr/bin/env bash\n[ -f "$STUB/server" ]\n',
    "sudo": '#!/usr/bin/env bash\n"$@"\n',
    "nvidia-cuda-mps-control": '#!/usr/bin/env bash\necho "mps $* $(cat)" >> "$STUB/log"\n',
    "pkill": '#!/usr/bin/env bash\necho "pkill $*" >> "$STUB/log"\n',            # never signal real processes
}


def dry_run(tmp_path, **env):
    stub = tmp_path / "stub"
    stub.mkdir()
    for name, body in STUBS.items():
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    data = tmp_path / "data"
    (data / "models" / "Qwen3.6-27B").mkdir(parents=True)
    (data / "models" / "Qwen3.6-27B" / "config.json").write_text(json.dumps(REAL))
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "STUB": str(stub), "DATA": str(data), "RUN": "r1",
         "MPS_PIPE": str(tmp_path / "mps"), "CO_WARMUP_S": "0", **env}
    p = subprocess.run(["bash", str(HERE / "proto" / "bench.sh")], env=e, capture_output=True, text=True, timeout=120,
                       stdin=subprocess.DEVNULL)
    log = (stub / "log").read_text().splitlines()
    return p, data / "proto" / "r1", log


def phase_meta(run, name):
    return json.loads((run / name / "phase.json").read_text())


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    return dry_run(tmp_path_factory.mktemp("full"))


def test_all_phases_run_in_order_with_their_settings(full):
    p, run, log = full
    assert p.returncode == 0, p.stderr[-2000:]
    order = [line.split("phase ")[1] for line in (run / "bench.log").read_text().splitlines() if "] phase " in line]
    assert order == ["pcie", "train_alone", "train_alone_offload", "serve_bf16", "serve_fp8_32", "serve_fp8",
                     "co_timeslice", "serve_fp8_mps", "co_mps", "co_mps70"]
    want = {"serve_bf16": ("bf16", False, 16, False, None), "serve_fp8_32": ("fp8", False, 32, False, None),
            "serve_fp8": ("fp8", False, 16, False, None), "co_timeslice": ("fp8", False, 16, True, None),
            "serve_fp8_mps": ("fp8", True, 16, False, None), "co_mps": ("fp8", True, 16, True, None),
            "co_mps70": ("fp8", True, 16, True, 70), "train_alone": (None, False, 0, False, None)}
    for name, (serve, mps, agents, co, pct) in want.items():
        m = phase_meta(run, name)
        assert (m["serve"], m["mps"], m["agents"], m["co"], m["mps_pct"], m["lora"]) == (serve, mps, agents, co, pct, True)
        assert m["t0"] <= m["t1"]
    assert (run / "config.json").exists() and (run / "serve_fp8" / "gpu.csv").read_text().count("143771") >= 1


def test_mps_and_trainer_flags(full):
    _, run, log = full
    assert [line.split()[1] for line in log if line.startswith("mps ")] == ["-d", "quit"]   # on, then off at exit
    servers = [line for line in log if "launch_server --model-path" in line]
    assert len(servers) == 3                                           # bf16, fp8, fp8 under MPS
    assert "mem-fraction-static 0.85" in servers[0] and "--kv-cache-dtype" not in servers[0]
    assert all("Qwen3.6-27B-FP8" in s and "0.36" in s and "fp8_e4m3" in s for s in servers[1:])
    assert "CUDA_MPS_PIPE_DIRECTORY" not in servers[1] and "CUDA_MPS_PIPE_DIRECTORY=/tmp/mps" in servers[2]
    first_mps = next(i for i, line in enumerate(log) if line.startswith("mps -d"))
    assert log.index(servers[2]) > first_mps > log.index(servers[1])  # MPS starts between the two FP8 servers
    trains = [line for line in log if "proto.train_proxy" in line and "--pcie-only" not in line]
    assert len(trains) == 5                                            # 2 alone + 3 co-run
    assert "--offload" not in trains[0] and all("--offload" in t for t in trains[1:])
    assert "CUDA_MPS" not in trains[2]                                 # time-slicing
    assert "CUDA_MPS_PIPE_DIRECTORY" in trains[3] and "ACTIVE_THREAD_PERCENTAGE" not in trains[3]
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70" in trains[4]
    agents = [line for line in log if "proto.agent_load" in line and "--hot-load" not in line]
    assert all("--lora proto" in a for a in agents) and any("--agents 32" in a for a in agents)
    assert sum("--hot-load" in line for line in log) == 1


def test_report_reads_the_dry_run(full):
    _, run, _ = full
    rep = rp.build(run)
    assert set(rep["verdicts"]) == {"co_timeslice", "co_mps", "co_mps70"}
    assert rep["verdicts"]["co_mps70"]["serve_baseline"] == "serve_fp8_mps"
    assert rep["verdicts"]["co_timeslice"]["train_baseline"] == "train_alone_offload"


def test_lora_fallback_and_oom_retry(tmp_path):
    p, run, log = dry_run(tmp_path, STUB_LORA_FAILS="1", STUB_OOM="1",
                          PHASES="train_alone_offload serve_fp8 co_timeslice", SERVE_MIN="0")
    assert p.returncode == 0, p.stderr[-2000:]
    servers = [line for line in log if "launch_server --model-path" in line]
    assert "--enable-lora" in servers[0] and "--enable-lora" not in servers[1] and len(servers) == 2
    assert phase_meta(run, "co_timeslice")["lora"] is False
    assert all("--lora" not in line for line in log if "proto.agent_load" in line)
    bench = (run / "bench.log").read_text()
    assert "SGLang failed to start (fp8, lora)" in bench and "retrying at 32k" in bench
    rows = [json.loads(x) for x in (run / "co_timeslice" / "train.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "oom" and rows[1]["seq"] == 32768


TRIAL_STUBS = {
    "curl": '#!/usr/bin/env bash\nn=$(( $(cat "$STUB/n" 2>/dev/null || echo 0) + 1 )); echo $n > "$STUB/n"\n[ "$n" -le "$UP_CALLS" ]\n',
    "harbor": '#!/usr/bin/env bash\necho "harbor $*" >> "$STUB/log"; sleep 0.3\n',
    "docker": '#!/usr/bin/env bash\necho "docker $*" >> "$STUB/log"\n',
    "ssh": '#!/usr/bin/env bash\necho "ssh $*" >> "$STUB/log"; sleep 1\n',
    "timeout": '#!/usr/bin/env bash\nshift; shift; exec "$@"\n',                  # timeout --signal=INT <s> cmd...
    "pkill": '#!/usr/bin/env bash\necho "pkill $*" >> "$STUB/log"\n',            # never signal real processes
}


def test_ec2_trials_keep_n_slots_busy_and_stop_when_the_server_goes(tmp_path):
    stub, home = tmp_path / "stub", tmp_path / "home"
    stub.mkdir()
    (home / "proto").mkdir(parents=True)
    for name, body in TRIAL_STUBS.items():
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    for f in ("ec2_trials.sh", "llm_log_proxy.py", "ec2_tasks.txt"):
        (home / "proto" / f).write_text((HERE / "proto" / f).read_text())
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "STUB": str(stub), "HOME": str(home),
           "H200": "10.0.0.5", "H200_USER": "nebius", "RUN": "r1", "N": "3", "MAX_MIN": "1", "DOWN_S": "2",
           "POLL_S": "1", "STAGGER_S": "0", "UP_CALLS": "12"}
    p = subprocess.run(["bash", str(home / "proto" / "ec2_trials.sh")], env=env, capture_output=True, text=True,
                       timeout=60, stdin=subprocess.DEVNULL)
    assert p.returncode == 0 and "server down 2 s: stopping" in p.stdout, p.stdout + p.stderr
    log = (stub / "log").read_text().splitlines()
    runs = [line for line in log if line.startswith("harbor run")]
    tasks = (HERE / "proto" / "ec2_tasks.txt").read_text().split()
    assert len(runs) >= 3
    for r in runs:
        assert "-m hosted_vllm/qwen3.6-27b" in r and "http://127.0.0.1:8001/v1" in r and "-n 1" in r
        assert '"enable_thinking": false' in r and '"max_input_tokens": 49152' in r
        assert r.split("-i terminal-bench/")[1].split()[0] in tasks
    assert {r.split("--job-name ")[1].split("_")[2] for r in runs} == {"s0", "s1", "s2"}     # 3 slots
    tunnel = next(line for line in log if line.startswith("ssh "))
    assert "-L 30001:127.0.0.1:30000 nebius@10.0.0.5" in tunnel and "proto_tunnel" in tunnel
    assert any(line.startswith("docker ps -q") for line in log)                              # cleanup
    assert "pkill -INT -f harbor run.*--job-name proto_r1_" in log                           # only this run's jobs


def test_fp8_kv_cache_fallback(tmp_path):
    p, run, log = dry_run(tmp_path, STUB_KV8_FAILS="1", PHASES="serve_fp8", SERVE_MIN="0")
    assert p.returncode == 0, p.stderr[-2000:]
    servers = [line for line in log if "launch_server --model-path" in line]
    assert ["--enable-lora" in x for x in servers] == [True, False, False]
    assert ["fp8_e4m3" in x for x in servers] == [True, True, False] and "Qwen3.6-27B-FP8" in servers[2]
    assert phase_meta(run, "serve_fp8")["lora"] is False
