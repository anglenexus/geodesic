"""Dry runs of the distributed-run scripts with stub aws / ssh / rsync / tmux / docker (dist_setup.md):
scripts/ec2_sandboxes.sh (laptop), scripts/ec2_box.sh (box), scripts/links.sh and scripts/preflight.sh (H200),
plus the config overrides they rely on."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from grpo.infra_filter import MIN_VALID

HERE = Path(__file__).resolve().parents[1]
LOG_STUB = '#!/usr/bin/env bash\na="$*"; echo "{name} ${{a//$\'\\n\'/ }}" >> "$STUB/log"\n{body}\nexit 0\n'
AWS = r'''
case "$*" in
  *"sts get-caller-identity --query Arn"*) echo arn:aws:iam::1:user/agent_aees ;;
  *"sts get-caller-identity --query Account"*) echo 1 ;;
  *budgets*) exit 1 ;;
  *describe-images*) echo ami-123 ;;
  *describe-security-groups*) echo sg-1 ;;
  *run-instances*)
    if [ -n "${STUB_NO_SPOT:-}" ] && [[ "$*" == *MarketType=spot* ]]; then exit 1; fi
    echo "i-1 i-2 i-3" ;;
  *describe-instances*) echo "10.0.0.1 10.0.0.2 10.0.0.3" ;;
  *terminate-instances*) echo "shutting-down shutting-down shutting-down" ;;
esac'''
SSH = r'''
for a in "$@"; do last="$a"; done
case "$last" in
  *"ss -ltn"*) echo 16 ;;
  *"127.0.0.1:30000"*) echo aees-stub ;;
esac
[[ "$*" == *"cat >>"* ]] && cat >> "$STUB/probe_rows"'''


def stubs(tmp_path, extra=None) -> dict:
    b = tmp_path / "bin"
    b.mkdir()
    bodies = {"aws": AWS, "ssh": SSH, "rsync": "", "scp": "", "curl": 'echo 1.2.3.4', "tmux": "", "sleep": "",
              "ssh-keygen": 'f="${@: -1}"; touch "$f" "$f.pub"', **(extra or {})}
    for name, body in bodies.items():
        (b / name).write_text(LOG_STUB.format(name=name, body=body))
        (b / name).chmod(0o755)
    return {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "STUB": str(tmp_path)}


def calls(tmp_path, name) -> list[str]:
    log = tmp_path / "log"
    return [l[len(name) + 1:] for l in log.read_text().splitlines() if l.startswith(name + " ")] if log.exists() else []


def sh(script, *args, env, cwd=HERE):
    return subprocess.run(["bash", str(HERE / script), *args], env=env, cwd=cwd, capture_output=True, text=True, timeout=60)


# ---- laptop: scripts/ec2_sandboxes.sh ----

def test_up_launches_spot_boxes_and_prepares_each(tmp_path):
    env = {**stubs(tmp_path), "STATE": str(tmp_path / "state")}
    r = sh("scripts/ec2_sandboxes.sh", "up", "3", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    run = [c for c in calls(tmp_path, "aws") if "run-instances" in c]
    assert len(run) == 1 and "--count 3" in run[0] and "m7a.8xlarge" in run[0] and "MarketType=spot" in run[0]
    assert "shutdown -h +900" in run[0] and "--instance-initiated-shutdown-behavior terminate" in run[0]
    state = (tmp_path / "state" / "boxes.env").read_text()
    assert 'BOX_IDS="i-1 i-2 i-3"' in state and 'BOX_IPS="10.0.0.1 10.0.0.2 10.0.0.3"' in state
    prep = [c for c in calls(tmp_path, "ssh") if "ec2_box.sh prepare" in c]
    assert sorted(c.split("ubuntu@")[1].split()[0] for c in prep) == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    copied = " ".join(calls(tmp_path, "rsync"))
    for f in ["ec2_box.sh", "agent_server.sh", "bake_tools.py", "setup.sh", "pins.sh", "harbor_parser.patch",
              "tunnel_key.pub", "sync_key.pub"]:
        assert f in copied
    assert (tmp_path / "state" / "tunnel_key").exists() and (tmp_path / "state" / "sync_key").exists()
    assert sh("scripts/ec2_sandboxes.sh", "up", env=env).returncode == 1      # refuses a second set


def test_up_defaults_to_two_boxes(tmp_path):                         # ec2_sizing.md: 2 x 32 trials
    env = {**stubs(tmp_path), "STATE": str(tmp_path / "state")}
    sh("scripts/ec2_sandboxes.sh", "up", env=env)
    assert any("run-instances" in c and "--count 2" in c for c in calls(tmp_path, "aws"))


def test_spot_failure_falls_back_to_on_demand(tmp_path):
    env = {**stubs(tmp_path), "STATE": str(tmp_path / "state"), "STUB_NO_SPOT": "1"}
    r = sh("scripts/ec2_sandboxes.sh", "up", "2", env=env)
    assert r.returncode == 0 and "on-demand instead" in r.stdout
    run = [c for c in calls(tmp_path, "aws") if "run-instances" in c]
    assert len(run) == 2 and "MarketType" not in run[1]


def test_allow_serve_probe_down(tmp_path):
    env = {**stubs(tmp_path), "STATE": str(tmp_path / "state")}
    sh("scripts/ec2_sandboxes.sh", "up", env=env)
    (tmp_path / "log").unlink()
    r = sh("scripts/ec2_sandboxes.sh", "allow", "9.9.9.9", env=env)
    assert r.returncode == 0
    assert any("authorize-security-group-ingress" in c and "9.9.9.9/32" in c for c in calls(tmp_path, "aws"))
    assert any(".ssh/aees_tunnel" in c for c in calls(tmp_path, "scp")) and any(".ssh/aees_sync" in c for c in calls(tmp_path, "scp"))
    assert any("~/aees_boxes" in c for c in calls(tmp_path, "ssh"))
    sh("scripts/ec2_sandboxes.sh", "serve", "t3_run", "t3", env=env)
    assert sum("ec2_box.sh serve t3_run t3" in c for c in calls(tmp_path, "ssh")) == 3
    r = sh("scripts/ec2_sandboxes.sh", "probe", "9.9.9.9", env=env)
    assert r.stdout.count("16/16 reverse listeners, H200 stub: aees-stub") == 3
    rows = [json.loads(l) for l in (tmp_path / "probe_rows").read_text().splitlines()]
    assert len(rows) == 3 and all(r["ok"] and r["test"] == "T0" for r in rows)
    r = sh("scripts/ec2_sandboxes.sh", "down", env=env)
    assert any("terminate-instances" in c and "i-1 i-2 i-3" in c for c in calls(tmp_path, "aws"))
    assert not (tmp_path / "state" / "boxes.env").exists()


def test_reap_removes_projects_created_before_the_last_rollout_line(tmp_path):
    """The cutoff comes from the H200's train log; each box gets one awk/compose-down pass with it."""
    ssh = r'''
for a in "$@"; do last="$a"; done
case "$last" in
  *train.log*) echo "2026-11-03 18:36:09" ;;
  *"docker ps -a"*) echo 3 ;;
esac'''
    state = tmp_path / "state"
    state.mkdir()
    (state / "boxes.env").write_text('BOX_IPS="10.0.0.1 10.0.0.2"\n')
    (state / "run.env").write_text("H200=9.9.9.9\nH200_USER=me\n")
    env = {**stubs(tmp_path, {"ssh": ssh}), "STATE": str(state)}
    r = sh("scripts/ec2_sandboxes.sh", "reap", "auxr_x", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "10.0.0.1: removed 3 stale trial projects (created before 2026-11-03 18:36:09)" in r.stdout
    assert "10.0.0.2: removed 3" in r.stdout
    c = calls(tmp_path, "ssh")
    assert any("me@9.9.9.9" in x and "/data/ckpt/auxr_x/train.log" in x and "rollout [0-9]+: " in x for x in c)
    box = [x for x in c if "ubuntu@10.0.0.1" in x and "docker ps -a" in x][0]
    assert "c='2026-11-03 18:36:09'" in box and "__[0-9]{8}-[0-9]{6}__" in box
    assert "compose -p {} down --volumes --remove-orphans" in box


def test_reap_does_nothing_before_the_first_rollout_closes(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "boxes.env").write_text('BOX_IPS="10.0.0.1"\n')
    (state / "run.env").write_text("H200=9.9.9.9\nH200_USER=me\n")
    env = {**stubs(tmp_path), "STATE": str(state)}                           # the log has no rollout line yet
    r = sh("scripts/ec2_sandboxes.sh", "reap", "auxr_x", env=env)
    assert r.returncode == 0 and "removed" not in r.stdout
    assert not [x for x in calls(tmp_path, "ssh") if "ubuntu@" in x]


# ---- box: scripts/ec2_box.sh ----

@pytest.fixture
def box(tmp_path):
    """A copy of the scripts as laid out on a box, a fake HOME, stub setup.sh / sg / rrsync."""
    root = tmp_path / "geodesic" / "aees" / "post_training"
    (root / "scripts").mkdir(parents=True)
    for f in ["ec2_box.sh", "agent_server.sh"]:
        shutil.copy(HERE / "scripts" / f, root / "scripts" / f)
    (root / "setup.sh").write_text('echo "setup DATA=$DATA STEPS=$STEPS TASK_LISTS=$TASK_LISTS" >> "$STUB/log"\n')
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / "tunnel_key.pub").write_text("ssh-ed25519 AAAAtunnel aees-grpo-tunnel_key\n")
    (home / "sync_key.pub").write_text("ssh-ed25519 AAAAsync aees-grpo-sync_key\n")
    env = stubs(tmp_path, {"sg": 'echo "sg-cmd $2" >> "$STUB/log"', "rrsync": "", "docker": "", "nproc": "echo 32",
                           "free": "echo 'Mem: 128'", "curl": '[ -f "$STUB/up" ] && exit 0; touch "$STUB/up"; exit 1'})
    env["HOME"] = str(home)
    return root, home, env


def test_box_prepare_restricts_the_h200_keys(box, tmp_path):
    root, home, env = box
    r = subprocess.run(["bash", str(root / "scripts/ec2_box.sh"), "prepare"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"setup DATA={home}/data STEPS=2 3 5 8 TASK_LISTS=tb21_selected_tasks.txt" in (tmp_path / "log").read_text()
    keys = (home / ".ssh" / "authorized_keys").read_text().splitlines()
    tunnel = next(k for k in keys if "AAAAtunnel" in k)
    assert tunnel.startswith('restrict,port-forwarding,permitopen="127.0.0.1:11000",permitlisten="127.0.0.1:30000",')
    assert len(re.findall(r'permitlisten="127\.0\.0\.1:300\d\d"', tunnel)) == 16 and ', ' not in tunnel.split(" ssh-ed25519")[0]
    sync = next(k for k in keys if "AAAAsync" in k)
    assert sync.startswith(f'restrict,command="rrsync -ro {home}/data/trials" ssh-ed25519')
    subprocess.run(["bash", str(root / "scripts/ec2_box.sh"), "prepare"], env=env, capture_output=True)
    assert len((home / ".ssh" / "authorized_keys").read_text().splitlines()) == 2   # re-run adds nothing


def test_box_serve_t3_sets_the_short_trial_env(box, tmp_path):
    root, home, env = box
    r = subprocess.run(["bash", str(root / "scripts/ec2_box.sh"), "serve", "t3_run", "t3"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    new = [c for c in calls(tmp_path, "tmux") if c.startswith("new")][0]
    assert "HARBOR_AGENT_MAX_ITERATIONS=2 HARBOR_VERIFIER_TIMEOUT_SEC=60" in new
    assert f"DATA={home}/data RUN=t3_run CONCURRENCY=32" in new and "scripts/agent_server.sh" in new
    (tmp_path / "log").unlink()
    (tmp_path / "up").unlink()
    subprocess.run(["bash", str(root / "scripts/ec2_box.sh"), "serve", "real_run"], env=env, capture_output=True)
    new = [c for c in calls(tmp_path, "tmux") if c.startswith("new")][0]
    assert "HARBOR_AGENT_MAX_ITERATIONS" not in new and "RUN=real_run" in new
    assert "HARBOR_VERIFIER_TIMEOUT_SEC=600" in new


# ---- H200: scripts/links.sh ----

def test_links_start_tunnels_sync_and_dispatcher(tmp_path):
    env = stubs(tmp_path, {"curl": "exit 0"})
    (tmp_path / "boxes").write_text("10.0.0.1 10.0.0.2 10.0.0.3\n")
    env.update({"BOXES_FILE": str(tmp_path / "boxes"), "DATA": str(tmp_path / "data"), "RUN": "r1", "HOME": str(tmp_path)})
    r = sh("scripts/links.sh", "start", env=env)
    assert r.returncode == 0, r.stderr
    tm = calls(tmp_path, "tmux")
    tunnels = [c for c in tm if "aees_tunnel" in c]
    assert len(tunnels) == 3
    for i, c in enumerate(tunnels, 1):
        assert f"-L 1100{i}:127.0.0.1:11000" in c and f"ubuntu@10.0.0.{i}" in c and "ExitOnForwardFailure=yes" in c
        assert sorted(re.findall(r"-R 127\.0\.0\.1:(\d+):127\.0\.0\.1:(\d+)", c)) == [(str(p), str(p)) for p in range(30000, 30016)]
    sync = next(c for c in tm if "-n sync" in c)
    assert sync.count("aees_sync") == 6 and f"ubuntu@10.0.0.2:r1/ {tmp_path}/data/trials/r1/" in sync
    assert "--exclude /agent_server.log" in sync and f"{tmp_path}/data/monitor/r1/box_3.log" in sync
    disp = next(c for c in tm if "-n dispatch" in c)
    assert "--backends 127.0.0.1:11001,127.0.0.1:11002,127.0.0.1:11003 --slots 32" in disp
    assert "grpo/task_minutes.json" in disp and f"{tmp_path}/data/trials/r1/dispatch.jsonl" in disp
    assert (tmp_path / "aees_run").read_text().strip() == "r1"


# ---- H200: scripts/preflight.sh and the configs it relies on ----

def grpo_arrays(**env) -> dict:
    out = subprocess.run(["bash", "-c", 'source configs/grpo.sh; for a in SGLANG_SERVER SGLANG_ARGS SAVE_ARGS GRPO_ARGS; do '
                          'eval "printf \'%s\\n\' \\"\\${$a[@]}\\"" | paste -sd"|" - | sed "s/^/$a=/"; done'],
                         env={**os.environ, "SAVE_DIR": "/tmp/s", "MAX_LEN": "49152", **env},
                         cwd=HERE, capture_output=True, text=True, check=True).stdout
    return {k: v.split("|") for k, v in (l.split("=", 1) for l in out.splitlines())}


def test_grpo_config_defaults_and_overrides():
    a = grpo_arrays()
    assert a["SGLANG_SERVER"] == ["--mem-fraction-static", "0.85", "--context-length", "57344", "--dtype", "bfloat16",
                                  "--lora-backend", "triton"]
    assert a["SGLANG_ARGS"][4:] == ["--sglang-mem-fraction-static", "0.85", "--sglang-context-length", "57344",
                                    "--sglang-dtype", "bfloat16", "--sglang-lora-backend", "triton"]
    assert a["SAVE_ARGS"] == ["--save", "/tmp/s", "--save-interval", "10"]
    g = a["GRPO_ARGS"]
    assert g[g.index("--rollout-batch-size") + 1] == "20" and g[g.index("--n-samples-per-prompt") + 1] == "8"
    b = grpo_arrays(SGLANG_EXTRA="--max-running-requests 96", SAVE_INTERVAL="1", ROLLOUT_BATCH="2", N_SAMPLES="4",
                    GLOBAL_BATCH="8")
    assert b["SGLANG_SERVER"][-2:] == ["--max-running-requests", "96"] and "--sglang-max-running-requests" in b["SGLANG_ARGS"]
    assert b["SAVE_ARGS"][-1] == "1"
    g = b["GRPO_ARGS"]
    assert [g[g.index(f) + 1] for f in ["--rollout-batch-size", "--n-samples-per-prompt", "--global-batch-size"]] == ["2", "4", "8"]


def test_run_sh_passes_the_overrides_and_the_jit_cache():
    run = (HERE / "scripts/run.sh").read_text()
    for v in ["SAVE_INTERVAL", "ROLLOUT_BATCH", "N_SAMPLES", "GLOBAL_BATCH", "SGLANG_EXTRA", "PROMPT_DATA", "NUM_ROLLOUT",
              "RESUME_FROM"]:
        assert f"-e {v}" in run
    assert '-v "$DATA/cache:/root/.cache"' in run


def test_t3_groups_survive_the_infra_filter():
    """T3's group size must be >= MIN_VALID, or grpo/infra_filter.py masks every group and stops the run."""
    t3 = (HERE / "scripts/preflight.sh").read_text()
    n, batch, glob = (int(re.search(rf"{k}=(\d+)", t3).group(1)) for k in ["N_SAMPLES", "ROLLOUT_BATCH", "GLOBAL_BATCH"])
    assert n >= MIN_VALID and glob == n * batch
    assert "NUM_ROLLOUT=2" in t3 and "SAVE_INTERVAL=1" in t3


def test_preflight_t0_records_rows(tmp_path):
    docker = r'''
case "$*" in
  *"--help"*) echo "--mem-fraction-static --context-length --dtype --lora-backend --enable-lora --max-lora-rank --lora-target-modules --lora-paths" ;;
  *nvidia-smi*) echo "GPU 0: NVIDIA H200" ;;
esac'''
    curl = r'''
case "$*" in
  *11000/health*) echo '{"status": "ok", "boxes_up": {"a": true, "b": true}, "boxes": {"a": {"running": 0, "free": 32}, "b": {"running": 0, "free": 32}}}' ;;
  *127.0.0.1:30000*) exit 7 ;;
  *) echo ok ;;
esac'''
    env = stubs(tmp_path, {"docker": docker, "curl": curl, "nvidia-smi": "echo 580.178.04",
                           "df": "echo Avail; echo 1400G", "timeout": "", "python3": 'exec /usr/bin/env python3.12 "$@" 2>/dev/null || exec "$(command -v -p python3)" "$@"'})
    data = tmp_path / "data"
    (data / "models" / "Qwen3.6-27B").mkdir(parents=True)
    (data / "models" / "Qwen3.6-27B" / "model-00001.safetensors").write_text("x")
    (tmp_path / "boxes").write_text("10.0.0.1 10.0.0.2\n")
    env.update({"DATA": str(data), "RUN": "r1", "BOXES_FILE": str(tmp_path / "boxes")})
    r = sh("scripts/preflight.sh", "t0", env=env)
    rows = [json.loads(l) for l in (data / "monitor" / "r1" / "preflight.jsonl").read_text().splitlines()]
    by = {x["check"]: x for x in rows}
    assert r.returncode == 0, r.stdout + r.stderr
    assert by["driver >= 580"]["ok"] and by["SGLang knows our flags"]["value"] == "all present"
    assert by["dispatcher: all boxes up"]["value"] == "2 of 2 boxes, 64 slots"
    assert by["torch_dist checkpoint"]["value"].startswith("pending")
    assert {f"box {i} agent server via tunnel (10.0.0.{i})" for i in (1, 2)} <= set(by)
    assert "stub on 127.0.0.1:30000" in r.stdout


def test_preflight_t0_fails_on_an_old_driver_and_missing_flags(tmp_path):
    env = stubs(tmp_path, {"docker": 'echo "--dtype"', "curl": "exit 7", "nvidia-smi": "echo 550.163.01",
                           "df": "echo Avail; echo 100G", "timeout": ""})
    data = tmp_path / "data"
    (data / "models" / "Qwen3.6-27B").mkdir(parents=True)
    (tmp_path / "boxes").write_text("10.0.0.1\n")
    env.update({"DATA": str(data), "RUN": "r1", "BOXES_FILE": str(tmp_path / "boxes")})
    r = sh("scripts/preflight.sh", "t0", env=env)
    rows = {x["check"]: x for x in map(json.loads, (data / "monitor" / "r1" / "preflight.jsonl").read_text().splitlines())}
    assert r.returncode == 1
    assert not rows["driver >= 580"]["ok"] and not rows["disk free >= 400 GB"]["ok"] and not rows["weights"]["ok"]
    assert "--mem-fraction-static" in rows["SGLang knows our flags"]["value"]


def test_every_script_parses():
    for f in sorted((HERE / "scripts").glob("*.sh")):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


def test_kickoff_stops_on_an_old_driver(tmp_path):
    env = stubs(tmp_path, {"ssh": 'for a in "$@"; do last="$a"; done; case "$last" in *driver_version*) echo 550.163.01 ;; *) echo 16 ;; esac'})
    r = sh("scripts/kickoff.sh", "9.9.9.9", env={**env, "STATE": str(tmp_path / "state")})
    assert r.returncode == 1 and "driver < 580" in r.stdout
    assert not calls(tmp_path, "aws")                                  # no EC2 launched for a box we can't use


def test_kickoff_moves_a_previous_runs_state_aside(tmp_path):
    state = tmp_path / "state"; state.mkdir()
    (state / "run.env").write_text("RUN_T3=t3_old\nRUN=grpo_old\nH200=1.1.1.1\nH200_USER=u\n")
    (state / "boxes.env").write_text('REGION=us-east-2\nBOX_IDS="i-1 i-2"\nBOX_IPS="2.2.2.2 3.3.3.3"\n')
    env = stubs(tmp_path, {"ssh": 'for a in "$@"; do last="$a"; done; case "$last" in *driver_version*) echo 550.163.01 ;; *) echo 16 ;; esac',
                           "aws": 'echo 0'})                         # the old boxes are gone
    r = sh("scripts/kickoff.sh", "9.9.9.9", env={**env, "STATE": str(state)})
    assert "moved to state/old_" in r.stdout
    old = next(state.glob("old_*"))
    assert (old / "run.env").exists() and (old / "boxes.env").exists()
    assert "H200=9.9.9.9" in (state / "run.env").read_text()        # fresh run names for the new H200


def test_stub_model_runs_ls_then_completes():
    import importlib.util
    import threading
    import urllib.request
    spec = importlib.util.spec_from_file_location("stub_model", HERE / "scripts/stub_model.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    srv = stub.ThreadingHTTPServer(("127.0.0.1", 0), stub.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert urllib.request.urlopen(url + "/").read() == b"aees-stub"

        def ask(messages):
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({"model": "m", "messages": messages}).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req).read())
            return json.loads(r["choices"][0]["message"]["content"]), r["usage"]["total_tokens"]
        first, usage = ask([{"role": "user", "content": "task"}])
        assert first["commands"][0]["keystrokes"] == "ls\n" and not first["task_complete"] and usage > 0
        later, _ = ask([{"role": "user", "content": "t"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "o"}])
        assert later["task_complete"] and later["commands"] == []
    finally:
        srv.shutdown()


def ckpt_and_grpo(**env) -> tuple[list[str], list[str]]:
    out = subprocess.run(["bash", "-c", 'source configs/common.sh 2>/dev/null; source configs/grpo.sh; '
                          'printf "%s\\n" "${CKPT_ARGS[@]}" | paste -sd"|" -; printf "%s\\n" "${GRPO_ARGS[@]}" | paste -sd"|" -'],
                         env={**os.environ, "SAVE_DIR": "/tmp/s", **env}, cwd=HERE, capture_output=True, text=True)
    ckpt, grpo = out.stdout.splitlines()[-2:]
    return ckpt.split("|"), grpo.split("|")


def test_resume_from_restores_the_optimizer_and_points_at_the_iteration():
    ckpt, grpo = ckpt_and_grpo()
    assert "--no-load-optim" in ckpt and "--finetune" in ckpt and "--lora-adapter-path" not in grpo
    ckpt, grpo = ckpt_and_grpo(RESUME_FROM="/root/ckpt/r/iter_0000007/adapter")
    assert "--no-load-optim" not in ckpt and "--finetune" in ckpt
    assert grpo[grpo.index("--lora-adapter-path") + 1] == "/root/ckpt/r/iter_0000007/adapter"
    r = subprocess.run(["bash", "-c", "source configs/common.sh 2>/dev/null; source configs/grpo.sh"], cwd=HERE,
                       env={**os.environ, "SAVE_DIR": str(HERE / ".pytest_cache/s"), "RESUME_FROM": "/x", "INIT_ADAPTER": "/y"},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "not both" in r.stdout


# ---- H200: scripts/box_check.sh, before launch.sh starts Miles ----

def box_check(tmp_path, shutdown_in_min, ping_ok=(True, True), need=200):
    """Two boxes; each rsync of shutdown_at writes now + shutdown_in_min[i] (None: empty file, 'x': rsync fails)."""
    (tmp_path / "boxes").write_text("10.0.0.1 10.0.0.2\n")
    import time
    at = {f"10.0.0.{i + 1}": ("" if m is None else m if m == "x" else str(int(time.time()) + m * 60))
          for i, m in enumerate(shutdown_in_min)}
    rsync = "src=\"${@: -2:1}\"; dst=\"${@: -1}\"; case \"$src\" in " + " ".join(
        f'*{ip}:*) {"exit 23" if v == "x" else f"echo {v} > $dst"} ;;' for ip, v in at.items()) + " esac"
    curl = "case \"$*\" in " + " ".join(f'*1100{i + 1}/health*) {"exit 0" if ok else "exit 7"} ;;'
                                         for i, ok in enumerate(ping_ok)) + " esac"
    env = stubs(tmp_path, {"rsync": rsync, "curl": curl})
    return sh("scripts/box_check.sh", env={**env, "NEED_MIN": str(need), "BOXES_FILE": str(tmp_path / "boxes"),
                                          "PING_TRIES": "2"})


def test_box_check_passes_when_every_box_answers_and_outlives_the_run(tmp_path):
    r = box_check(tmp_path, (420, None))
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert "box 1 (10.0.0.1): ping ok; shuts down" in lines[0] and lines[0].endswith("need 200 min: OK")
    assert "no shutdown scheduled" in lines[1] and lines[1].endswith("OK")


def test_box_check_fails_on_a_box_shutting_down_before_the_steps_left(tmp_path):   # auxr_1103_0033 step 10
    r = box_check(tmp_path, (420, 60))
    assert r.returncode == 1
    two = r.stdout.splitlines()[1]
    assert two.endswith("FAIL") and ("in 59 min" in two or "in 60 min" in two)


def test_box_check_fails_on_no_ping_or_an_unreadable_shutdown_time(tmp_path):
    r = box_check(tmp_path, (420, "x"), ping_ok=(False, True))
    assert r.returncode == 1
    one, two = r.stdout.splitlines()
    assert "ping no answer" in one and one.endswith("FAIL")
    assert "unreadable" in two and two.endswith("FAIL")


# ---- teardown: scripts/finish.sh (H200) publishes its status; scripts/teardown_watch.sh (laptop) deletes ----

FINISH_AWS = r'''
case "$*" in
  *"s3 ls --recursive"*)                                 # every local file uploaded at its size
    cd "$STUB/data/ckpt/r1" && find . -type f | while read -r f; do
      echo "2026-11-03 00:00:00 $(wc -c < "$f" | tr -d ' ') r1/${f#./}"; done ;;
  *"s3 cp"*"backup_status"*) cp "$3" "$STUB/published" ;;
esac'''


def finish(tmp_path, nebius_ok):
    data = tmp_path / "data"
    (data / "ckpt/r1/iter_0000000/adapter").mkdir(parents=True)
    (data / "ckpt/r1/iter_0000000/adapter/adapter_config.json").write_text("{}")
    (data / "monitor/r1").mkdir(parents=True)
    (data / "trials/r1").mkdir(parents=True)
    (tmp_path / "box_ids").write_text("i-1 i-2\n")
    env = stubs(tmp_path, {"aws": FINISH_AWS, "sudo": '"$@"',
                           "nebius": "exit 0" if nebius_ok else "exit 1"})
    return sh("scripts/finish.sh", env={**env, "DATA": str(data), "RUN": "r1", "DEST": "s3://b",
                                        "BOX_IDS_FILE": str(tmp_path / "box_ids"), "NEBIUS_INSTANCE_ID": "ci-1"})


def test_finish_publishes_backup_ok_even_when_the_h200_cannot_delete_itself(tmp_path):   # auxr_1103_0033
    r = finish(tmp_path, nebius_ok=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "published").read_text().splitlines()[0] == "BACKUP_OK"
    assert any("terminate-instances" in c for c in calls(tmp_path, "aws"))
    assert not any("instance delete" in c for c in calls(tmp_path, "nebius"))
    assert "the laptop's watcher deletes it" in r.stdout


def test_finish_deletes_the_h200_itself_when_nebius_is_authorized(tmp_path):
    r = finish(tmp_path, nebius_ok=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert any("instance delete --id ci-1" in c for c in calls(tmp_path, "nebius"))
    assert (tmp_path / "published").read_text().splitlines()[0] == "BACKUP_OK"


def watch(tmp_path, status):
    aws = f'case "$*" in *"s3 cp"*backup_status*) printf "{status}" ;; *describe-instances*) echo 0 ;; esac'
    env = stubs(tmp_path, {"aws": aws, "nebius": ""})
    return sh("scripts/teardown_watch.sh", "r1", "ci-1", "1", env=env)


def test_teardown_watch_deletes_the_h200_on_backup_ok(tmp_path):
    r = watch(tmp_path, r"BACKUP_OK\n44 files\n")
    assert r.returncode == 0 and "deleting H200 ci-1" in r.stdout
    assert any("--profile aees compute instance delete --id ci-1" in c for c in calls(tmp_path, "nebius"))


def test_teardown_watch_never_deletes_on_an_incomplete_backup(tmp_path):
    r = watch(tmp_path, r"BACKUP_INCOMPLETE\nmissing x\n")
    assert r.returncode == 1 and "NOT deleted" in r.stdout
    assert calls(tmp_path, "nebius") == []
