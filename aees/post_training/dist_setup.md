# Distributed setup: 1× H200 (trainer + SGLang) + 3× EC2 (Harbor sandboxes)

What works, measured on real machines, so setup isn't redone or re-debugged. **Update after every setup.**
Tests: `integration_tests.md`. Problems: `integration_issues.md`. Why this layout: PLAN.md §5, §5b.

## Layout

```
laptop ── ssh ──> H200 (Nebius VM, 16 vCPU, 200 GB RAM, 1.5 TB disk)
                    Miles container: Megatron trainer + SGLang + session servers (127.0.0.1:30000-30015)
                    grpo/dispatch.py on 127.0.0.1:11000   <- Miles posts every trial here
                    monitor/server.py on 127.0.0.1:8770   <- tracker (laptop: scripts/tunnel.sh)
                    scripts/tunnels.sh: one ssh per EC2 box, from the H200:
                       -L 1100{1,2,3} -> box:127.0.0.1:11000     (dispatcher -> agent server)
                       -R 30000-30015 -> H200:127.0.0.1:same     (terminus-2 on the box -> session server)
laptop ── aws ──> 3× EC2 m7a.8xlarge (us-east-2, 32 vCPU, 128 GB): Docker + Harbor agent server, 32 trials each
```

Miles hands each trial a session URL `http://127.0.0.1:300xx/v1` (Ray runs at `--node-ip-address 127.0.0.1`).
The reverse tunnel makes that same address work on the box, so Miles needs no change.

## Scripts (one command each)

| Where | Script | Does |
|---|---|---|
| laptop | `scripts/kickoff.sh <h200-ip>` | probe → EC2 boxes (background) → code to H200 → `h200_setup.sh` → allow + T3 agent servers → links + tracker |
| laptop | `scripts/ec2_sandboxes.sh up / allow / serve / probe / status / down` | the boxes (spot m7a.8xlarge, self-terminate after 15 h) |
| laptop | `scripts/rehearse.sh` | EC2-only rehearsal with a stand-in H200 and `scripts/stub_model.py` (integration_tests.md, R) |
| box | `scripts/ec2_box.sh prepare / serve <run> [t3]` | `setup.sh` steps 2 3 5 8 (training tasks only), restricted H200 keys, agent server (32 at once) |
| H200 | `scripts/h200_setup.sh` | `setup.sh` 1–4, then 6 ∥ 7, then 9; times → `$DATA/monitor/setup_times.txt` |
| H200 | `scripts/links.sh start / stop / status` | tunnels, trial sync, dispatcher in tmux "links" |
| H200 | `scripts/preflight.sh t0 / t2 / t3` | integration tests → `$DATA/monitor/$RUN/preflight.jsonl` (tracker card) |

Keys: the laptop makes `state/tunnel_key` and `state/sync_key` and copies them to the H200 (`~/.ssh/aees_tunnel`,
`~/.ssh/aees_sync`). On each box they're restricted: the tunnel key can only forward 127.0.0.1:11000 and listen on
127.0.0.1:30000–30015, and the sync key can only run `rrsync -ro ~/data/trials`.

## Timing target: ≤ 10 min from IP to "ready for tests"

| Step | Runs on | Last measured | Notes |
|---|---|---|---|
| Probe (driver, disk, docker) | H200 | 2 s | `nvidia-smi`, `df`, `which docker` |
| Weights, `Qwen/Qwen3.6-27B` only (52 GB) | H200 | ~3.5 min (2026-10-25: 52 GB + 29 GB FP8 in 4 min) | `hf download` in its own tmux; FP8 not needed |
| Miles image pull | H200 | ~4 min (with steps 1–4) | parallel with weights |
| setup.sh steps 1–4 | H200 | < 1 min | Nebius image already has Docker and nvidia-container-toolkit 1.20.1 |
| Miles source checkout | H200 | < 1 min | Harbor and task images are **not** needed on the H200 anymore |
| HF → torch_dist conversion | H200 GPU | **not yet measured** | starts once image + weights are in; T0 runs meanwhile |
| 3 EC2 boxes: launch + setup_ec2 + agent server + 20 images | EC2 | ~6 min for 1 box, 8 images (2026-10-25) | the 3 boxes run in parallel; start at the same moment as the H200 steps |
| Tunnels + dispatcher | H200 | – | seconds |

**Measured 2026-10-27** (H200 204.12.188.130, Ubuntu 24.04, driver 580.173.02; 3 × m7a.8xlarge on-demand):
steps 1–4 33 s · weights 194 s · Miles image + source 411 s (ready for T0) · torch_dist 669 s (**11 min** total);
EC2 boxes ready (20 images each) in **3.5 min**.

Critical path: weights or image (~4 min), then conversion. **If the IP is the same box as 2026-10-25
(204.12.163.30), driver 580, weights and image are already there: ~3 min.**

## H200: what works

1. **Driver ≥ 580 is required** (the Miles image is CUDA 13.0). Ask for an image that ships 580. Nebius' default
   Ubuntu 22.04 image came with **550.163.01**. Upgrading it in place cost ~45 min and a console Stop/Start:
   - `apt-get purge nvidia-fabricmanager-550 libnvidia-nscq-550`, then
     `apt-get install nvidia-headless-580-server-open nvidia-utils-580-server` (4 min);
   - **a warm `sudo reboot` leaves the GPU stuck**: dmesg `WPR2 already up`, nvidia-smi "No devices were found";
   - an in-VM PCI reset (`echo 1 > /sys/bus/pci/devices/0000:8d:00.0/reset`) brings nvidia-smi back, **but CUDA
     then fails with error 802** (`nvidia-smi -q` → Fabric State "In Progress");
   - **only a console Stop/Start fixes it** (the user does it in the Nebius console). Don't PCI-reset.
   - Fabric manager in the VM is useless (no NVSwitch visible); `nvidia-fabricmanager-580` fails to start. Leave it
     disabled.
   - So, if the driver is 550: install 580, then **ask the user for a console Stop/Start** (not a reboot).
2. The host runs `nvidia-dcgm` (dcgmi 3.3.9). Don't start a DCGM container (exits 255); use the host `dcgmi`.
3. Disk: `/` is 1.5 TB; `/data` doesn't exist → `sudo mkdir -p /data && sudo chown ubuntu /data`.
4. `scripts/kickoff.sh` copies the code. `data/grpo_{train,val}.jsonl` come from the laptop, since the H200 has
   no task folders for `grpo/split_tasks.py`. It then runs `scripts/h200_setup.sh`.
   `scripts/run.sh` and `preflight.sh` mount a shared JIT cache (`$DATA/cache:/root/.cache`) into every Miles
   container. SGLang's first start compiles kernels once (60–220 s), and T2 warms the cache for Miles.
5. `pkill -f` patterns must never match your own ssh command line (killed the session on 2026-10-25).

## EC2: what works

- **Laptop credentials:** AWS CLI default profile (IAM user `agent_aees`), region us-east-2, key pair `aees-ec2`
  (`~/.ssh/aees-ec2.pem`). Nothing is copied to the instances. The profile can't read AWS Budgets, so the user
  checks headroom in the console before launch.
- **AMI:** latest Canonical Ubuntu 24.04 amd64 (owner 099720109477). Root volume 200 GB gp3.
- **Type:** m7a.8xlarge. **Spot is denied for `agent_aees`** by an organization service control policy
  (2026-10-27: `UnauthorizedOperation ... explicit deny in a service control policy`), so `ec2_sandboxes.sh` falls
  back to on-demand (~$1.85/h per box). If spot is ever allowed: m7a (5–10% interruption band); avoid m7i/c7i/m6i
  spot (>20%).
- **Concurrency:** ≤ ~50 trials per box (150 simultaneous starts on one box gave 41% tool-install timeouts).
  We run 32 per box.
- **Self-terminate:** user-data `shutdown -h +TTL_MIN` and `--instance-initiated-shutdown-behavior terminate`.
- **Security group `aees-proto`:** SSH from the laptop's IP, plus the H200's IP for the tunnels.
- **Setup:** rsync `aees/harbor` and the few post_training files (create the remote folders first: rsync makes
  only the last folder). Then `setup.sh` steps 2 3 5 8 with `DATA=~/data TASK_LISTS=tb21_selected_tasks.txt`:
  - Docker and its config;
  - Harbor `harbor-miles-v0.20.0` @ ac63e51 plus `grpo/harbor_parser.patch` and `uv sync`;
  - the TB 2.1 download (layout `~/data/tasks/terminal-bench-2-1/<task>`);
  - the 20 training images. `harbor/ec2/setup_ec2.sh` isn't needed.
- **Agent server:** `scripts/agent_server.sh` unchanged, with `DATA=~/data CONCURRENCY=32` (it already binds
  127.0.0.1 and sets `DOCKER_PRUNE_INTERVAL_SEC=1e9`; the server's prune deletes the pre-pulled images).
- **setup.sh on Ubuntu 24.04:** step 3 now restarts Docker itself. On the H200, step 4 used to apply `daemon.json`;
  EC2 skips step 4.
- zsh on the laptop: write `${EC2_IP}`, not `$IP:...` (zsh reads `:g` as a modifier).

## Miles on one node with remote agents (found in T3)

- Miles places its session servers, router and SGLang on the VM's address (10.96.0.14), even with Ray at
  127.0.0.1. Agents on the boxes can only use 127.0.0.1 (the tunnels), so `configs/grpo.sh` sets
  `--session-server-ip 0.0.0.0 --session-server-external-host 127.0.0.1`. Miles' readiness check uses the placed
  address, so binding 127.0.0.1 alone would fail.
- Those ports answered from the internet (checked from the laptop: 30000 and 31000 → HTTP 404). `h200_setup.sh`
  drops new inbound TCP on the public interface except ssh. Local traffic uses `lo`, and the tunnels are outbound ssh.
- Files written by containers are root-owned (`/data/ckpt/<run>`, adapters with mode 600): use `sudo rm`.
- Start long-lived tmux sessions only after setup.sh has added the user to the docker group.

## Teardown

Final save → copy `$DATA/ckpt/$RUN` and trials to the laptop → terminate the EC2 boxes → the user stops the H200.
