#!/usr/bin/env bash
# On the EC2 box (run by proto/ec2_up.sh): Docker, uv, Harbor 0.23.0 via aees/harbor/ec2/setup_ec2.sh, then
# the TB 2.1 task definitions and the images of the prototype's tasks (proto/ec2_tasks.txt).
set -euo pipefail
mkdir -p ~/geodesic/.git        # aees/harbor was copied, not cloned (private repo); this stops setup_ec2.sh cloning
bash ~/geodesic/aees/harbor/ec2/setup_ec2.sh > ~/proto/setup_ec2.log 2>&1 || { tail -30 ~/proto/setup_ec2.log; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
sg docker -c '
  set -e
  [ -d ~/tb21 ] || harbor datasets download terminal-bench/terminal-bench-2-1 -o ~/tb21 >/dev/null
  python3 ~/geodesic/aees/harbor/scripts/task_images.py ~/tb21 ~/proto/ec2_tasks.txt | sort -u | xargs -r -P 8 -n 1 docker pull -q
  docker images --format "{{.Repository}}:{{.Tag}}" | head -20'
echo "prepared: $(nproc) vCPU, $(free -g | awk "/Mem/{print \$2}") GB RAM, disk free $(df -h ~ | tail -1 | awk '{print $4}')"
