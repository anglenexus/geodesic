#!/usr/bin/env bash
# One-time setup of an x86_64 Ubuntu 24.04 EC2 instance for ~100 concurrent Harbor trials.
# Run as the default `ubuntu` user (uses sudo). Safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/anglenexus/geodesic/main/aees/harbor/ec2/setup_ec2.sh | bash
#   # or, after cloning the repo:  bash aees/harbor/ec2/setup_ec2.sh
#
# Afterwards: log out and back in (docker group), add the OpenAI key, `docker login`,
# then run ec2/prepare_tasks.sh. See ec2/EC2.md.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/anglenexus/geodesic.git}
REPO_DIR=${REPO_DIR:-$HOME/geodesic}
HARBOR_VERSION=${HARBOR_VERSION:-0.23.0}

[ "$(uname -m)" = "x86_64" ] || { echo "expected an x86_64 instance (task images are amd64-only)"; exit 1; }

echo "== 1/6 system packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git jq python3 tmux

echo "== 2/6 docker engine"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"

echo "== 3/6 docker daemon config for many concurrent trials"
# Harbor creates one compose network per trial. Docker's default pools hand out /16 subnets and
# run out after ~30 networks, which breaks at 100 concurrent trials. Use /24 subnets from a
# dedicated range (1024 networks). If your VPC uses 10.200.0.0/14, pick another base.
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "default-address-pools": [{"base": "10.200.0.0/14", "size": 24}],
  "log-driver": "json-file",
  "log-opts": {"max-size": "20m", "max-file": "2"},
  "default-ulimits": {"nofile": {"Name": "nofile", "Hard": 65536, "Soft": 65536}}
}
JSON
# Many containers each want inotify watches/instances; raise the per-user limits.
sudo tee /etc/sysctl.d/90-harbor.conf >/dev/null <<'CONF'
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
fs.file-max = 2097152
CONF
sudo sysctl --system >/dev/null
sudo systemctl restart docker

echo "== 4/6 uv + harbor $HARBOR_VERSION"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv tool install --force "harbor==$HARBOR_VERSION"
harbor --version

echo "== 5/6 repo + aees_harbor environment"
[ -d "$REPO_DIR/.git" ] || git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR/aees/harbor"
uv sync --python 3.12
uv run --with pytest pytest -q tests/unit

echo "== 6/6 secrets placeholder"
mkdir -p "$HOME/.config/harbor" "$HOME/jobs"
if [ ! -f "$HOME/.config/harbor/.env" ]; then
  printf 'OPENAI_API_KEY=REPLACE_ME\n' > "$HOME/.config/harbor/.env"
  chmod 600 "$HOME/.config/harbor/.env"
fi

cat <<'NEXT'

Setup done. Next:
  1. Log out and back in (so your user is in the docker group), then: docker info
  2. Put the OpenAI key in ~/.config/harbor/.env   (OPENAI_API_KEY=sk-...)
  3. docker login                                  (avoids Docker Hub pull rate limits)
  4. bash ~/geodesic/aees/harbor/ec2/prepare_tasks.sh
NEXT
