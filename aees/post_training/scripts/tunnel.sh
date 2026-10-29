#!/usr/bin/env bash
# On the laptop: forward the box's training tracker (monitor/server.py, 127.0.0.1:8770 there) to
# http://localhost:8770 here. Ctrl-C closes it. Extra ssh options go after the host (e.g. -i key.pem).
#
#   bash scripts/tunnel.sh ubuntu@<box> [-i ~/.ssh/key.pem]
set -euo pipefail
HOST=${1:?usage: tunnel.sh user@box [ssh options]}; shift
PORT=${PORT:-8770}
echo "tracker: http://localhost:$PORT   (Ctrl-C to close)"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "$PORT:127.0.0.1:$PORT" "$@" "$HOST"
