#!/usr/bin/env bash
# fi_jit_setup.sh - build a "cold" venv for Phase 4 experiment 2b (JIT compilation).
# Same torch and flashinfer-python versions as the serving venv, but WITHOUT the prebuilt
# flashinfer-cubin / flashinfer-jit-cache wheels, and with an empty, private JIT cache,
# so every attention variant is compiled from its template on first use.
#
# Usage:  bash fi_jit_setup.sh
# Then:   source ~/fi-cold/bin/activate && source ~/fi-cold/jit_env.sh
#         python fi_bench.py jit
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
MAIN_VENV="${MAIN_VENV:-$WORK_DIR/.venv}"
COLD_VENV="${COLD_VENV:-$HOME/fi-cold}"
export PATH="$HOME/.local/bin:$PATH"

TORCH=$("$MAIN_VENV/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
FI=$("$MAIN_VENV/bin/python" -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
echo "Matching serving venv: torch $TORCH, flashinfer-python $FI"

[[ -d "$COLD_VENV" ]] || uv venv --python 3.12 "$COLD_VENV"
VIRTUAL_ENV="$COLD_VENV" uv pip install "torch==$TORCH" --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV="$COLD_VENV" uv pip install "flashinfer-python==$FI" numpy ninja

# Private JIT workspace so earlier compiles elsewhere can't be reused.
cat > "$COLD_VENV/jit_env.sh" <<EOF
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=\$CUDA_HOME/bin:\$PATH
export FLASHINFER_WORKSPACE_BASE=$COLD_VENV/jit-workspace
export FI_OUT=${FI_OUT:-$WORK_DIR/results/fi}
EOF
rm -rf "$COLD_VENV/jit-workspace"
VIRTUAL_ENV="$COLD_VENV" uv pip list 2>/dev/null | grep -i flashinfer
echo
echo "Cold venv ready. Next:"
echo "  source $COLD_VENV/bin/activate && source $COLD_VENV/jit_env.sh"
echo "  python fi_bench.py jit   (from the folder holding the scripts)"
echo "To repeat a cold run later: rm -rf $COLD_VENV/jit-workspace"
