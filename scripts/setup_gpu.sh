#!/usr/bin/env bash
# Set up a fresh Linux GPU VM (Ubuntu + NVIDIA driver, e.g. Lambda/RunPod/GCP/AWS images).
#
#   1. Copy this script to the VM, or clone the repo first and run it from the repo root.
#   2. bash scripts/setup_gpu.sh
#   3. wandb login          (paste your API key once)
#   4. Launch (see the end of this script's output).
#
# The repo is private: clone with an SSH key added to GitHub, or with `gh auth login`.
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:athorneak13/adam-dreamer-torch.git}"
REPO_DIR="${REPO_DIR:-$HOME/adam-dreamer-torch}"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only

echo "== dependencies (Python 3.12, torch with CUDA, dm_control)"
uv sync

echo "== checks"
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
uv run python -c "from src.envs.dmc_env import DMCEnvs; e = DMCEnvs('walker_walk', 2, action_repeat=2); o, _ = e.reset(); print('dm_control walker ok, obs', o.shape)"
uv run pytest -q

echo "== train_step timing on this GPU"
uv run python scripts/benchmark_train_step.py --config configs/dmc_walker_walk.yaml --device cuda

cat <<'EOF'

== Setup done. Next:
  wandb login
  mkdir -p logs
  nohup uv run python -m src.train --config configs/dmc_walker_walk.yaml --seed 0 \
      > logs/walker_seed0.log 2>&1 &
  tail -f logs/walker_seed0.log

If the VM restarts or the run stops, continue from the last checkpoint with the same
command plus --resume.
EOF
