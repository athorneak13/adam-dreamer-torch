#!/usr/bin/env bash
# Imagination-horizon ablation on walker-walk: H in {5, 15, 30} x seeds {0, 1, 2} = 9 runs,
# all on one GPU Pod. Run from the repo root (e.g. /workspace/adam-dreamer-torch).
#
#   bash scripts/launch_horizon_sweep.sh            # check resources, then launch
#   bash scripts/launch_horizon_sweep.sh --resume   # restart after an interruption
#
# Each run: logs/walker_h<H>_seed<S>.log, checkpoints/walker_h<H>_seed<S>/latest.pt,
# returns_walker_h<H>_seed<S>.csv, and a W&B run named walker_h<H>_seed<S>.
set -euo pipefail

HORIZONS="${HORIZONS:-5 15 30}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-300000}"
EXTRA="${1:-}"

echo "== resources"
echo "vCPUs: $(nproc)   (want >= 12 for 9 runs; each run uses about one core)"
free -g | awk 'NR==2 {print "RAM: " $2 " GB total, " $7 " GB available"}'
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "commit: $(git log -1 --oneline)"
if ! git diff --quiet; then echo "WARNING: uncommitted changes in the repo"; fi

mkdir -p logs
for H in $HORIZONS; do
  for S in $SEEDS; do
    NAME="walker_h${H}_seed${S}"
    echo "launching $NAME"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 nohup uv run python -m src.train \
      --config configs/dmc_walker_walk.yaml --seed "$S" --run-name "$NAME" \
      --set "train.horizon=$H" --set "run.total_steps=$STEPS" $EXTRA \
      > "logs/$NAME.log" 2>&1 &
    sleep 2
  done
done

cat <<MSG

Launched. In about 10-15 minutes, check speed and memory:
  nvidia-smi                                   # GPU memory per process, utilization
  grep -h "rec " logs/walker_h*_seed0.log | tail -6
Compare the time between metrics lines with the earlier ~24 min per 5,120 steps.
Stop all runs cleanly (so the returns CSVs get written):
  pkill -INT -f "src.train"
MSG
