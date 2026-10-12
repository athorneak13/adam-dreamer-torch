"""Time Agent.train_step on random data, and estimate the full run's training time.

uv run python scripts/benchmark_train_step.py --config configs/dmc_walker_walk.yaml --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import Agent  # noqa: E402
from src.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dmc_walker_walk.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--obs-dim", type=int, default=24)
    parser.add_argument("--act-dim", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.device)
    agent = Agent(cfg, args.obs_dim, args.act_dim).to(device)
    B, T = cfg.train.batch_size, cfg.train.seq_len
    batch = dict(obs=torch.randn(B, T, args.obs_dim, device=device),
                 action=torch.rand(B, T, args.act_dim, device=device) * 2 - 1,
                 reward=torch.randn(B, T, device=device),
                 cont=torch.ones(B, T, device=device),
                 is_first=torch.zeros(B, T, dtype=torch.bool, device=device))

    for _ in range(3):                                   # warm-up (CUDA init, allocator)
        agent.train_step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.steps):
        agent.train_step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.time() - start) / args.steps

    n_train = (cfg.run.total_steps - cfg.run.warmup_steps) // cfg.run.train_every
    print(f"device {device} | {per_step:.3f} s per train_step (B={B}, T={T}, H={cfg.train.horizon})")
    print(f"full run: {n_train:,} train steps -> ~{n_train * per_step / 3600:.1f} h of training "
          f"(plus env and policy time)")


if __name__ == "__main__":
    main()
