"""Training loop entry point. Written by Adam.

Usage: uv run python -m src.train --config configs/pendulum.yaml

One cycle: collect env steps with the current policy into replay, then train.
Replay records follow the TD3 convention:
    record t = (obs_t, action_t, reward_t, cont_t, is_first_t)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import apply_overrides, load_config
from src.envs.gym_env import GymEnvs
from src.agent import Agent
from src.replay import ReplayBuffer
from src.envs import make_envs
from src.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from src.logger import Logger



def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """numpy array from the envs -> torch tensor on device (dtype kept)."""
    return torch.as_tensor(x, device=device)


def train(cfg) -> None:
    """Run training.

    Setup:  envs (GymEnvs), agent (Agent), buffer (ReplayBuffer)
            obs, is_first = envs.reset(); state = None; prev_action = zeros (E, A)

    Each env step:
      1. act:     random in [-1, 1] during warm-up, else agent.policy(...)
      2. step:    out = envs.step(action)
      3. store:   buffer.add(obs, action, out["reward"], out["cont"], is_first)
      4. advance: obs, is_first = out["next_obs"], out["next_is_first"]; prev_action = action
      5. train:   after warm-up, every N env steps: agent.train_step(buffer.sample(B, T))
      6. log:     episode returns when episodes end; train metrics now and then

    Open decisions: warm-up length, train every N steps, store clipped or raw action.
    """
    device = torch.device(cfg.device)
    envs = make_envs(cfg)#GymEnvs(cfg.env.name, cfg.env.num_envs, cfg.seed)
    agent = Agent(cfg=cfg, obs_dim=envs.obs_dim, act_dim=envs.act_dim).to(device)
    buffer = ReplayBuffer(capacity_per_env=int(cfg.replay.capacity)//cfg.env.num_envs, 
                          num_envs=cfg.env.num_envs, 
                          obs_dim=envs.obs_dim, 
                          act_dim=envs.act_dim, 
                          device=device)

    # resume: restore networks, optimizers, S, RNG and the step counter (replay starts empty)
    start_step = 0
    ckpt_path = Path(cfg.checkpoint.dir) / "latest.pt"
    if cfg.checkpoint.resume and latest_checkpoint(cfg.checkpoint.dir) is not None:
        start_step = load_checkpoint(ckpt_path, agent, device)
        print(f"resumed from {ckpt_path} at step {start_step}")
    logger = Logger(cfg)

    obs, is_first = envs.reset()
    state = None
    prev_action = np.zeros((cfg.env.num_envs, envs.act_dim), dtype=np.float32)

    returns_log = []
    returns_file = f"returns_{cfg.logging.run_name or f'seed{cfg.seed}'}.csv"
    if start_step > 0 and Path(returns_file).exists():
        returns_log = [tuple(r) for r in np.loadtxt(returns_file, delimiter=",", ndmin=2)]
    metrics = None

    try:
        step = start_step
        updates_due = 0
        for it in range(start_step // cfg.env.num_envs, cfg.run.total_steps // cfg.env.num_envs):
            step = it * cfg.env.num_envs

            if step < cfg.run.warmup_steps and start_step == 0:   # random warm-up only on a fresh run
                action = np.random.uniform(-1.0, 1.0, size=(cfg.env.num_envs, envs.act_dim)).astype(np.float32)
            else:
                action_t, state = agent.policy(obs=to_torch(obs, device),
                                               state=state,
                                               prev_action=to_torch(prev_action, device),
                                               is_first=to_torch(is_first, device))

                action = action_t.cpu().numpy()
                # train only once replay holds warmup_steps records (after a resume it starts empty)
                if len(buffer) * cfg.env.num_envs >= cfg.run.warmup_steps:
                    updates_due += cfg.env.num_envs
                
            action = np.clip(action, -1.0, 1.0)
            out = envs.step(action)
            buffer.add(obs, action, out["reward"], out["cont"], is_first)
            obs, is_first = out["next_obs"], out["next_is_first"]

            # episode returns, right after envs.step / buffer.add
            done = ~np.isnan(out["episode_return"])
            for ret, length in zip(out["episode_return"][done], out["episode_length"][done]):
                logger.episode(step, float(ret), float(length))
                returns_log.append((step, float(ret)))

            prev_action = action
        

            while updates_due >= cfg.run.train_every:#if step >= cfg.run.warmup_steps and step % cfg.run.train_every == 0:
                batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len)
                metrics = agent.train_step(batch)
                updates_due -= cfg.run.train_every

            # training metrics, every log_every steps (once training has started)
            if metrics is not None and step % cfg.run.log_every == 0:
                logger.metrics(step, metrics)

            # checkpoint: the saved step is the next one to run, so resuming repeats nothing
            if step > 0 and step % cfg.checkpoint.every == 0:
                save_checkpoint(ckpt_path, agent, step + cfg.env.num_envs)
    finally:
        # runs even on Ctrl-C, so the returns so far are always saved
        np.savetxt(returns_file, np.array(returns_log), delimiter=",", header="step,return")
        logger.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <checkpoint.dir>/latest.pt if it exists")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                        help="--wandb / --no-wandb overrides the config's logging.wandb")
    parser.add_argument("--run-name", default=None,
                        help="names the W&B run, checkpoint folder and returns CSV")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.horizon=5 (repeatable)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.run_name is not None:
        cfg.logging.run_name = args.run_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.wandb is not None:
        cfg.logging.wandb = args.wandb
    if cfg.logging.run_name is None:
        task = cfg.env.name.split(":")[-1]
        cfg.logging.run_name = f"{task}_seed{cfg.seed}"
    cfg.logging.run_id = cfg.logging.run_name          # same W&B run when resuming
    cfg.checkpoint.dir = f"{cfg.checkpoint.dir}/{cfg.logging.run_name}"
    cfg.checkpoint.resume = args.resume
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
