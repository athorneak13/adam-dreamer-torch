"""Training loop entry point. Written by Adam.

Usage: uv run python -m src.train --config configs/pendulum.yaml

One cycle: collect env steps with the current policy into replay, then train.
Replay records follow the TD3 convention:
    record t = (obs_t, action_t, reward_t, cont_t, is_first_t)
"""

import argparse

import numpy as np
import torch

from src.config import load_config
from src.envs.gym_env import GymEnvs
from src.agent import Agent
from src.replay import ReplayBuffer



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
    envs = GymEnvs(cfg.env.name, cfg.env.num_envs, cfg.seed)
    agent = Agent(cfg=cfg, obs_dim=envs.obs_dim, act_dim=envs.act_dim).to(device)
    buffer = ReplayBuffer(capacity_per_env=int(cfg.replay.capacity)//cfg.env.num_envs, 
                          num_envs=cfg.env.num_envs, 
                          obs_dim=envs.obs_dim, 
                          act_dim=envs.act_dim, 
                          device=device)

    obs, is_first = envs.reset()
    state = None
    prev_action = np.zeros((cfg.env.num_envs, envs.act_dim), dtype=np.float32)

    returns_log = []
    metrics = None

    try:
        step = 0
        for it in range(cfg.run.total_steps // cfg.env.num_envs):
            step = it * cfg.env.num_envs

            if step < cfg.run.warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(cfg.env.num_envs, envs.act_dim)).astype(np.float32)
            else:
                action_t, state = agent.policy(obs=to_torch(obs, device),
                                               state=state,
                                               prev_action=to_torch(prev_action, device),
                                               is_first=to_torch(is_first, device))

                action = action_t.cpu().numpy()

            action = np.clip(action, -1.0, 1.0)
            out = envs.step(action)
            buffer.add(obs, action, out["reward"], out["cont"], is_first)
            obs, is_first = out["next_obs"], out["next_is_first"]

            # episode returns, right after envs.step / buffer.add
            done = ~np.isnan(out["episode_return"])
            for ret in out["episode_return"][done]:
                print(f"step {step:>7d} | episode return {ret:8.1f}")
                returns_log.append((step, float(ret)))

            prev_action = action
        

            if step >= cfg.run.warmup_steps and step % cfg.run.train_every == 0:
                batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len)
                metrics = agent.train_step(batch)

            # training metrics, every log_every steps (once training has started)
            if metrics is not None and step % cfg.run.log_every == 0:
                print(f"step {step:>7d} | rec {metrics['rec']:.3f}  rew {metrics['rew']:.3f}  "
                    f"value {metrics['value']:.3f}  ret {metrics['ret']:.2f}  "
                    f"entropy {metrics['entropy']:.2f}  scale {metrics['ret_scale']:.2f}")
    finally:
        # runs even on Ctrl-C, so the returns so far are always saved
        np.savetxt(f"returns_seed{cfg.seed}.csv", np.array(returns_log), delimiter=",",
                   header="step,return")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_cpu.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
