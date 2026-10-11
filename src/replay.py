"""Replay buffer of per-env streams, sampled as (B, T) sequences. Written by Adam.

Each env is its own continuous stream of records, in the TD3 convention:

    record t = (obs_t, action_t, reward_t, cont_t, is_first_t)

A sampled sequence is T consecutive records from one env's stream. It may cross
an episode boundary; is_first marks it, and the RSSM reset and loss masks handle it.

Deferred (not needed for the first run): online queue, stored latents.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity_per_env: int, num_envs: int, obs_dim: int,
                 act_dim: int, device: torch.device):
        """Preallocate one array per field, shape (capacity_per_env, E, ...).

        obs (C, E, O) float32, action (C, E, A) float32, reward (C, E) float32,
        cont (C, E) float32, is_first (C, E) bool. Plus a write index and a count.
        """
        self.obs = np.zeros((capacity_per_env, num_envs, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity_per_env, num_envs, act_dim), dtype=np.float32)
        self.reward = np.zeros((capacity_per_env, num_envs), dtype=np.float32)
        self.cont = np.zeros((capacity_per_env, num_envs), dtype=np.float32)
        self.is_first = np.zeros((capacity_per_env, num_envs), dtype=bool)
        self.index = 0
        self.count = 0
        self.capacity = capacity_per_env
        self.num_envs = num_envs
        self.device=device

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
            cont: np.ndarray, is_first: np.ndarray) -> None:
        """Write one record per env at the current index.

        obs (E, O), action (E, A), reward (E,), cont (E,), is_first (E,).
        """
        self.obs[self.index] = obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.cont[self.index] = cont
        self.is_first[self.index] = is_first
        self.index = (self.index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def __len__(self) -> int:
        """Number of records stored per env."""
        return self.count

    def sample(self, batch_size: int, seq_len: int) -> dict:
        """B sequences of T consecutive records, each from a random env and start.

        Returns torch tensors on self.device, the same keys as Agent.train_step:
        obs (B, T, O), action (B, T, A), reward (B, T), cont (B, T), is_first (B, T).
        A sequence must not run past the newest record (or across the write index
        once the buffer wraps around).
        """
        env = np.random.randint(0, self.num_envs, size=batch_size)
        k = np.random.randint(0, self.count - seq_len + 1, size=batch_size)

        if self.count < self.capacity:
            oldest = 0
        else:
            oldest = self.index

        start = (oldest + k) % self.capacity
        time_idx = (start[:, None] + np.arange(seq_len)) % self.capacity

        batch_obs = self.obs[time_idx, env[:, None]]
        batch_action = self.action[time_idx, env[:, None]]
        batch_reward = self.reward[time_idx, env[:, None]]
        batch_cont = self.cont[time_idx, env[:, None]]
        batch_is_first = self.is_first[time_idx, env[:, None]]

        return {
            "obs": torch.as_tensor(batch_obs, device=self.device),
            "action": torch.as_tensor(batch_action, device=self.device),
            "reward": torch.as_tensor(batch_reward, device=self.device),
            "cont": torch.as_tensor(batch_cont, device=self.device),
            "is_first": torch.as_tensor(batch_is_first, device=self.device)
        }


