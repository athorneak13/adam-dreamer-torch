"""E parallel DeepMind Control tasks (proprio), with the same interface as GymEnvs.

Task loading, action repeat (summing rewards) and the discount-based terminal
flag follow the DeepMindControl wrapper in the NM512 dreamerv3-torch scaffold
(MIT License, Copyright (c) 2023 NM512). Adapted here to:
  - E envs stepped in lockstep,
  - a flat proprio vector (observation spec entries concatenated in spec order),
  - the replay convention used by GymEnvs (see src/envs/gym_env.py):
        record t = (obs_t, action_t, reward_t, cont_t, is_first_t)
    cont_t = TimeStep.discount (DMC time limits keep discount 1.0, so cont stays 1),
    auto-reset with next_is_first = True, the final observation is dropped.

Actions come in as [-1, 1] and are rescaled to the action spec bounds
(already [-1, 1] for the standard suite).
"""

import numpy as np
from dm_control import suite


def _flatten(observation) -> np.ndarray:
    return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1)
                           for v in observation.values()])


class DMCEnvs:
    def __init__(self, name: str, num_envs: int, seed: int = 0, action_repeat: int = 1):
        """name: '<domain>_<task>', e.g. 'walker_walk', 'cartpole_swingup', 'cup_catch'."""
        domain, task = name.split("_", 1)
        if domain == "cup":  # the only domain with a multi-word name
            domain = "ball_in_cup"
        self.envs = [suite.load(domain, task, task_kwargs={"random": seed + i})
                     for i in range(num_envs)]
        self.num_envs = num_envs
        self.action_repeat = action_repeat

        spec = self.envs[0].action_spec()
        self.act_low = spec.minimum.astype(np.float32)
        self.act_high = spec.maximum.astype(np.float32)
        self.act_dim = spec.shape[0]
        self.obs_dim = sum(int(np.prod(v.shape)) for v in self.envs[0].observation_spec().values())

        self._obs = None
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset all envs. Returns obs (E, O) and is_first (E,) = True."""
        self._obs = np.stack([_flatten(env.reset().observation) for env in self.envs])
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return self._obs.copy(), np.ones(self.num_envs, dtype=bool)

    def step(self, action: np.ndarray) -> dict:
        """Same return dict as GymEnvs.step. episode_length counts agent steps."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        env_action = self.act_low + (action + 1.0) / 2.0 * (self.act_high - self.act_low)

        reward = np.zeros(self.num_envs, dtype=np.float32)
        cont = np.ones(self.num_envs, dtype=np.float32)
        next_obs = np.zeros_like(self._obs)
        next_is_first = np.zeros(self.num_envs, dtype=bool)
        ep_return = np.full(self.num_envs, np.nan, dtype=np.float32)
        ep_length = np.full(self.num_envs, np.nan, dtype=np.float32)

        for i, env in enumerate(self.envs):
            r = 0.0
            for _ in range(self.action_repeat):
                ts = env.step(env_action[i])
                r += ts.reward or 0.0
                if ts.last():
                    break
            reward[i] = r
            cont[i] = float(ts.discount) if ts.discount is not None else 1.0
            self._returns[i] += r
            self._lengths[i] += 1
            obs = _flatten(ts.observation)
            if ts.last():
                ep_return[i], ep_length[i] = self._returns[i], self._lengths[i]
                self._returns[i], self._lengths[i] = 0.0, 0
                obs = _flatten(env.reset().observation)
                next_is_first[i] = True
            next_obs[i] = obs

        self._obs = next_obs
        return dict(reward=reward, cont=cont, next_obs=next_obs.copy(),
                    next_is_first=next_is_first, episode_return=ep_return,
                    episode_length=ep_length)

    def close(self):
        for env in self.envs:
            env.close()
