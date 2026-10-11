"""E parallel Gymnasium envs with continuous actions, in the replay convention.

Each call to step() returns everything for one replay record per env:

    record t = (obs_t, action_t, reward_t, cont_t, is_first_t)

- reward_t, cont_t result from taking action_t in obs_t (TD3 convention).
- cont_t = 1 - terminated. A time limit (truncated) is NOT a termination: cont stays 1.
- When an episode ends, the env resets automatically. The returned next obs is then
  the first obs of the new episode, with is_first = True. The final observation of
  the old episode is not returned (the replay never stores it).

Actions come in as [-1, 1] (the actor's range) and are rescaled to the env's bounds.
Arrays are numpy float32; convert to torch in the training loop.
"""

import gymnasium as gym
import numpy as np


class GymEnvs:
    def __init__(self, name: str, num_envs: int, seed: int = 0):
        self.envs = [gym.make(name) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed

        space = self.envs[0].action_space
        assert isinstance(space, gym.spaces.Box), "continuous actions only"
        self.act_low = space.low.astype(np.float32)
        self.act_high = space.high.astype(np.float32)
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.act_dim = space.shape[0]

        self._obs = None
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self._lengths = np.zeros(num_envs, dtype=np.int64)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset all envs. Returns obs (E, O) and is_first (E,) = True."""
        obs = [env.reset(seed=self.seed + i)[0] for i, env in enumerate(self.envs)]
        self._obs = np.stack(obs).astype(np.float32)
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return self._obs.copy(), np.ones(self.num_envs, dtype=bool)

    def step(self, action: np.ndarray) -> dict:
        """Take one step in every env.

        action: (E, A) in [-1, 1].
        Returns a dict of numpy arrays:
            reward (E,), cont (E,)     for the action just taken
            next_obs (E, O)            the obs to act from next
            next_is_first (E,)         True where an episode just ended and reset
            episode_return, episode_length: (E,) totals for envs whose episode
                                       just ended, NaN elsewhere (for logging)
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        env_action = self.act_low + (action + 1.0) / 2.0 * (self.act_high - self.act_low)

        reward = np.zeros(self.num_envs, dtype=np.float32)
        cont = np.ones(self.num_envs, dtype=np.float32)
        next_obs = np.zeros_like(self._obs)
        next_is_first = np.zeros(self.num_envs, dtype=bool)
        ep_return = np.full(self.num_envs, np.nan, dtype=np.float32)
        ep_length = np.full(self.num_envs, np.nan, dtype=np.float32)

        for i, env in enumerate(self.envs):
            obs, r, terminated, truncated, info = env.step(env_action[i])
            reward[i] = r
            cont[i] = 0.0 if terminated else 1.0
            self._returns[i] += r
            self._lengths[i] += 1
            if terminated or truncated:
                ep_return[i], ep_length[i] = self._returns[i], self._lengths[i]
                self._returns[i], self._lengths[i] = 0.0, 0
                obs, _ = env.reset()
                next_is_first[i] = True
            next_obs[i] = obs

        self._obs = next_obs
        return dict(reward=reward, cont=cont, next_obs=next_obs.copy(),
                    next_is_first=next_is_first, episode_return=ep_return,
                    episode_length=ep_length)

    def close(self):
        for env in self.envs:
            env.close()
