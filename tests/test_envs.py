"""Tests for the Gymnasium env wrapper (Pendulum-v1)."""

import numpy as np
import pytest

from src.envs.gym_env import GymEnvs


@pytest.fixture
def envs():
    e = GymEnvs("Pendulum-v1", num_envs=3, seed=0)
    yield e
    e.close()


def test_spaces(envs):
    assert envs.obs_dim == 3 and envs.act_dim == 1
    assert np.allclose(envs.act_low, -2.0) and np.allclose(envs.act_high, 2.0)


def test_reset(envs):
    obs, is_first = envs.reset()
    assert obs.shape == (3, 3) and obs.dtype == np.float32
    assert is_first.all()


def test_step_shapes(envs):
    envs.reset()
    out = envs.step(np.zeros((3, 1)))
    assert out["reward"].shape == (3,) and out["cont"].shape == (3,)
    assert out["next_obs"].shape == (3, 3)
    assert not out["next_is_first"].any()
    assert np.isnan(out["episode_return"]).all()


def test_action_rescaling(envs):
    # Pendulum reward depends on torque: -(... + 0.001 * u^2). Compare u = +1 (-> 2.0) vs 0.
    envs.reset()
    r_zero = envs.step(np.zeros((3, 1)))["reward"]
    envs.reset()
    r_full = envs.step(np.ones((3, 1)))["reward"]
    assert np.allclose(r_zero - r_full, 0.001 * 2.0 ** 2, atol=1e-5)


def test_time_limit_is_truncation_not_termination(envs):
    envs.reset()
    for t in range(200):
        out = envs.step(np.random.uniform(-1, 1, (3, 1)))
        assert (out["cont"] == 1.0).all()          # Pendulum never terminates
        if t < 199:
            assert not out["next_is_first"].any()
    assert out["next_is_first"].all()              # auto-reset after 200 steps
    assert (out["episode_length"] == 200).all()
    assert np.isfinite(out["episode_return"]).all()


# ---------- DMC ----------

from types import SimpleNamespace  # noqa: E402

from src.envs import make_envs  # noqa: E402
from src.envs.dmc_env import DMCEnvs  # noqa: E402


@pytest.fixture
def walker():
    e = DMCEnvs("walker_walk", num_envs=2, seed=0, action_repeat=2)
    yield e
    e.close()


def test_dmc_walker_spaces_and_reset(walker):
    assert walker.obs_dim == 24 and walker.act_dim == 6     # 14 orientations + 1 height + 9 velocity
    obs, is_first = walker.reset()
    assert obs.shape == (2, 24) and obs.dtype == np.float32 and is_first.all()


def test_dmc_episode_length_with_action_repeat(walker):
    walker.reset()
    for t in range(500):                                      # 1000 env steps / repeat 2
        out = walker.step(np.random.uniform(-1, 1, (2, 6)))
        assert (out["cont"] == 1.0).all()                     # time limit, not termination
        assert out["next_is_first"].all() == (t == 499)
    assert (out["episode_length"] == 500).all()
    assert ((out["episode_return"] >= 0) & (out["episode_return"] <= 1000)).all()


def test_dmc_reward_sums_over_action_repeat():
    one = DMCEnvs("walker_walk", 1, seed=3, action_repeat=1)
    two = DMCEnvs("walker_walk", 1, seed=3, action_repeat=2)
    one.reset(); two.reset()
    a = np.zeros((1, 6))
    r1 = one.step(a)["reward"] + one.step(a)["reward"]
    r2 = two.step(a)["reward"]
    assert np.allclose(r1, r2, atol=1e-6)


def test_make_envs_dispatch():
    cfg = lambda name: SimpleNamespace(seed=0, env=SimpleNamespace(name=name, num_envs=1, action_repeat=2))
    assert isinstance(make_envs(cfg("dmc:walker_walk")), DMCEnvs)
    assert isinstance(make_envs(cfg("gym:Pendulum-v1")), GymEnvs)
    assert isinstance(make_envs(cfg("Pendulum-v1")), GymEnvs)
