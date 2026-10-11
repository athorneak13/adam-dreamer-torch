"""Tests for the replay buffer.

Each record stores a counter as its observation (record n has obs = n), so a
correctly sampled sequence holds consecutive counter values.
"""

import numpy as np
import pytest
import torch

from src.replay import ReplayBuffer

E, O, A = 3, 2, 1


def fill(buf, n):
    """Add n records per env; record i of env e has obs = [i, e]."""
    for i in range(n):
        obs = np.stack([[i, e] for e in range(E)]).astype(np.float32)
        buf.add(obs, np.full((E, A), i, np.float32), np.full(E, i, np.float32),
                np.ones(E, np.float32), np.array([i % 7 == 0] * E))


def make(capacity):
    return ReplayBuffer(capacity, E, O, A, torch.device("cpu"))


def check_consecutive(batch, newest):
    counters = batch["obs"][..., 0]                       # (B, T)
    diffs = counters[:, 1:] - counters[:, :-1]
    assert (diffs == 1).all(), f"non-consecutive sequence:\n{counters}"
    assert (counters <= newest).all()


def test_len_and_capacity():
    buf = make(8)
    fill(buf, 5)
    assert len(buf) == 5
    fill(buf, 10)
    assert len(buf) == 8


def test_sample_shapes_and_dtypes():
    buf = make(50)
    fill(buf, 20)
    b = buf.sample(4, 6)
    assert b["obs"].shape == (4, 6, O) and b["action"].shape == (4, 6, A)
    for key in ("reward", "cont", "is_first"):
        assert b[key].shape == (4, 6)
    assert b["obs"].dtype == torch.float32 and b["is_first"].dtype == torch.bool


def test_sequences_are_consecutive_before_wrap():
    buf = make(50)
    fill(buf, 20)
    for _ in range(20):
        check_consecutive(buf.sample(16, 6), newest=19)


def test_sequences_are_consecutive_after_wrap():
    # 23 records in 8 slots: only records 15..22 remain, and the write index is mid-array
    buf = make(8)
    fill(buf, 23)
    for _ in range(50):
        b = buf.sample(16, 5)
        check_consecutive(b, newest=22)
        assert (b["obs"][..., 0] >= 15).all()         # overwritten records never appear


def test_every_valid_start_and_env_is_reachable():
    buf = make(8)
    fill(buf, 23)
    starts, envs = set(), set()
    for _ in range(200):
        b = buf.sample(16, 5)
        starts.update(b["obs"][:, 0, 0].int().tolist())
        envs.update(b["obs"][:, 0, 1].int().tolist())
    assert starts == {15, 16, 17, 18}                 # oldest record up to newest - T + 1
    assert envs == set(range(E))


def test_one_env_per_sequence():
    buf = make(50)
    fill(buf, 20)
    env_ids = buf.sample(16, 6)["obs"][..., 1]        # (B, T)
    assert (env_ids == env_ids[:, :1]).all()


def test_fields_stay_aligned():
    buf = make(8)
    fill(buf, 23)
    b = buf.sample(16, 5)
    assert torch.equal(b["action"][..., 0], b["obs"][..., 0])
    assert torch.equal(b["reward"], b["obs"][..., 0])
    assert torch.equal(b["is_first"], b["obs"][..., 0] % 7 == 0)


def test_batch_feeds_train_step(cfg):
    from src.agent import Agent
    buf = ReplayBuffer(100, E, cfg.obs_dim, cfg.act_dim, torch.device("cpu"))
    for i in range(30):
        buf.add(np.random.randn(E, cfg.obs_dim).astype(np.float32),
                np.random.uniform(-1, 1, (E, cfg.act_dim)).astype(np.float32),
                np.random.randn(E).astype(np.float32), np.ones(E, np.float32),
                np.array([i == 0] * E))
    agent = Agent(cfg, cfg.obs_dim, cfg.act_dim)
    metrics = agent.train_step(buf.sample(cfg.train.batch_size, cfg.train.seq_len))
    assert np.isfinite(list(metrics.values())).all()
