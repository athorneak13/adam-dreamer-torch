"""Tests for the actor, critic and slow-critic update."""

import copy
import math

import pytest
import torch

from src.agent import Actor, Critic, update_slow_critic


@pytest.fixture
def feat_dim(cfg):
    return cfg.model.recurrent_dim + cfg.model.K * cfg.model.C


@pytest.fixture
def actor(cfg, feat_dim):
    return Actor(feat_dim, cfg.model.mlp_units, cfg.model.mlp_layers, cfg.act_dim)


@pytest.fixture
def critic(cfg, feat_dim):
    return Critic(feat_dim, cfg.model.mlp_units, cfg.model.mlp_layers, cfg.model.bins)


def test_actor_shapes(cfg, actor, feat_dim):
    dist = actor(torch.randn(16, feat_dim))
    a = dist.sample()
    assert a.shape == (16, cfg.act_dim)
    assert dist.log_prob(a).sum(-1).shape == (16,)


def test_actor_initial_policy(actor, feat_dim):
    # outscale 0.01 and zero bias: mean ~ 0, std ~ 0.9 * sigmoid(2) + 0.1 for every state
    dist = actor(torch.randn(64, feat_dim))
    assert dist.mean.abs().max() < 0.05
    expected_std = 0.9 * torch.sigmoid(torch.tensor(2.0)) + 0.1
    assert torch.allclose(dist.stddev, expected_std.expand_as(dist.stddev), atol=0.01)


def test_actor_gradients_through_log_prob(actor, feat_dim):
    dist = actor(torch.randn(4, feat_dim))
    (-dist.log_prob(dist.sample()).sum()).backward()
    assert all(p.grad is not None for p in actor.parameters())


def test_critic_shapes_and_zero_init(cfg, critic, feat_dim):
    feat = torch.randn(2, 8, feat_dim)
    assert critic(feat).shape == (2, 8, cfg.model.bins)
    assert (critic.predict(feat) == 0).all()
    assert "bins" in dict(critic.named_buffers())


def test_slow_critic_ema(critic):
    slow = copy.deepcopy(critic).requires_grad_(False)
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(1.0)
    old = [p.clone() for p in slow.parameters()]
    update_slow_critic(critic, slow, rate=0.02)
    for p_slow, p_old, p in zip(slow.parameters(), old, critic.parameters()):
        assert torch.allclose(p_slow, 0.98 * p_old + 0.02 * p)
    assert not any(p.requires_grad for p in slow.parameters())


def test_slow_critic_stays_equal_when_critic_unchanged(critic):
    slow = copy.deepcopy(critic)
    update_slow_critic(critic, slow)
    for p_slow, p in zip(slow.parameters(), critic.parameters()):
        assert torch.allclose(p_slow, p)


# ---------- Agent ----------

from src.agent import Agent  # noqa: E402


@pytest.fixture
def agent(cfg):
    torch.manual_seed(0)
    return Agent(cfg, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim)


def make_batch(cfg, B=None, T=None):
    B = B or cfg.train.batch_size
    T = T or cfg.train.seq_len
    return dict(obs=torch.randn(B, T, cfg.obs_dim), action=torch.randn(B, T, cfg.act_dim),
                reward=torch.randn(B, T) * 5, cont=torch.ones(B, T),
                is_first=torch.zeros(B, T, dtype=torch.bool))


def test_agent_parameter_count(agent):
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    assert trainable == 484_677 + 43_652 + 59_967
    assert not any(p.requires_grad for p in agent.slow_critic.parameters())


def test_train_step_metrics_and_initial_values(cfg, agent):
    m = agent.train_step(make_batch(cfg))
    expected = {"rec", "rew", "con", "dyn", "rep", "value", "repval", "policy",
                "ret", "adv_mag", "ret_scale", "entropy"}
    assert expected <= set(m)
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())
    assert m["ret"] == 0.0 and m["adv_mag"] == 0.0     # zero-initialized reward head and critic


def test_train_step_any_batch_shape(cfg, agent):
    agent.train_step(make_batch(cfg, B=3, T=6))


def test_train_step_learns_and_updates_slow_critic(cfg, agent):
    batch = make_batch(cfg)
    first = agent.train_step(batch)
    for _ in range(100):
        m = agent.train_step(batch)
    assert m["rec"] < first["rec"]
    assert any(not torch.equal(a, b) for a, b in
               zip(agent.slow_critic.parameters(), agent.critic.parameters()))
    assert all(torch.isfinite(p).all() for p in agent.parameters())


def test_policy_round_trip(cfg, agent):
    E = 3
    action = torch.zeros(E, cfg.act_dim)
    state, first = None, torch.ones(E, dtype=torch.bool)
    for _ in range(4):
        action, state = agent.policy(torch.randn(E, cfg.obs_dim), state, action, first)
        first = torch.zeros(E, dtype=torch.bool)
    assert action.shape == (E, cfg.act_dim)
    assert state["h"].shape == (E, cfg.model.recurrent_dim)
    assert not action.requires_grad
    mean_action, _ = agent.policy(torch.randn(E, cfg.obs_dim), state, action, first, sample=False)
    assert mean_action.abs().max() < 1.0
