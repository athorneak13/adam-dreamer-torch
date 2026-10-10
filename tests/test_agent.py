"""Tests for the actor, critic and slow-critic update."""

import copy

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
