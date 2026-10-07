"""Tests for the reward and continue heads."""

import pytest
import torch

from src.models.heads import ContinueHead, RewardHead


@pytest.fixture
def feat_dim(cfg):
    return cfg.model.recurrent_dim + cfg.model.K * cfg.model.C


@pytest.fixture
def feat(cfg, feat_dim):
    return torch.randn(cfg.train.batch_size, cfg.train.seq_len, feat_dim)


@pytest.fixture
def reward_head(cfg, feat_dim):
    return RewardHead(feat_dim, cfg.model.mlp_units, 1, cfg.model.bins)


@pytest.fixture
def continue_head(cfg, feat_dim):
    return ContinueHead(feat_dim, cfg.model.mlp_units, 1)


def test_reward_shapes(cfg, reward_head, feat):
    B, T = cfg.train.batch_size, cfg.train.seq_len
    assert reward_head(feat).shape == (B, T, cfg.model.bins)
    assert reward_head.predict(feat).shape == (B, T)


def test_reward_starts_at_exactly_zero(reward_head, feat):
    assert (reward_head.predict(feat) == 0).all()


def test_reward_bins_are_a_buffer(reward_head):
    assert "bins" in dict(reward_head.named_buffers())
    assert not any("bins" in n for n, _ in reward_head.named_parameters())
    assert reward_head.to(torch.float64).bins.dtype == torch.float64


def test_reward_zeroed_output_still_learns(reward_head, feat):
    logits = reward_head(feat)
    loss = -torch.log_softmax(logits, -1)[..., 130].mean()
    loss.backward()
    assert reward_head.net[-1].weight.grad.abs().sum() > 0


def test_continue_shapes(cfg, continue_head, feat):
    B, T = cfg.train.batch_size, cfg.train.seq_len
    assert continue_head(feat).shape == (B, T)
    assert continue_head.predict(feat).shape == (B, T)
    assert continue_head(feat[0, :1]).shape == (1,)  # batch of 1 keeps its dim


def test_continue_forward_is_logit_predict_is_prob(continue_head, feat):
    logits = continue_head(feat)
    probs = continue_head.predict(feat)
    assert torch.allclose(probs, torch.sigmoid(logits))
    assert ((probs > 0) & (probs < 1)).all()
