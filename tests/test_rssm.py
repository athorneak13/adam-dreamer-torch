"""RSSM tests. Skipped until written; Adam decides who writes each one."""

import pytest

todo = pytest.mark.skip(reason="TODO")


@todo
def test_initial_state_shapes(cfg):
    """h (B, D), z (B, K*C), logits (B, K, C)."""


@todo
def test_z_is_one_hot_forward(cfg):
    """Each of the K rows of z sums to 1 and has exactly one nonzero entry."""


@todo
def test_straight_through_gradient(cfg):
    """d(sum(w * z))/d(probs) equals w: gradient passes as if z were the probs."""


@todo
def test_unimix_floor(cfg):
    """Every class probability is >= unimix / C."""


@todo
def test_observe_shapes(cfg):
    """observe() over T returns posts and priors with a (B, T, ...) leading shape."""


@todo
def test_imagine_shapes(cfg):
    """imagine() from N = B*T starts returns (N, H+1, ...) states and actions."""


@todo
def test_is_first_resets_state(cfg):
    """With is_first set at step t, outputs from step t onward do not depend on anything before t."""
