"""Tests for the lambda-return, replay target and return scale."""

import torch

from src.returns import ReturnScale, lambda_return, replay_return

GAMMA, LAM = 0.9, 0.5


# ---------- lambda_return ----------

def test_lambda_return_two_step_hand_case():
    # r = [1, 2], c = [1, 1], v = [0, 3, 4], gamma 0.9, lam 0.5
    # R_2 = 4, R_1 = 2 + 0.9 * (0.5*4 + 0.5*4) = 5.6, R_0 = 1 + 0.9 * (0.5*3 + 0.5*5.6) = 4.87
    R = lambda_return(torch.tensor([[1., 2.]]), torch.ones(1, 2),
                      torch.tensor([[0., 3., 4.]]), GAMMA, LAM)
    assert torch.allclose(R, torch.tensor([[4.87, 5.6]]))


def test_lambda_return_three_step_hand_case():
    # r = [1, 0, 2], c = 1, v = [0, 1, 2, 3], gamma 0.5, lam 0.5
    # R_3 = 3; R_2 = 2 + 0.5*(0.5*3 + 0.5*3) = 3.5
    # R_1 = 0 + 0.5*(0.5*2 + 0.5*3.5) = 1.375; R_0 = 1 + 0.5*(0.5*1 + 0.5*1.375) = 1.59375
    R = lambda_return(torch.tensor([[1., 0., 2.]]), torch.ones(1, 3),
                      torch.tensor([[0., 1., 2., 3.]]), 0.5, 0.5)
    assert torch.allclose(R, torch.tensor([[1.59375, 1.375, 3.5]]))


def test_lambda_return_shape():
    R = lambda_return(torch.randn(4, 5), torch.ones(4, 5), torch.randn(4, 6), GAMMA, LAM)
    assert R.shape == (4, 5)


def test_lambda_zero_is_one_step_td():
    r, c, v = torch.randn(4, 5), torch.rand(4, 5), torch.randn(4, 6)
    assert torch.allclose(lambda_return(r, c, v, GAMMA, 0.0), r + GAMMA * c * v[:, 1:])


def test_lambda_one_is_full_return_plus_bootstrap():
    r, c, v = torch.randn(4, 5), torch.ones(4, 5), torch.randn(4, 6)
    full = sum(GAMMA ** k * r[:, k] for k in range(5)) + GAMMA ** 5 * v[:, 5]
    assert torch.allclose(lambda_return(r, c, v, GAMMA, 1.0)[:, 0], full, atol=1e-5)


def test_continue_zero_cuts_the_future():
    r, v = torch.randn(4, 5), torch.randn(4, 6)
    c = torch.ones(4, 5)
    c[:, 2] = 0.0
    R = lambda_return(r, c, v, GAMMA, LAM)
    assert torch.allclose(R[:, 2], r[:, 2])


# ---------- replay_return ----------

def test_replay_return_hand_case():
    # r = [1, 2, 3], c = 1, img = [10, 20, 30], gamma 0.5, lam 0.5
    # R_2 = 30 (start), R_1 = 2 + 0.5*(0.5*30 + 0.5*30) = 17, R_0 = 1 + 0.5*(0.5*20 + 0.5*17) = 10.25
    r, c = torch.tensor([[1., 2., 3.]]), torch.ones(1, 3)
    img = torch.tensor([[10., 20., 30.]])
    R = replay_return(r, c, img, torch.zeros(1, 3, dtype=torch.bool), 0.5, 0.5)
    assert torch.allclose(R, torch.tensor([[10.25, 17.]]))


def test_replay_return_restarts_at_episode_boundary():
    # New episode at step 2: R_1 restarts at img[1] = 20, R_0 = 1 + 0.5*(0.5*20 + 0.5*20) = 11
    r, c = torch.tensor([[1., 2., 3.]]), torch.ones(1, 3)
    img = torch.tensor([[10., 20., 30.]])
    is_first = torch.tensor([[False, False, True]])
    R = replay_return(r, c, img, is_first, 0.5, 0.5)
    assert torch.allclose(R, torch.tensor([[11., 20.]]))


def test_replay_return_shape_and_row_independence():
    B, T = 2, 8
    r, c, img = torch.randn(B, T), torch.ones(B, T), torch.randn(B, T)
    no_reset = torch.zeros(B, T, dtype=torch.bool)
    reset = no_reset.clone()
    reset[1, 4] = True
    R0 = replay_return(r, c, img, no_reset, 0.997, 0.95)
    R1 = replay_return(r, c, img, reset, 0.997, 0.95)
    assert R0.shape == (B, T - 1)
    assert torch.allclose(R0[0], R1[0])          # row 0 unaffected
    assert torch.allclose(R0[1, 4:], R1[1, 4:])  # after the boundary: unchanged
    assert not torch.allclose(R0[1, :4], R1[1, :4])


# ---------- ReturnScale ----------

def spread_20_returns():
    """101 values with 5th percentile exactly -2 and 95th exactly 18."""
    return torch.tensor([-2.] * 5 + [(i / 90) * 20 - 2 for i in range(91)] + [18.] * 5)


def test_return_scale_ema_example():
    # CLAUDE.md: spread 20 with previous S = 19 gives 19.01
    s = ReturnScale()
    s.spread = 19.0
    s.update(spread_20_returns())
    assert abs(s.spread - 19.01) < 1e-6
    assert abs(s.scale() - 19.01) < 1e-6


def test_return_scale_never_divides_by_less_than_one():
    s = ReturnScale()
    assert s.scale() == 1.0          # before any update
    s.spread = 0.3
    assert s.scale() == 1.0


def test_return_scale_first_update_sets_spread():
    s = ReturnScale()
    s.update(spread_20_returns())
    assert abs(s.spread - 20.0) < 1e-5
    assert isinstance(s.spread, float)
