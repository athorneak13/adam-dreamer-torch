"""Tests for LaProp, AGC and the LR warmup."""

import pytest
import torch

from src.optim import LaProp, agc_clip_, warmup_lr


# ---------- warmup ----------

def test_warmup_ramp():
    assert [warmup_lr(1.0, s, 1000) for s in (1, 501, 1001, 5000)] == [0.0, 0.5, 1.0, 1.0]


def test_no_warmup():
    assert warmup_lr(4e-5, 1, 0) == 4e-5


# ---------- AGC ----------

def test_agc_clips_to_limit():
    p = torch.full((4,), 0.5)                 # ||p|| = 1 -> limit 0.3
    g = torch.full((4,), 5.0)                 # ||g|| = 10
    agc_clip_(g, p, 0.3, 1e-3)
    assert torch.isclose(g.norm(), torch.tensor(0.3))


def test_agc_leaves_small_gradients_and_keeps_direction():
    p = torch.full((4,), 0.5)
    g = torch.full((4,), 0.05)
    before = g.clone()
    agc_clip_(g, p, 0.3, 1e-3)
    assert torch.allclose(g, before)

    d = torch.tensor([3.0, 4.0])
    agc_clip_(d, torch.tensor([1.0, 0.0]), 0.3, 1e-3)
    assert torch.allclose(d / d.norm(), torch.tensor([0.6, 0.8]))


def test_agc_pmin_floor():
    g = torch.ones(3)
    agc_clip_(g, torch.zeros(3), 0.3, 1e-3)
    assert torch.isclose(g.norm(), torch.tensor(3e-4))


# ---------- LaProp ----------

def test_first_step_moves_by_lr_times_sign():
    # With both bias corrections and eps ~ 0, the first update is lr * sign(g).
    for g in (5.0, -0.003):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        opt = LaProp([p], lr=0.1, warmup=0, agc=1e9)
        p.grad = torch.tensor([g])
        opt.step()
        assert torch.allclose(p.detach(), torch.tensor([2.0 - 0.1 * (1 if g > 0 else -1)]))


def test_warmup_first_update_is_zero():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    opt = LaProp([p], lr=0.1, warmup=1000)
    p.grad = torch.tensor([5.0])
    opt.step()
    assert p.item() == 2.0
    assert opt.state[p]["step"] == 1


def reference_laprop(p0, grads, lr, b1, b2, eps, agc, pmin, warmup):
    """Plain re-statement of the official chain, one tensor, for comparison."""
    p, v, m = p0.clone(), torch.zeros_like(p0), torch.zeros_like(p0)
    for t, g in enumerate(grads, start=1):
        g = g.clone()
        limit = agc * max(pmin, p.norm().item())
        g = g / max(1.0, g.norm().item() / limit)
        v = b2 * v + (1 - b2) * g * g
        u = g / (torch.sqrt(v / (1 - b2 ** t)) + eps)
        m = b1 * m + (1 - b1) * u
        lr_t = lr if warmup == 0 else lr * min(1.0, (t - 1) / warmup)
        p = p - lr_t * m / (1 - b1 ** t)
    return p


@pytest.mark.parametrize("warmup", [0, 5])
def test_matches_reference_over_many_steps(warmup):
    torch.manual_seed(0)
    p0 = torch.randn(3, 4)
    grads = [torch.randn(3, 4) * s for s in (1.0, 0.1, 10.0, 0.5, 3.0, 0.01, 1.0, 2.0)]
    p = torch.nn.Parameter(p0.clone())
    opt = LaProp([p], lr=0.01, betas=(0.9, 0.999), eps=1e-20, agc=0.3, pmin=1e-3, warmup=warmup)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = reference_laprop(p0, grads, 0.01, 0.9, 0.999, 1e-20, 0.3, 1e-3, warmup)
    assert torch.allclose(p.detach(), ref, atol=1e-6)


def test_skips_parameters_without_grad_and_trains_a_model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))
    unused = torch.nn.Parameter(torch.ones(2))
    opt = LaProp(list(model.parameters()) + [unused], lr=3e-3, warmup=10)
    x = torch.randn(64, 4)
    y = torch.sin(x @ torch.tensor([[1.0], [-2.0], [0.5], [1.5]]))   # learnable target
    first = None
    for _ in range(500):
        loss = ((model(x) - y) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    assert torch.equal(unused.detach(), torch.ones(2)) and unused not in opt.state
