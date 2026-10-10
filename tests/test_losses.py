"""Tests for the world-model losses."""

import math

import pytest
import torch

from src.losses import (categorical_kl, continue_loss, dyn_rep_losses, recon_loss,
                        reward_loss, world_model_loss)
from src.models.distributions import make_bins, symlog, twohot_encode
from src.models.world_model import WorldModel

SCALES = {"rec": 1.0, "rew": 1.0, "con": 1.0, "dyn": 1.0, "rep": 0.1}


@pytest.fixture
def bins():
    return make_bins()


# ---------- per-step terms ----------

def test_recon_hand_value():
    obs = torch.tensor([[[1000., 0.5, -3.]]])
    recon = symlog(obs) + torch.tensor([0.1, -0.2, 0.3])
    loss = recon_loss(recon, obs)
    assert loss.shape == (1, 1)
    assert torch.allclose(loss, torch.tensor([[0.14]]))


def test_categorical_kl_worked_example():
    # Study notes: q = [0.7, 0.2, 0.1], p = [0.5, 0.3, 0.2] -> 0.085 nats
    q = torch.tensor([0.7, 0.2, 0.1]).view(1, 1, 1, 3)
    p = torch.tensor([0.5, 0.3, 0.2]).view(1, 1, 1, 3)
    assert torch.allclose(categorical_kl(q, p), torch.tensor([[0.0851]]), atol=1e-4)
    # A second, identical latent adds nothing
    same = torch.tensor([0.1, 0.1, 0.8]).view(1, 1, 1, 3)
    kl2 = categorical_kl(torch.cat([q, same], 2), torch.cat([p, same], 2))
    assert torch.allclose(kl2, torch.tensor([[0.0851]]), atol=1e-4)


def test_categorical_kl_shape_and_identity():
    q = torch.softmax(torch.randn(2, 8, 4, 8), -1)
    p = torch.softmax(torch.randn(2, 8, 4, 8), -1)
    kl = categorical_kl(q, p)
    assert kl.shape == (2, 8)
    assert (kl >= 0).all()
    assert torch.allclose(categorical_kl(q, q), torch.zeros(2, 8), atol=1e-6)


def test_dyn_rep_values_and_stop_gradients():
    q_logits = torch.randn(2, 8, 4, 8, requires_grad=True)
    p_logits = torch.randn(2, 8, 4, 8, requires_grad=True)
    dyn, rep = dyn_rep_losses(torch.softmax(q_logits, -1), torch.softmax(p_logits, -1))
    assert dyn.shape == rep.shape == (2, 8)
    assert torch.allclose(dyn, rep)
    assert (dyn >= 1.0).all()

    dyn.sum().backward(retain_graph=True)
    assert q_logits.grad is None          # dyn trains the prior only
    assert p_logits.grad is not None

    p_logits.grad = None
    rep.sum().backward()
    assert q_logits.grad is not None      # rep trains the posterior only
    assert p_logits.grad is None


def test_free_bits_zero_gradient_below_one_nat():
    # CLAUDE.md: a KL below 1 nat gives exactly zero gradient
    q = torch.softmax(0.01 * torch.randn(1, 1, 4, 8), -1)
    p_logits = torch.zeros(1, 1, 4, 8, requires_grad=True)
    dyn, _ = dyn_rep_losses(q, torch.softmax(p_logits, -1))
    assert dyn.item() == 1.0
    dyn.sum().backward()
    assert (p_logits.grad == 0).all()


def test_reward_loss_uniform_is_log_bins(bins):
    reward = torch.tensor([[0., 100., -5.], [1e4, 0.3, -2e6]])
    loss = reward_loss(torch.zeros(2, 3, 255), reward, bins)
    assert torch.allclose(loss, torch.full((2, 3), math.log(255)))


def test_reward_loss_minimum_and_bounded_gradient(bins):
    reward = torch.tensor([[100.]])
    label = twohot_encode(reward, bins)
    loss = reward_loss(torch.log(label + 1e-12), reward, bins)
    entropy = -(label[label > 0] * label[label > 0].log()).sum()
    assert torch.allclose(loss, entropy.view(1, 1), atol=1e-5)

    logits = torch.zeros(2, 3, 255, requires_grad=True)
    reward_loss(logits, torch.randn(2, 3) * 1e4, bins).sum().backward()
    assert logits.grad.abs().max() <= 1.0


def test_continue_loss_values():
    loss = continue_loss(torch.zeros(2, 3), torch.ones(2, 3))
    assert torch.allclose(loss, torch.full((2, 3), math.log(2)))
    loss = continue_loss(torch.tensor([[2., 2.]]), torch.tensor([[1., 0.]]))
    assert torch.allclose(loss, torch.tensor([[0.1269, 2.1269]]), atol=1e-4)
    assert torch.allclose(continue_loss(torch.tensor([[-200.]]), torch.tensor([[1.]])),
                          torch.tensor([[200.]]))


# ---------- total ----------

@pytest.fixture
def wm_batch(cfg):
    torch.manual_seed(0)
    B, T = cfg.train.batch_size, cfg.train.seq_len
    wm = WorldModel(cfg, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim)
    batch = dict(obs=torch.randn(B, T, cfg.obs_dim), action=torch.randn(B, T, cfg.act_dim),
                 reward=torch.randn(B, T) * 10, cont=torch.ones(B, T),
                 is_first=torch.zeros(B, T, dtype=torch.bool))
    return wm, batch


def run_loss(wm, batch, is_first=None, out=None):
    is_first = batch["is_first"] if is_first is None else is_first
    out = wm(batch["obs"], batch["action"], is_first) if out is None else out
    loss, metrics = world_model_loss(out, batch["obs"], batch["reward"], batch["cont"],
                                     is_first, wm.reward.bins, SCALES)
    return loss, metrics, out


def test_total_is_weighted_sum_and_backprops(wm_batch):
    wm, batch = wm_batch
    loss, m, _ = run_loss(wm, batch)
    expected = sum(SCALES[k] * m[k] for k in SCALES)
    assert abs(loss.item() - expected) < 1e-4
    assert all(isinstance(v, float) for v in m.values())
    loss.backward()
    assert all(p.grad is not None for p in wm.parameters())


def test_initial_values(wm_batch):
    wm, batch = wm_batch
    _, m, _ = run_loss(wm, batch)
    assert abs(m["rew"] - math.log(255)) < 1e-4   # zero-initialized reward head
    assert m["dyn"] == 1.0 and m["rep"] == 1.0    # near-uniform prior and posterior


def test_offset_and_boundary_mask(wm_batch):
    wm, batch = wm_batch
    _, m, out = run_loss(wm, batch)
    per_pair = continue_loss(out["cont_logits"][:, 1:], batch["cont"][:, :-1])
    assert abs(m["con"] - per_pair.mean().item()) < 1e-6   # no boundaries: plain mean

    reset = batch["is_first"].clone()
    reset[0, 3] = True
    _, m2, _ = run_loss(wm, batch, is_first=reset, out=out)
    keep = torch.ones_like(per_pair, dtype=torch.bool)
    keep[0, 2] = False                                     # pair (head 3 <- target 2) dropped
    assert abs(m2["con"] - per_pair[keep].mean().item()) < 1e-6


def test_world_model_overfits_one_batch(wm_batch):
    # CLAUDE.md: the world model overfits one fixed batch
    wm, batch = wm_batch
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    _, first, _ = run_loss(wm, batch)
    for _ in range(300):
        loss, m, _ = run_loss(wm, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert m["rec"] < 0.1 * first["rec"]
    assert m["rew"] < 0.2 * first["rew"]
    assert m["con"] < 0.05
