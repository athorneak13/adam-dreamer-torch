"""Tests for symlog/symexp, twohot and the bounded-normal actor distribution."""

import pytest
import torch

from src.models.distributions import (bins_expectation, bounded_normal, make_bins,
                                      symexp, symlog, twohot_encode)


@pytest.fixture
def bins():
    return make_bins()


def test_symlog_values():
    x = torch.tensor([-1000., 0., 1000.])
    assert torch.allclose(symlog(x), torch.tensor([-6.9088, 0., 6.9088]), atol=1e-4)


def test_symexp_inverts_symlog():
    x = torch.randn(10_000) * 100
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5, atol=1e-4)


def test_bins_shape_and_values(bins):
    assert bins.shape == (255,)
    assert bins[127] == 0
    assert torch.allclose(bins[[128, 156, 157]], torch.tensor([0.1706, 95.248, 111.664]), atol=1e-3)
    assert (bins[1:] > bins[:-1]).all()


def test_bins_exactly_symmetric(bins):
    assert torch.equal(bins, -bins.flip(0))


def test_twohot_worked_example(bins):
    # CLAUDE.md: y = 100 puts 0.71 on 95.25 and 0.29 on 111.66
    label = twohot_encode(torch.tensor(100.), bins)
    assert label.shape == (255,)
    assert torch.allclose(label[156], torch.tensor(0.7105), atol=1e-4)
    assert torch.allclose(label[157], torch.tensor(0.2895), atol=1e-4)
    assert torch.allclose(label @ bins, torch.tensor(100.))


def test_twohot_batch(bins):
    y = torch.randn(2, 3, 4) * 1000
    label = twohot_encode(y, bins)
    assert label.shape == (2, 3, 4, 255)
    assert torch.allclose(label.sum(-1), torch.ones(2, 3, 4))
    assert (label >= 0).all()
    assert torch.allclose(label @ bins, y, rtol=1e-4, atol=1e-2)


def test_twohot_on_bin_and_edges(bins):
    y = torch.stack([bins[0], bins[127], bins[200], bins[-1]])
    label = twohot_encode(y, bins)
    assert (label >= 0).all()
    assert torch.allclose(label @ bins, y)
    assert label[0, 0] == 1.0 and label[-1, -1] == 1.0


def test_twohot_out_of_range(bins):
    label = twohot_encode(torch.tensor([1e10, -1e10]), bins)
    assert (label >= 0).all()
    assert label[0, -1] == 1.0 and label[1, 0] == 1.0


def test_uniform_expectation_is_exactly_zero(bins):
    probs = torch.ones(3, 255) / 255
    assert (bins_expectation(probs, bins) == 0).all()


def test_expectation_inverts_twohot(bins):
    y = torch.tensor([[100., -3.5, 0.], [1e4, -2e6, 0.17]])
    assert torch.allclose(bins_expectation(twohot_encode(y, bins), bins), y, rtol=1e-5)


def test_expectation_matches_naive_sum(bins):
    probs = torch.softmax(torch.randn(4, 5, 255) * 3, -1)
    v = bins_expectation(probs, bins)
    assert v.shape == (4, 5)
    assert torch.allclose(v, (probs * bins).sum(-1), rtol=1e-4, atol=1e-2)


def test_bounded_normal_init_and_bounds():
    d = bounded_normal(torch.zeros(4, 2), torch.zeros(4, 2))
    assert torch.allclose(d.mean, torch.zeros(4, 2))
    assert torch.allclose(d.stddev, torch.full((4, 2), 0.9 * torch.sigmoid(torch.tensor(2.)).item() + 0.1))

    d = bounded_normal(torch.tensor([[50., -50.]]), torch.tensor([[-50., 50.]]))
    assert torch.allclose(d.mean, torch.tensor([[1., -1.]]))
    assert torch.allclose(d.stddev, torch.tensor([[0.1, 1.0]]))


def test_bounded_normal_log_prob_shape_and_grads():
    mean_raw = torch.zeros(3, 2, requires_grad=True)
    std_raw = torch.zeros(3, 2, requires_grad=True)
    d = bounded_normal(mean_raw, std_raw)
    logp = d.log_prob(d.sample()).sum(-1)
    assert logp.shape == (3,)
    logp.sum().backward()
    assert mean_raw.grad is not None and std_raw.grad is not None


def test_continuous_actor_gradient_example():
    # CLAUDE.md: mu = 0, sigma = 0.5, a = 0.3 -> dlogpi/dmu = 1.2, dlogpi/dsigma = -1.28
    mu = torch.tensor(0., requires_grad=True)
    sigma = torch.tensor(0.5, requires_grad=True)
    torch.distributions.Normal(mu, sigma).log_prob(torch.tensor(0.3)).backward()
    assert torch.allclose(mu.grad, torch.tensor(1.2))
    assert torch.allclose(sigma.grad, torch.tensor(-1.28))


def test_discrete_actor_gradient_example():
    # CLAUDE.md: pi = (0.5, 0.3, 0.2), middle action, A = 0.16 -> (0.08, -0.112, 0.032)
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2])).requires_grad_()
    loss = -0.16 * torch.log_softmax(logits, -1)[1]
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([0.08, -0.112, 0.032]), atol=1e-6)
