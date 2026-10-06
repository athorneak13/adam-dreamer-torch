"""RSSM tests on the tiny_cpu config with random tensors."""

import pytest
import torch

from src.models.rssm import RSSM, sample_categorical

E = 64  # stand-in encoder embedding size
CPU = torch.device("cpu")


@pytest.fixture
def rssm(cfg):
    m = cfg.model
    return RSSM(m.recurrent_dim, m.gru_blocks, m.K, m.C, m.mlp_units,
                m.prior_layers, m.post_layers, E, cfg.act_dim, m.unimix)


@pytest.fixture
def seq(cfg):
    """Random (embed, action, is_first) for B sequences of length T."""
    B, T = cfg.train.batch_size, cfg.train.seq_len
    embed = torch.randn(B, T, E)
    action = torch.randn(B, T, cfg.act_dim)
    is_first = torch.zeros(B, T, dtype=torch.bool)
    return embed, action, is_first


def observe_seeded(rssm, embed, action, is_first, seed=1):
    """observe() with a fixed sampling seed, so two calls are comparable."""
    torch.manual_seed(seed)
    return rssm.observe(embed, action, is_first)


def test_initial_state_shapes(cfg, rssm):
    B, m = cfg.train.batch_size, cfg.model
    s = rssm.initial_state(B, CPU)
    assert s["h"].shape == (B, m.recurrent_dim)
    assert s["z"].shape == (B, m.K * m.C)
    assert s["probs"].shape == (B, m.K, m.C)
    assert torch.allclose(s["probs"].sum(-1), torch.ones(B, m.K))


def test_z_is_one_hot_forward(cfg):
    K, C = cfg.model.K, cfg.model.C
    z, _ = sample_categorical(torch.randn(5, K * C), K, C, cfg.model.unimix)
    assert z.shape == (5, K * C)
    rows = z.reshape(5, K, C)
    assert torch.allclose(rows.sum(-1), torch.ones(5, K))
    assert ((rows > 0.5).sum(-1) == 1).all()


def test_straight_through_gradient(cfg):
    K, C = cfg.model.K, cfg.model.C
    logits = torch.randn(5, K * C, requires_grad=True)
    z, probs = sample_categorical(logits, K, C, cfg.model.unimix)
    probs.retain_grad()
    w = torch.randn_like(z)
    (w * z).sum().backward()
    assert torch.allclose(probs.grad.flatten(-2), w)
    assert logits.grad is not None


def test_unimix_floor(cfg):
    K, C, unimix = cfg.model.K, cfg.model.C, cfg.model.unimix
    logits = torch.randn(5, K * C) * 50  # extreme logits push softmax to ~0
    _, probs = sample_categorical(logits, K, C, unimix)
    assert (probs >= unimix / C - 1e-7).all()
    assert torch.allclose(probs.sum(-1), torch.ones(5, K))


def test_unimix_worked_example():
    # Study-notes example: logits [0, 2, 0, 0] -> [0.0978, 0.7066, 0.0978, 0.0978]
    _, probs = sample_categorical(torch.tensor([[0., 2., 0., 0.]]), 1, 4, 0.01)
    expected = torch.tensor([0.0978, 0.7066, 0.0978, 0.0978])
    assert torch.allclose(probs.flatten(), expected, atol=1e-4)


def test_observe_shapes(cfg, rssm, seq):
    B, T, m = cfg.train.batch_size, cfg.train.seq_len, cfg.model
    posts, priors = rssm.observe(*seq)
    for states in (posts, priors):
        assert states["h"].shape == (B, T, m.recurrent_dim)
        assert states["z"].shape == (B, T, m.K * m.C)
        assert states["probs"].shape == (B, T, m.K, m.C)
    assert torch.equal(posts["h"], priors["h"])


def test_posts_differ_from_priors(rssm, seq):
    posts, priors = rssm.observe(*seq)
    assert not torch.equal(posts["probs"], priors["probs"])


def test_observe_memory(rssm, seq):
    """Changing the observation at t=0 changes h at t=3."""
    embed, action, is_first = seq
    changed = embed.clone()
    changed[:, 0] += 5.0
    p1, _ = observe_seeded(rssm, embed, action, is_first)
    p2, _ = observe_seeded(rssm, changed, action, is_first)
    assert not torch.allclose(p1["h"][:, 3], p2["h"][:, 3])


def test_action_shift(rssm, seq):
    """h_t uses a_{t-1}: a_0 first affects h_1, and the last action is unused."""
    embed, action, is_first = seq
    base, _ = observe_seeded(rssm, embed, action, is_first)

    a0 = action.clone()
    a0[:, 0] += 5.0
    p, _ = observe_seeded(rssm, embed, a0, is_first)
    assert torch.allclose(base["h"][:, 0], p["h"][:, 0])
    assert not torch.allclose(base["h"][:, 1], p["h"][:, 1])

    last = action.clone()
    last[:, -1] += 5.0
    p, _ = observe_seeded(rssm, embed, last, is_first)
    assert torch.allclose(base["h"], p["h"])


def test_is_first_resets_state(rssm, seq):
    """After is_first at t=4 in row 1, row 1 forgets everything before t=4."""
    embed, action, is_first = seq
    reset = is_first.clone()
    reset[1, 4] = True
    changed = embed.clone()
    changed[:, 0] += 5.0

    p1, _ = observe_seeded(rssm, embed, action, reset)
    p2, _ = observe_seeded(rssm, changed, action, reset)
    # Row 1 from t=4 on no longer depends on e_0; row 0 still does.
    assert torch.allclose(p1["h"][1, 4:], p2["h"][1, 4:])
    assert not torch.allclose(p1["h"][0, 4:], p2["h"][0, 4:])


def test_imagine_shapes(cfg, rssm, seq):
    B, T, H, m = cfg.train.batch_size, cfg.train.seq_len, cfg.train.horizon, cfg.model
    posts, _ = rssm.observe(*seq)
    N = B * T
    start = {k: posts[k].reshape(N, *posts[k].shape[2:]) for k in ("h", "z")}
    policy = lambda feat: torch.tanh(feat[:, :cfg.act_dim])

    states, actions = rssm.imagine(start, policy, H)
    assert states["h"].shape == (N, H + 1, m.recurrent_dim)
    assert states["z"].shape == (N, H + 1, m.K * m.C)
    assert actions.shape == (N, H + 1, cfg.act_dim)
    assert not states["h"].requires_grad and not actions.requires_grad
    assert torch.allclose(states["h"][:, 0], start["h"])
    # actions[t] is the action taken in states[t]
    assert torch.allclose(actions, torch.tanh(states["h"][..., :cfg.act_dim]))
