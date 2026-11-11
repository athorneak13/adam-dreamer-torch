"""Tests for checkpoint save/restore and the logger's offline mode."""

import numpy as np
import torch

from src.agent import Agent
from src.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from src.logger import Logger


def make_batch(cfg):
    B, T = cfg.train.batch_size, cfg.train.seq_len
    return dict(obs=torch.randn(B, T, cfg.obs_dim), action=torch.randn(B, T, cfg.act_dim),
                reward=torch.randn(B, T) * 5, cont=torch.ones(B, T),
                is_first=torch.zeros(B, T, dtype=torch.bool))


def test_round_trip_restores_everything(cfg, tmp_path):
    torch.manual_seed(0)
    agent = Agent(cfg, cfg.obs_dim, cfg.act_dim)
    batch = make_batch(cfg)
    for _ in range(3):
        agent.train_step(batch)
    path = tmp_path / "run" / "latest.pt"
    save_checkpoint(path, agent, step=1234)

    torch.manual_seed(1)
    fresh = Agent(cfg, cfg.obs_dim, cfg.act_dim)
    step = load_checkpoint(path, fresh, torch.device("cpu"))

    assert step == 1234
    for (n, a), (_, b) in zip(agent.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), n                          # networks, slow critic, bins
    for name in ("wm_opt", "actor_opt", "critic_opt"):
        sa, sb = getattr(agent, name).state_dict(), getattr(fresh, name).state_dict()
        assert sa["param_groups"] == sb["param_groups"]
        for k in sa["state"]:
            assert sa["state"][k]["step"] == sb["state"][k]["step"]
            assert torch.equal(sa["state"][k]["v"], sb["state"][k]["v"])
            assert torch.equal(sa["state"][k]["m"], sb["state"][k]["m"])
    assert fresh.return_scale.spread == agent.return_scale.spread


def test_resumed_training_matches_uninterrupted(cfg, tmp_path):
    """Two more train steps after a save/load give the same weights as without it."""
    batch = make_batch(cfg)
    torch.manual_seed(0); np.random.seed(0)
    a = Agent(cfg, cfg.obs_dim, cfg.act_dim)
    a.train_step(batch)
    save_checkpoint(tmp_path / "latest.pt", a, step=10)
    for _ in range(2):
        a.train_step(batch)

    torch.manual_seed(5)
    b = Agent(cfg, cfg.obs_dim, cfg.act_dim)
    load_checkpoint(tmp_path / "latest.pt", b, torch.device("cpu"))   # also restores RNG
    for _ in range(2):
        b.train_step(batch)

    for (n, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.allclose(pa, pb), n


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    (tmp_path / "latest.pt").write_bytes(b"x")
    assert latest_checkpoint(tmp_path) == tmp_path / "latest.pt"


def test_logger_offline_prints(cfg, capsys):
    logger = Logger(cfg)                     # tiny_cpu has logging.wandb = false
    assert logger.run is None
    logger.episode(400, -1234.5, 200)
    logger.metrics(500, {"rec": 0.1, "rew": 4.0, "value": 2.0, "ret": -3.0,
                         "entropy": 1.3, "ret_scale": 1.0})
    logger.close()
    out = capsys.readouterr().out
    assert "episode return  -1234.5" in out and "rec 0.100" in out


def test_config_overrides(cfg):
    import pytest
    from src.config import apply_overrides
    apply_overrides(cfg, ["train.horizon=30", "seed=3", "actor.entropy_coef=1.0e-3"])
    assert cfg.train.horizon == 30 and isinstance(cfg.train.horizon, int)
    assert cfg.seed == 3
    assert cfg.actor.entropy_coef == 1e-3
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.horizn=5"])
