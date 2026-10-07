"""Reward and continue heads. Written by Adam.

Both read features [h_t, z_t] of size F = D + K*C (544 in tiny_cpu).
Official config: 1 hidden layer each (rewhead/conhead layers: 1).
"""

import torch
from torch import nn
from src.models.networks import MLP
from src.models.distributions import bins_expectation, make_bins


class RewardHead(nn.Module):
    def __init__(self, feat_dim: int, units: int, layers: int, num_bins: int = 255):
        """MLP trunk + Linear(units, num_bins) output layer, zero-initialized.

        Also holds the twohot bins as a buffer (make_bins(num_bins)).
        """
        super().__init__()

        trunk = MLP(feat_dim, units, layers)
        out = nn.Linear(units, num_bins)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)

        self.net = nn.Sequential(
            trunk,
            out
        )

        self.register_buffer("bins", make_bins(num_bins))
        

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (..., F) -> bin logits (..., num_bins)."""
        return self.net(feat)

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Expected reward in real units: bins_expectation(softmax(logits), bins).

        feat: (..., F) -> (...)
        """
        return bins_expectation(torch.softmax(self(feat), dim=-1), self.bins)


class ContinueHead(nn.Module):
    def __init__(self, feat_dim: int, units: int, layers: int):
        """MLP trunk + Linear(units, 1) output layer."""
        super().__init__()

        self.net = nn.Sequential(
                   MLP(feat_dim, units, layers),
                   nn.Linear(units, 1) 
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (..., F) -> raw logit per state (...), for the BCE-with-logits loss.

        Use predict() for the probability.
        """

        return self.net(feat).squeeze(-1)

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Continue probability p(c_t = 1) = sigmoid(logit).

        feat: (..., F) -> (...)
        """
        return torch.sigmoid(self(feat))
