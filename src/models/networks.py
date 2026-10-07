"""Generic blocks (MLP, RMSNorm, CNN), encoder, decoder, posterior net. Written by Adam."""

import torch
from torch import nn
import math
from src.models.distributions import symlog, symexp


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """Learned scale g of shape (dim,), initialized to ones."""
        super().__init__()
        self.g = nn.Parameter(torch.ones((dim,)))
        self.eps = eps
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = x / sqrt(mean(x^2) + eps) * g, over the last dim.

        x: (..., dim) -> (..., dim)
        """

        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms+self.eps)
        return (x/rms)*self.g


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int):
        """Trunk only: `layers` x (Linear -> RMSNorm -> SiLU). No output layer.

        First Linear is in_dim -> hidden, the rest hidden -> hidden.
        """
        super().__init__()
        start = nn.Sequential(
            nn.Linear(in_dim, hidden),
            RMSNorm(hidden),
            nn.SiLU()
        )
        middle = [nn.Sequential(nn.Linear(hidden, hidden),
                                RMSNorm(hidden),
                                nn.SiLU()) for _ in range(layers-1)]

        self.net = nn.Sequential(start, *middle)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) -> (..., hidden)"""
        return self.net(x)


class BlockLinear(nn.Module):
    def __init__(self, G: int, in_per_block: int, out_per_block: int):
        """G independent Linears applied in parallel (a block-diagonal weight).

        Weight (G, in_per_block, out_per_block) and bias (G, out_per_block),
        both nn.Parameter.
        """
        super().__init__()
        self.w = nn.Parameter(torch.empty(G, in_per_block, out_per_block))       # uninitialized memory
        bound = 1 / math.sqrt(in_per_block)
        nn.init.uniform_(self.w, -bound, bound)                   # in place: fills w, keeps it a Parameter
        self.b = nn.Parameter(torch.zeros(G, out_per_block))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Block g only uses weight[g].

        x: (..., G, in_per_block) -> (..., G, out_per_block)
        """
        y = torch.einsum('...gi,gio->...go', x, self.w) + self.b
        
        return y

class VectorEncoder(nn.Module):
    def __init__(self, obs_dim: int, units: int, layers: int):
        """symlog (a function, no layer) + MLP(obs_dim, units, layers).

        Output size is self.out_dim (= units); pass it as the RSSM's embed_dim.
        """
        super().__init__()

        self.out_dim = units
        self.net = MLP(obs_dim, self.out_dim, layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., obs_dim) raw -> e: (..., out_dim)."""
        x = symlog(obs)
        return self.net(x)


class VectorDecoder(nn.Module):
    def __init__(self, feat_dim: int, units: int, layers: int, obs_dim: int):
        """MLP trunk + Linear(units, obs_dim) output layer."""
        super().__init__()

        self.net = nn.Sequential(
                    MLP(feat_dim, units, layers),
                    nn.Linear(units, obs_dim)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (..., F) -> predicted symlog(obs): (..., obs_dim).

        Compare with symlog(obs) by MSE; do not symexp before the loss.
        """
        return self.net(feat)
