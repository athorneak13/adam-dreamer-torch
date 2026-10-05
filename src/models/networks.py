"""Generic blocks (MLP, RMSNorm, CNN), encoder, decoder, posterior net. Written by Adam."""

import torch
from torch import nn
import math


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