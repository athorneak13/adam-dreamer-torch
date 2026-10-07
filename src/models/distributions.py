"""symlog/symexp, twohot, bounded-normal actor distribution. Written by Adam."""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(|x| + 1), element-wise. Any shape -> same shape."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * (exp(|x|) - 1), element-wise. Inverse of symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_bins(num_bins: int = 255, high: float = 20.0) -> torch.Tensor:
    """Bin locations in real units: symexp of num_bins points evenly spaced
    in [-high, high], built by mirroring the non-negative half so the
    middle bin is exactly 0 and the bins are exactly symmetric.

    Returns the bins, shape (num_bins,). num_bins must be odd.
    Computed once at setup; register as a buffer.
    """
    pos = torch.linspace(0, high, (num_bins+1)//2)
    full = torch.cat([-pos[1:].flip(0), pos])
    return symexp(full)


def twohot_encode(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Split each target between its two neighboring bins (real-unit distances).

    y: (...) raw targets, no symlog. bins: (num_bins,)
    Returns (..., num_bins); each row sums to 1 and row @ bins == y.
    Example: y = 100 -> 0.71 on 95.25, 0.29 on 111.66.
    """
    C = bins.shape[-1]
    y = torch.clamp(y, bins[0], bins[-1])
    above_index = torch.clamp(torch.searchsorted(bins, y), 1, C-1)
    below_index = above_index - 1
    above_difference = (bins[above_index] - y).unsqueeze(-1)
    below_difference =  (y - bins[below_index]).unsqueeze(-1)
    gap = (bins[above_index] - bins[below_index]).unsqueeze(-1)

    onehot_above = F.one_hot(above_index, C).float() * (below_difference/gap)
    onehot_below = F.one_hot(below_index, C).float() * (above_difference/gap)
    

    return onehot_above + onehot_below



def bins_expectation(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value sum(probs * bins), summed outward from the middle bin.

    probs: (..., num_bins), bins: (num_bins,) -> (...)
    Uniform probs must give exactly 0.
    """
    mid = bins.shape[-1] // 2
    neg_half = (probs[...,:mid] * bins[...,:mid]).flip(-1).sum(dim=-1)
    pos_half = (probs[...,mid+1:] * bins[...,mid+1:]).sum(dim=-1)
    return neg_half + pos_half


def bounded_normal(mean_raw: torch.Tensor, std_raw: torch.Tensor,
                   min_std: float = 0.1, max_std: float = 1.0) -> torch.distributions.Normal:
    """Actor distribution: mean = tanh(mean_raw),
    std = (max_std - min_std) * sigmoid(std_raw + 2) + min_std.

    mean_raw, std_raw: (..., A). Samples are not squashed.
    log_prob(a).sum(-1) gives log pi(a | s).
    """
    mean = torch.tanh(mean_raw)
    std = (max_std - min_std) * torch.sigmoid(std_raw + 2.0) + min_std
    
    return torch.distributions.Normal(mean, std)
