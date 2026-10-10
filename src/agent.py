"""Agent: world model, actor, critic, one gradient step. Written by Adam.

Shapes (tiny_cpu): features F = recurrent_dim + K*C = 544, action size A.
Official config: actor and critic MLPs have 3 layers; policy outscale 0.01;
critic output zero-initialized (outscale 0.0); slow critic rate 0.02 every step.
"""

import torch
from torch import nn
from src.models.networks import MLP
import torch.nn.functional as F
from src.models.distributions import bounded_normal
from src.models.heads import RewardHead
import copy


class Actor(nn.Module):
    def __init__(self, feat_dim: int, units: int, layers: int, act_dim: int,
                 min_std: float = 0.1, max_std: float = 1.0, outscale: float = 0.01):
        """MLP trunk + two output layers Linear(units, act_dim): mean_raw and std_raw.

        Output layers start with weights scaled by outscale (0.01), so at init the
        policy is mean ~ 0, std ~ 0.89 for every state.
        """
        super().__init__()

        self.min_std = min_std
        self.max_std = max_std

        self.trunk = MLP(feat_dim, units, layers)

        self.mean_head = nn.Linear(units, act_dim)
        self.std_head = nn.Linear(units, act_dim)

        with torch.no_grad():
            self.mean_head.weight.mul_(outscale)
            nn.init.zeros_(self.mean_head.bias)
            self.std_head.weight.mul_(outscale)
            nn.init.zeros_(self.std_head.bias)



    def forward(self, feat: torch.Tensor) -> torch.distributions.Normal:
        """feat: (..., F) -> bounded_normal distribution over actions (..., A)."""
        x = self.trunk(feat)

        mean_raw = self.mean_head(x)
        std_raw = self.std_head(x)

        return bounded_normal(mean_raw=mean_raw, std_raw=std_raw, min_std=self.min_std, max_std=self.max_std)



class Critic(RewardHead):
    """Value network: same structure as RewardHead, built with 3 layers."""



def update_slow_critic(critic: nn.Module, slow_critic: nn.Module, rate: float = 0.02) -> None:
    """EMA update, no gradient: slow <- (1 - rate) * slow + rate * critic, per parameter.

    slow_critic starts as an exact copy (copy.deepcopy(critic)) and is never
    trained by the optimizer; call this once per gradient step.
    """
    with torch.no_grad():
        for p_slow, p in zip(slow_critic.parameters(), critic.parameters()):
            p_slow.mul_(1.0 - rate).add_(p, alpha=rate)
        
