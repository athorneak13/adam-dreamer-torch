"""Recurrent state-space model (paper Eq. 1, first three lines). Written by Adam.

Symbols (tiny_cpu values in brackets):
    B batch [2], T sequence length [8], H imagination horizon [5]
    D deter = GRU state size [512], G GRU blocks [8]
    K stoch = number of categorical latents [4], C classes per latent [8]
    E embedding size from enc, A action size [2]

State (suggested layout, change freely):
    h:      (..., D)      deterministic GRU state
    z:      (..., K * C)  flattened one-hot sample (straight-through)
    logits: (..., K, C)   logits of the distribution z was sampled from

Index convention (CLAUDE.md): replay record t holds (x_t, a_t, r_t, c_t), and
h_t = f(h_{t-1}, z_{t-1}, a_{t-1}). So the action that feeds step t is the
*previous* record's action. Decide where that shift happens.

Open decisions (see CLAUDE.md): GRU input embedding and normalization;
how is_first resets h, z and the previous action.
"""

import torch
from torch import nn


class RSSM(nn.Module):
    def __init__(self, deter: int, blocks: int, stoch: int, classes: int,
                 hidden: int, embed_dim: int, act_dim: int, unimix: float):
        super().__init__()
        raise NotImplementedError

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        """State used at t = 0 and wherever is_first is set.

        Returns: h (B, D), z (B, K*C), logits (B, K, C).
        """
        raise NotImplementedError

    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """One prior step: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ p(z_t | h_t).

        prev_state: h (B, D), z (B, K*C)
        prev_action: (B, A)
        Returns the prior state at t: h (B, D), z (B, K*C), logits (B, K, C).
        """
        raise NotImplementedError

    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor, is_first: torch.Tensor) -> tuple[dict, dict]:
        """One posterior step: same h_t as the prior, then z_t ~ q(z_t | h_t, e_t).

        prev_state: h (B, D), z (B, K*C)
        prev_action: (B, A)
        embed: (B, E)
        is_first: (B,) bool
        Returns (post, prior), each h (B, D), z (B, K*C), logits (B, K, C).
        """
        raise NotImplementedError

    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor, state: dict | None = None) -> tuple[dict, dict]:
        """Run obs_step over T real steps.

        embed: (B, T, E), action: (B, T, A), is_first: (B, T) bool
        state: carried-over state from the previous chunk, or None
        Returns (posts, priors), each h (B, T, D), z (B, T, K*C), logits (B, T, K, C).
        """
        raise NotImplementedError

    def imagine(self, start: dict, policy, horizon: int) -> tuple[dict, torch.Tensor]:
        """Roll the prior forward H steps with actions from the policy.

        start: h (N, D), z (N, K*C), where N = B*T posterior states
        policy: callable, features (N, D + K*C) -> action (N, A)
        Returns states h (N, H+1, D), z (N, H+1, K*C) (index 0 = start)
        and actions (N, H+1, A).
        """
        raise NotImplementedError

    def get_feat(self, state: dict) -> torch.Tensor:
        """Concatenate [h, z] -> (..., D + K*C), the input to heads, actor and critic."""
        raise NotImplementedError
