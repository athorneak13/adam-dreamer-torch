"""Recurrent state-space model (paper Eq. 1, first three lines). Written by Adam.

Symbols (tiny_cpu values in brackets):
    B batch [2], T sequence length [8], H imagination horizon [5]
    D recurrent_dim = size of h_t [512], G gru_blocks [8]
    K number of categorical latents [4], C classes per latent [8]
    E embedding size from enc, A action size [2]

State (suggested layout, change freely):
    h:      (..., D)      deterministic GRU state
    z:      (..., K * C)  flattened one-hot sample (straight-through)
    probs:  (..., K, C)   unimixed probabilities z was sampled from

Index convention (CLAUDE.md): replay record t holds (x_t, a_t, r_t, c_t), and
h_t = f(h_{t-1}, z_{t-1}, a_{t-1}). So the action that feeds step t is the
*previous* record's action. Decide where that shift happens.

Open decisions (see CLAUDE.md): GRU input embedding and normalization;
how is_first resets h, z and the previous action.
"""

import torch
from torch import nn
from src.models.networks import MLP, BlockLinear, RMSNorm
import torch.nn.functional as F


def sample_categorical(logits: torch.Tensor, K: int, C: int,
                       unimix: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits -> straight-through one-hot sample z, plus the distribution the KL needs.

    logits: (..., K*C)
    Returns z (..., K*C) and the unimixed distribution (..., K, C)
    (you decide: probabilities p~ or log p~).
    """
    x = logits.reshape(*logits.shape[:-1], K, C)
    probs = torch.softmax(x, dim=-1)
    probs = ((1 - unimix) * probs) + (unimix / C)
    idx = torch.distributions.Categorical(probs=probs).sample()
    z = F.one_hot(idx, num_classes = C).float() + probs - probs.detach()
    z = z.reshape(*z.shape[:-2], -1)
    return z, probs


    


class RSSM(nn.Module):
    def __init__(self, recurrent_dim: int, gru_blocks: int, K: int,
                 C: int, mlp_units: int, prior_layers: int,
                 post_layers: int, embed_dim: int, act_dim: int, unimix: float):
        super().__init__()

        self.K = K
        self.C = C
        self.unimix = unimix
        self.recurrent_dim = recurrent_dim
        self.G = gru_blocks
        self.block_size = recurrent_dim // gru_blocks
        assert recurrent_dim % gru_blocks == 0
        

        self.prior = nn.Sequential(
            MLP(recurrent_dim, mlp_units, prior_layers),
            nn.Linear(mlp_units, K*C))
        self.posterior = nn.Sequential(
            MLP(recurrent_dim+embed_dim, mlp_units, post_layers),
            nn.Linear(mlp_units, K*C)
            )

        self.h_embedding = MLP(recurrent_dim, mlp_units, 1)
        self.z_embedding = MLP(K*C, mlp_units, 1)
        self.a_embedding = MLP(act_dim, mlp_units, 1)

        self.gru_hidden = BlockLinear(self.G, self.block_size + 3*mlp_units, self.block_size)
        self.gru_norm = RMSNorm(recurrent_dim)
        self.gru_gates = BlockLinear(self.G, self.block_size, 3*self.block_size)



    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        """State used at t = 0 and wherever is_first is set.

        Returns: h (B, D), z (B, K*C), probs (B, K, C).
        """
        h_0 = torch.zeros(batch_size, self.recurrent_dim, device=device)
        z_0 = torch.zeros(batch_size, self.K*self.C, device=device)
        probs = torch.ones(batch_size, self.K, self.C, device=device)/self.C

        return {"h":h_0, "z": z_0, "probs": probs}

    def _gru_step(self, h: torch.Tensor, z: torch.Tensor,
                  a: torch.Tensor) -> torch.Tensor:
        """Block GRU: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}).

        h: (B, D), z: (B, K*C), a: (B, A) -> h_t: (B, D)

        1. embed h, z, a and concat           -> (B, 3*units)
        2. split h into blocks, append embed  -> (B, G, D/G + 3*units)
        3. block hidden layer, norm, SiLU     -> (B, G, D/G)
        4. gate layer, split in 3             -> 3 x (B, G, D/G)
        5. gate equations, flatten            -> (B, D)
        """
        h_embed = self.h_embedding(h)
        z_embed = self.z_embedding(z)
        a_embed = self.a_embedding(a)

        h_blocks = h.unflatten(-1, (self.G, self.block_size))   # (B, 512) → (B, 8, 64)


        gru_input = torch.cat((h_embed, z_embed, a_embed), dim=-1)
        gru_input = gru_input.unsqueeze(-2).expand(-1, self.G, -1)
        block_input = torch.cat((gru_input, h_blocks), dim=-1)

        x = self.gru_hidden(block_input)
        x = x.flatten(-2)
        x = self.gru_norm(x)
        x = F.silu(x)
        x = x.unflatten(-1, (self.G, self.block_size))   # (B, 512) → (B, 8, 64)
        x = self.gru_gates(x)
        raw_r, raw_c, raw_u = torch.chunk(x, 3, dim=-1)

        reset = F.sigmoid(raw_r)
        cand = F.tanh(reset * raw_c)
        update = F.sigmoid(raw_u - 1.0)

        h_t = update * cand + (1.0 - update)*h_blocks
        h_t = h_t.flatten(-2)

        return h_t
    
    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """One prior step: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ p(z_t | h_t).

        prev_state: h (B, D), z (B, K*C)
        prev_action: (B, A)
        Returns the prior state at t: h (B, D), z (B, K*C), probs (B, K, C).
        """
        h_tminus1 = prev_state["h"]
        z_tminus1 = prev_state["z"]
        h_t = self._gru_step(h_tminus1, z_tminus1, prev_action)
        logits = self.prior(h_t)
        z_t, probs = sample_categorical(logits, self.K, self.C, self.unimix)

        return {"h": h_t, "z": z_t, "probs": probs}



    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor, is_first: torch.Tensor) -> tuple[dict, dict]:
        """One posterior step: same h_t as the prior, then z_t ~ q(z_t | h_t, e_t).

        prev_state: h (B, D), z (B, K*C)
        prev_action: (B, A)
        embed: (B, E)
        is_first: (B,) bool
        Returns (post, prior), each h (B, D), z (B, K*C), probs (B, K, C).
        """

        keep = 1.0 - is_first.float()
        keep = keep.unsqueeze(-1)
        prev_state = {"h": prev_state["h"]*keep, "z": prev_state["z"]*keep, "probs":prev_state["probs"]}
        prev_action = prev_action * keep
        prior_state = self.img_step(prev_state, prev_action)
        h_t = prior_state["h"]
        logits = self.posterior(torch.cat([h_t, embed], dim=-1))
        z_t, probs = sample_categorical(logits, self.K, self.C, self.unimix)
        return {"h": h_t, "z": z_t, "probs": probs}, prior_state



    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor, state: dict | None = None) -> tuple[dict, dict]:
        """Run obs_step over T real steps.

        embed: (B, T, E), action: (B, T, A), is_first: (B, T) bool
        state: carried-over state from the previous chunk, or None
        Returns (posts, priors), each h (B, T, D), z (B, T, K*C), probs (B, T, K, C).
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
