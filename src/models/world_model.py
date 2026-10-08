"""World model: encoder, RSSM, decoder, reward and continue heads (phi). Written by Adam.

Shapes (tiny_cpu): B batch, T sequence length, obs_dim O, action size A,
features F = recurrent_dim + K*C = 544.
"""

import torch
from torch import nn
from src.models.networks import VectorDecoder, VectorEncoder
from src.models.heads import RewardHead, ContinueHead
from src.models.rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, cfg, obs_dim: int, act_dim: int):
        """Build all parts from cfg.model; obs_dim and act_dim come from the env.

        encoder: VectorEncoder(obs_dim, mlp_units, mlp_layers)
        rssm:    RSSM(..., embed_dim=encoder.out_dim, act_dim, ...)
        decoder: VectorDecoder(F, mlp_units, mlp_layers, obs_dim)
        reward:  RewardHead(F, mlp_units, 1, bins)
        cont:    ContinueHead(F, mlp_units, 1)
        """
        super().__init__()

        feat_dim = cfg.model.recurrent_dim + (cfg.model.K*cfg.model.C)

        self.encoder = VectorEncoder(obs_dim=obs_dim, units=cfg.model.mlp_units, layers=cfg.model.mlp_layers)
        self.rssm = RSSM(recurrent_dim=cfg.model.recurrent_dim, 
                         gru_blocks=cfg.model.gru_blocks, 
                         K=cfg.model.K, C=cfg.model.C, 
                         mlp_units=cfg.model.mlp_units,
                         prior_layers=cfg.model.prior_layers,
                         post_layers=cfg.model.post_layers,
                         embed_dim=self.encoder.out_dim,
                         act_dim=act_dim,
                         unimix=cfg.model.unimix
                         )
        self.decoder = VectorDecoder(feat_dim=feat_dim,
                                     units=cfg.model.mlp_units, 
                                     layers=cfg.model.mlp_layers,
                                     obs_dim=obs_dim)

        self.reward = RewardHead(feat_dim=feat_dim,
                                 units=cfg.model.mlp_units,
                                 layers=cfg.model.head_layers,
                                 num_bins=cfg.model.bins
                                 )

        self.cont = ContinueHead(feat_dim=feat_dim,
                                    units=cfg.model.mlp_units,
                                    layers=cfg.model.head_layers
                                )

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor, state: dict | None = None) -> dict:
        """Run the world model on a batch of real sequences.

        obs: (B, T, O) raw, action: (B, T, A), is_first: (B, T) bool
        state: carried-over RSSM state, or None

        Returns a dict:
            posts, priors: RSSM state dicts, h (B, T, D), z (B, T, K*C), probs (B, T, K, C)
            feat:          (B, T, F), features from the posteriors
            recon:         (B, T, O), predicted symlog(obs)
            reward_logits: (B, T, bins)
            cont_logits:   (B, T)
        """
        embed = self.encoder(obs)
        posts, priors = self.rssm.observe(embed=embed, 
                                          action=action, 
                                          is_first=is_first, 
                                          state=state)
        feat = self.rssm.get_feat(posts)
        recon = self.decoder(feat)
        reward = self.reward(feat)
        cont = self.cont(feat)

        return {"posts": posts, "priors": priors, "feat": feat, "recon": recon, 
                "reward_logits": reward, "cont_logits": cont}

