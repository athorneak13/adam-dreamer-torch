"""Agent: world model, actor, critic, one gradient step. Written by Adam.

Shapes (tiny_cpu): features F = recurrent_dim + K*C = 544, action size A.
Official config: actor and critic MLPs have 3 layers; policy outscale 0.01;
critic output zero-initialized (outscale 0.0); slow critic rate 0.02 every step.
"""

import torch
from torch import nn
from src.models.networks import MLP
import torch.nn.functional as F
from src.models.distributions import bounded_normal, make_bins
from src.models.heads import RewardHead
import copy
from src.models.world_model import WorldModel
from src.returns import ReturnScale, replay_return
from src.losses import world_model_loss, continue_weights, critic_loss, actor_loss
from src.returns import lambda_return
from src.optim import LaProp


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
        


class Agent(nn.Module):
    def __init__(self, cfg, obs_dim: int, act_dim: int):
        """Build all learners and their optimizers.

        world_model: WorldModel(cfg, obs_dim, act_dim)
        actor:       Actor(F, mlp_units, mlp_layers, act_dim)
        critic:      Critic(F, mlp_units, mlp_layers, bins)
        slow_critic: copy.deepcopy(critic).requires_grad_(False), never optimized
        ret_scale:   ReturnScale()
        optimizers:  built from each module's own parameters, never self.parameters()
                     (placeholder optimizer until optim.py exists)
        """
        super().__init__()

        self.feat_dim = cfg.model.recurrent_dim + (cfg.model.K * cfg.model.C)
        self.world_model = WorldModel(cfg=cfg, obs_dim=obs_dim, act_dim=act_dim)

        self.actor = Actor(feat_dim=self.feat_dim, 
                           units=cfg.model.mlp_units,
                           layers = cfg.model.mlp_layers, 
                           act_dim=act_dim, 
                           min_std=cfg.actor.min_std,
                           max_std=cfg.actor.max_std,
                           outscale=cfg.actor.outscale)
        
        self.critic = Critic(feat_dim=self.feat_dim,
                             units=cfg.model.mlp_units,
                             layers=cfg.model.mlp_layers,
                             num_bins=cfg.model.bins
                             )
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.return_scale = ReturnScale()

        # self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=cfg.optim.lr)
        # self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.optim.lr)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr = cfg.optim.lr)

        self.wm_opt = LaProp(self.world_model.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
                            agc=cfg.optim.agc, pmin=cfg.optim.pmin, warmup=cfg.optim.warmup
                            )

        self.actor_opt = LaProp(self.actor.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
                            agc=cfg.optim.agc, pmin=cfg.optim.pmin, warmup=cfg.optim.warmup
                            )

        self.critic_opt = LaProp(self.critic.parameters(), lr=cfg.optim.lr,
                            betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
                            agc=cfg.optim.agc, pmin=cfg.optim.pmin, warmup=cfg.optim.warmup
                            )

        self.scales = vars(cfg.loss_scales)
        self.cfg = cfg
        
        self.horizon = cfg.train.horizon

        

    def train_step(self, batch: dict) -> dict:
        """One gradient step on a replay batch. Returns metrics (floats).

        batch: obs (B, T, O), action (B, T, A), reward (B, T), cont (B, T), is_first (B, T)

        1. World model: out = world_model(obs, action, is_first); world_model_loss.
        2. Starts: posterior states flattened to N = B*T, detached.
        3. Imagine H steps with the actor: states (N, H+1, ...), actions (N, H+1, A).
        4. Targets (no grad), on imagined features feat (N, H+1, F):
           reward = reward head on feat[:, 1:], cont = continue head on feat[:, 1:],
           value = critic on feat; returns = lambda_return; weight = continue_weights.
        5. Advantage: ret_scale.update(returns); (returns - value[:, :-1]) / scale.
        6. Critic loss on feat[:, :-1] toward returns, with slow-critic regularizer.
        7. Replay critic loss: R^img = returns[:, 0] reshaped to (B, T); R^rep =
           replay_return(...); critic on the posterior features out["feat"][:, :-1]
           (gradients flow into the world model), mask valid = ~is_first[:, 1:].
        8. Actor loss on actor(feat[:, :-1]) and actions[:, :-1].
        9. Backward the weighted total, step the optimizers, update_slow_critic.
        """
        B, T = batch["obs"].shape[:2]
        world_model_dict = self.world_model(batch["obs"], batch["action"], batch["is_first"])
        wm_loss, wm_metrics = world_model_loss(world_model_dict, 
                                   batch["obs"], 
                                   batch["reward"],
                                   batch["cont"],
                                   batch["is_first"],
                                   bins=self.world_model.reward.bins,
                                   scales=self.scales,
                                   free_nats=self.cfg.free_nats
                                )
        hidden_post = world_model_dict["posts"]["h"]
        z_post = world_model_dict["posts"]["z"]
        hidden_post = hidden_post.flatten(0, 1).detach()
        z_post = z_post.flatten(0, 1).detach()

        start = {"h": hidden_post, "z": z_post}
        policy_fn = lambda feat: self.actor(feat).sample()
        states, actions = self.world_model.rssm.imagine(start, policy_fn, self.horizon)

        with torch.no_grad():
            feat = self.world_model.rssm.get_feat(states)
            rewards = self.world_model.reward.predict(feat[:, 1:])
            cont = self.world_model.cont.predict(feat[:, 1:])
            values = self.critic.predict(feat)
            returns = lambda_return(reward=rewards,
                                    cont=cont,
                                    value=values,
                                    gamma=self.cfg.critic.gamma,
                                    lam=self.cfg.critic.lam)
            cont_weights = continue_weights(cont)
            self.return_scale.update(returns)
            advantage = (returns - values[:, :-1]) / self.return_scale.scale()

        critic_logits = self.critic(feat[:, :-1])
        slow_value = self.slow_critic.predict(feat[:, :-1])
        value_loss = critic_loss(logits=critic_logits, 
                                 target=returns, 
                                 slow_value=slow_value,
                                 weight=cont_weights,
                                 bins=self.world_model.reward.bins,
                                 slow_reg=self.cfg.critic.slow_reg
                                 )

        with torch.no_grad():
            R_img_t = returns[:, 0].reshape(B, T)
            replay_targets = replay_return(reward=batch["reward"], 
                                           cont=batch["cont"], 
                                           img_return=R_img_t, 
                                           is_first=batch["is_first"], 
                                           gamma=self.cfg.critic.gamma, 
                                           lam=self.cfg.critic.lam)

        replay_logits = self.critic(world_model_dict["feat"][:, :-1])
        with torch.no_grad():
            replay_slow_value = self.slow_critic.predict(world_model_dict["feat"][:, :-1])
        replay_mask = (~batch["is_first"][:, 1:]).float()
        repval_loss = critic_loss(logits=replay_logits, 
                                  target=replay_targets, 
                                  slow_value=replay_slow_value, 
                                  weight=replay_mask, 
                                  bins=self.world_model.reward.bins, 
                                  slow_reg=self.cfg.critic.slow_reg)

        actor_dist = self.actor(feat[:, :-1])
        policy_loss = actor_loss(dist=actor_dist,
                                 action=actions[:, :-1],
                                 advantage=advantage,
                                 weight=cont_weights,
                                 entropy_coef=self.cfg.actor.entropy_coef)

        total = wm_loss + self.scales["value"]*value_loss + self.scales["repval"]*repval_loss + self.scales["policy"] * policy_loss

        self.wm_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()

        total.backward()

        for opt, module in [(self.wm_opt, self.world_model),
                    (self.actor_opt, self.actor),
                    (self.critic_opt, self.critic)]:
            
            #torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=100.0)
            opt.step()

        update_slow_critic(self.critic, self.slow_critic, self.cfg.critic.slow_rate)

        metrics_dict = {**wm_metrics, "value": value_loss.item(),
                        "repval": repval_loss.item(),
                        "policy": policy_loss.item(),
                        "ret": returns.mean().item(),
                        "adv_mag": advantage.abs().mean().item(),
                        "ret_scale": self.return_scale.scale(),
                        "entropy": actor_dist.entropy().sum(-1).mean().item(),
                        # fraction of imagined actions outside [-1, 1] (clipped in the RSSM)
                        "act_oob": (actions.abs() > 1).float().mean().item()}

        
        return metrics_dict



    @torch.no_grad()
    def policy(self, obs: torch.Tensor, state: dict | None, prev_action: torch.Tensor,
               is_first: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, dict]:
        """Act in the real environment from the posterior state.

        obs: (E, O) raw, prev_action: (E, A), is_first: (E,) bool, state: RSSM state or None
        encode -> rssm.obs_step -> actor(feat) -> sample (or mean when sample=False).
        Returns action (E, A) and the new posterior state, to pass back next step.
        """
        E = obs.shape[0]
        embed = self.world_model.encoder(obs) # (E, 64)
        if state is None:
            state = self.world_model.rssm.initial_state(E, obs.device)

        post, _ = self.world_model.rssm.obs_step(prev_state=state, 
                                                 prev_action=prev_action,
                                                 embed=embed,
                                                 is_first=is_first)
        dist = self.actor(self.world_model.rssm.get_feat(post))
        action = dist.sample() if sample else dist.mean

        return action, post
