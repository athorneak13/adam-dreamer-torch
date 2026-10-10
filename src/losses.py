"""World-model, critic and actor losses. Written by Adam.

Reduction (matches the official code): each term is first reduced to one
number per step, shape (B, T), by summing over its feature dims. Free bits
are applied per step. Each term is then averaged over all B*T steps, and the
total is the weighted sum of those means.
"""

import torch
from src.models.distributions import twohot_encode, symlog
import torch.nn.functional as F


# ---------- world model: per-step terms, each returns (B, T) ----------

def recon_loss(recon: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Squared error in symlog space, summed over obs dims.

    recon: (B, T, O) predicted symlog(obs), obs: (B, T, O) raw -> (B, T)
    """
    return torch.sum((recon - symlog(obs).detach())**2, dim=-1)


def reward_loss(reward_logits: torch.Tensor, reward: torch.Tensor,
                bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between twohot(reward) and softmax(logits).

    reward_logits: (B, T, num_bins), reward: (B, T) raw -> (B, T)
    Predictions and targets must already be time-aligned (see the offset).
    """
    label = twohot_encode(reward, bins)
    return -(label * torch.log_softmax(reward_logits, dim=-1)).sum(-1)


def continue_loss(cont_logits: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits.

    cont_logits: (B, T), cont: (B, T) in {0, 1} -> (B, T)
    Predictions and targets must already be time-aligned (see the offset).
    """
    return F.binary_cross_entropy_with_logits(cont_logits, cont, reduction="none")


def categorical_kl(q_probs: torch.Tensor, p_probs: torch.Tensor) -> torch.Tensor:
    """KL[q || p] for K independent categoricals, summed over C and K.

    q_probs, p_probs: (B, T, K, C), already unimixed -> (B, T)
    """
    return torch.sum(q_probs*(torch.log(q_probs) - torch.log(p_probs)), dim=(-2, -1))


def dyn_rep_losses(post_probs: torch.Tensor, prior_probs: torch.Tensor,
                   free_nats: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """dyn = max(free_nats, KL[sg(q) || p]),  rep = max(free_nats, KL[q || sg(p)]).

    post_probs, prior_probs: (B, T, K, C) -> dyn (B, T), rep (B, T)
    Free bits per step, before any averaging.
    """
    dyn_loss = torch.clamp(categorical_kl(post_probs.detach(), prior_probs), min=free_nats)

    rep_loss = torch.clamp(categorical_kl(post_probs, prior_probs.detach()), min=free_nats)

    return dyn_loss, rep_loss


# ---------- world model: total ----------

def world_model_loss(out: dict, obs: torch.Tensor, reward: torch.Tensor,
                     cont: torch.Tensor, is_first: torch.Tensor, bins: torch.Tensor,
                     scales: dict, free_nats: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Combine the five terms.

    out: WorldModel.forward output. obs (B, T, O), reward (B, T), cont (B, T),
    is_first (B, T) bool.
    scales: {"rec": 1, "rew": 1, "con": 1, "dyn": 1, "rep": 0.1}
    Reward/continue offset: the head at t+1 predicts r_t, c_t. Pairs where
    is_first[:, t+1] is True cross an episode boundary and are masked out.

    Returns (total scalar, {term name: mean over B*T}) for logging.
    """
    valid = (~is_first[:, 1:]).float()
    rec = recon_loss(recon=out["recon"], obs=obs)
    rew = reward_loss(reward_logits=out["reward_logits"][:,1:], reward=reward[:, :-1], bins=bins)
    rew = (rew * valid).sum() / valid.sum().clamp(min=1)
    con = continue_loss(cont_logits=out["cont_logits"][:,1:], cont=cont[:, :-1])
    con = (con * valid).sum() / valid.sum().clamp(min=1)
    dyn, rep = dyn_rep_losses(post_probs=out["posts"]["probs"], prior_probs=out["priors"]["probs"], free_nats=free_nats)

    metrics = {"rec": rec.mean().item(), "rew": rew.item(), "con": con.item(), "dyn": dyn.mean().item(), "rep": rep.mean().item()}

    loss = scales["rec"] * rec.mean() + scales["rew"] * rew + scales["con"] * con + scales["dyn"] * dyn.mean() + scales["rep"] * rep.mean()

    return loss, metrics
    


# ---------- actor-critic (imagination and replay) ----------

def continue_weights(cont: torch.Tensor) -> torch.Tensor:
    """Probability each imagined state is still alive: w_0 = 1, w_{t+1} = w_t * c_t.

    cont: (N, H) predicted c_t for t = 0..H-1 (from the continue head at state t+1)
    Returns (N, H): w_t for t = 0..H-1, i.e. [1, c_0, c_0*c_1, ...]. No gradient.
    """
    with torch.no_grad():
        ones_col = torch.ones_like(cont[:, :1])
        cont = torch.cat([ones_col, cont[:, :-1]], dim=1)
        return torch.cumprod(cont, dim=1)
        


def critic_loss(logits: torch.Tensor, target: torch.Tensor, slow_value: torch.Tensor,
                weight: torch.Tensor, bins: torch.Tensor, slow_reg: float = 1.0) -> torch.Tensor:
    """Twohot cross-entropy toward the return, plus the slow-critic regularizer.

    logits:     (N, L, num_bins) critic logits on the states that have a target
    target:     (N, L) return (lambda-return in imagination, R^rep on replay)
    slow_value: (N, L) slow critic's value on the same states
    weight:     (N, L) continue weights w_t (imagination) or the boundary mask (replay)
    per step:   weight * (CE(twohot(target)) + slow_reg * CE(twohot(slow_value)))
    Targets and weights carry no gradient. Returns a scalar: mean over N*L.
    """
    loss = weight.detach() * (reward_loss(logits, target.detach(), bins) + slow_reg * reward_loss(logits, slow_value.detach(), bins))
    loss = loss.mean()
    return loss


def actor_loss(dist: torch.distributions.Normal, action: torch.Tensor, advantage: torch.Tensor,
               weight: torch.Tensor, entropy_coef: float = 3e-4) -> torch.Tensor:
    """Reinforce with entropy bonus.

    dist:      actor distribution on states 0..H-1, batch shape (N, H), event (A,)
    action:    (N, H, A) the imagined actions taken in those states
    advantage: (N, H) (R_t - v(s_t)) / max(1, S)
    weight:    (N, H) continue weights w_t
    per step:  -sg(w_t) * (sg(adv_t) * log pi(a_t | s_t) + entropy_coef * H[pi(.|s_t)])
    log pi and entropy are summed over action dims. Returns a scalar: mean over N*H.
    """
    log_prob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    loss = -weight.detach() * ( advantage.detach() * log_prob + entropy_coef * entropy)
    loss = loss.mean()
    return loss
