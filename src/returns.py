"""Lambda-return, replay target and return scale. Written by Adam.

Index convention (TD3 style): r_t and c_t result from taking a_t in state t,
and bootstraps use v_{t+1}.
"""

import torch


def lambda_return(reward: torch.Tensor, cont: torch.Tensor, value: torch.Tensor,
                  gamma: float, lam: float) -> torch.Tensor:
    """R_t = r_t + gamma * c_t * ((1 - lam) * v_{t+1} + lam * R_{t+1}),  R_H = v_H.

    reward: (N, H)    r_t for t = 0..H-1
    cont:   (N, H)    c_t for t = 0..H-1
    value:  (N, H+1)  v_t for t = 0..H
    Returns (N, H): R_t for t = 0..H-1. Computed backward in time from t = H-1.

    In imagination, r_t and c_t come from the reward/continue heads at state
    t+1 (the first state that has seen a_t); the caller does that alignment.
    """
    H = reward.shape[-1]
    returns = []

    R_tplus1 = value[:, -1]
    for i in range(H-1, -1, -1):
        r_t = reward[:, i]
        c_t = cont[:, i]
        v_tplus1 = value[:, i+1]
        R_t = r_t + gamma * c_t * ((1-lam)*v_tplus1 + lam * R_tplus1)
        returns.append(R_t)
        R_tplus1 = R_t

    return torch.stack(returns[::-1], dim=1)



def replay_return(reward: torch.Tensor, cont: torch.Tensor, img_return: torch.Tensor,
                  is_first: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """R^rep_t = r_t + gamma * c_t * ((1 - lam) * R^img_{t+1} + lam * R^rep_{t+1}).

    reward, cont: (B, T) real r_t, c_t from replay
    img_return:   (B, T) R^img_t, the imagination return of the rollout that
                  started at real state t
    is_first:     (B, T) bool
    Returns (B, T-1): R^rep_t for t = 0..T-2. Computed backward in time.

    Start: R^rep_{T-1} = R^img_{T-1} (no step T in the chunk); loop t = T-2..0.
    Boundary: where is_first[:, t+1] is True, t+1 is a new episode, so restart
    with R^rep_t = R^img_t (treat it like the end of the chunk).
    The replay loss masks those steps out: valid = ~is_first[:, 1:].
    """

    T = reward.shape[-1]

    returns = []

    R_rep_tplus1 = img_return[:, -1]
    for i in range(T-2, -1, -1):
        r_t = reward[:, i]
        c_t = cont[:, i]
        r_img_tplus1 = img_return[:, i+1]
        R_t = r_t + gamma * c_t * ((1-lam)*r_img_tplus1 + lam * R_rep_tplus1)
        R_t = torch.where(is_first[:, i+1], img_return[:, i], R_t)
        returns.append(R_t)
        R_rep_tplus1 = R_t

    return torch.stack(returns[::-1], dim=1)
    


class ReturnScale:
    """S = EMA(Per(R, 95) - Per(R, 5)), decay 0.99. Divide advantages by max(limit, S).

    Example: spread 20 with previous S = 19 -> 0.99 * 19 + 0.01 * 20 = 19.01.
    A spread of 0.3 gives max(1, 0.3) = 1.
    """

    def __init__(self, decay: float = 0.99, low: float = 5.0, high: float = 95.0,
                 limit: float = 1.0):

        self.decay = decay
        self.low = low/100.0
        self.high = high/100.0
        self.limit = limit
        self.spread = None

    def update(self, returns: torch.Tensor) -> None:
        """Update the EMA from a batch of lambda-returns (any shape). No gradient."""
        with torch.no_grad():
            per_high = torch.quantile(returns.flatten(), self.high)
            per_low = torch.quantile(returns.flatten(), self.low)
            spread = (per_high - per_low).item()
            if self.spread is not None:
                self.spread = self.decay * self.spread + (1 - self.decay) * spread
            else:
                self.spread = spread
            


    def scale(self) -> float:
        """max(limit, S): the divisor for the advantages."""
        if self.spread is None:
            return self.limit
        else:
            return max(self.limit, self.spread)
        
