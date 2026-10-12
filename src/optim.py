"""LaProp optimizer with adaptive gradient clipping and LR warmup. Written by Adam.

Matches the official DreamerV3 optimizer chain (per parameter, in this order):
    1. AGC:      g <- g / max(1, ||g|| / (agc * max(pmin, ||p||)))   (whole-tensor norms)
    2. RMS:      v <- beta2 * v + (1 - beta2) * g^2
                 u  = g / (sqrt(v / (1 - beta2^t)) + eps)
    3. Momentum: m <- beta1 * m + (1 - beta1) * u
                 p <- p - lr_t * m / (1 - beta1^t)
LR warmup: lr_t ramps linearly from 0 to lr over `warmup` steps, then stays constant.
Settings: lr 4e-5, beta1 0.9, beta2 0.999, eps 1e-20, agc 0.3, pmin 1e-3, warmup 1000.
"""

import torch


class LaProp(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 4e-5, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-20, agc: float = 0.3, pmin: float = 1e-3, warmup: int = 1000):
        """Store the settings as defaults; per-parameter state (step, v, m) is created lazily."""
        defaults = dict(lr=lr, betas=betas, eps=eps, agc=agc, pmin=pmin, warmup=warmup)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """One update for every parameter with a gradient: AGC -> RMS -> momentum -> lr_t.

        State per parameter, in self.state[p]: "step" (int), "v" and "m" (like p).
        """
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            agc = group["agc"]
            pmin = group["pmin"]
            warmup = group["warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(p)
                    state["m"] = torch.zeros_like(p)
                
                v, m = state["v"], state["m"]
                state["step"] += 1
                t = state["step"]
                agc_clip_(p.grad, p, agc, pmin)
                g = p.grad
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                u = g / ((v / (1 - beta2 ** t)).sqrt() + eps)
                m.mul_(beta1).add_(u, alpha=1 - beta1)
                p.add_(m, alpha=-warmup_lr(lr, t, warmup) / (1 - beta1 ** t))



def agc_clip_(grad: torch.Tensor, param: torch.Tensor, agc: float, pmin: float) -> None:
    """In place: scale grad so ||grad|| <= agc * max(pmin, ||param||). Whole-tensor norms."""
    grad_2norm = grad.norm()
    param_2norm = param.norm()

    agc_clamped_param_2norm = agc * torch.clamp(param_2norm, min=pmin)

    factor = 1 / torch.clamp(grad_2norm / agc_clamped_param_2norm, min=1.0)
    grad.mul_(factor)
    
    


def warmup_lr(lr: float, step: int, warmup: int) -> float:
    """Linear ramp from 0 to lr over `warmup` steps (step counts from 1), then lr."""
    if warmup == 0:
        return lr
    else:
        return lr * min(1.0, ((step-1.0)/warmup))
