"""Save and restore training state (model only; the replay buffer is not saved).

A checkpoint holds everything needed to continue learning where it stopped:
all networks (world model, actor, critic, slow critic), the three optimizers'
state (LaProp v, m, step counts), the return scale S, the env step counter,
and the random number generator states.

On resume the replay buffer starts empty, so training continues after a short
warm-up refill (see train.py).
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

OPTIMIZERS = ("wm_opt", "actor_opt", "critic_opt")


def save_checkpoint(path: str | Path, agent, step: int) -> None:
    """Write the agent's full training state to `path` (atomically: tmp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "agent": agent.state_dict(),
        "optimizers": {name: getattr(agent, name).state_dict() for name in OPTIMIZERS},
        "return_scale": agent.return_scale.spread,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # a crash mid-save never leaves a half-written checkpoint


def load_checkpoint(path: str | Path, agent, device: torch.device) -> int:
    """Restore the agent's training state from `path`. Returns the saved env step."""
    state = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(state["agent"])
    for name in OPTIMIZERS:
        getattr(agent, name).load_state_dict(state["optimizers"][name])
    agent.return_scale.spread = state["return_scale"]
    rng = state["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    return state["step"]


def latest_checkpoint(directory: str | Path) -> Path | None:
    """The checkpoint file 'latest.pt' in `directory`, if it exists."""
    path = Path(directory) / "latest.pt"
    return path if path.exists() else None
