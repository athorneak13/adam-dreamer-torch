"""Metric logging: prints to the terminal, and optionally sends to Weights & Biases.

Usage in train.py:
    logger = Logger(cfg)
    logger.episode(step, episode_return, episode_length)   # when an episode ends
    logger.metrics(step, metrics)                           # train_step metrics
    logger.close()

W&B is used only when cfg.logging.wandb is true. Run `wandb login` once on the
machine first. If W&B can't start, the logger keeps printing and says so.
"""

from types import SimpleNamespace


def _to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


class Logger:
    def __init__(self, cfg):
        self.run = None
        log_cfg = getattr(cfg, "logging", None)
        if log_cfg is not None and getattr(log_cfg, "wandb", False):
            try:
                import wandb
                self.run = wandb.init(
                    project=log_cfg.project,
                    name=getattr(log_cfg, "run_name", None),
                    config=_to_dict(cfg),
                    resume="allow",
                    id=getattr(log_cfg, "run_id", None),
                )
            except Exception as e:  # keep training even if W&B is unavailable
                print(f"W&B disabled: {e}")
                self.run = None

    def episode(self, step: int, episode_return: float, episode_length: float) -> None:
        print(f"step {step:>7d} | episode return {episode_return:8.1f}")
        if self.run is not None:
            self.run.log({"episode/return": episode_return,
                          "episode/length": episode_length}, step=step)

    def metrics(self, step: int, metrics: dict) -> None:
        print(f"step {step:>7d} | rec {metrics['rec']:.3f}  rew {metrics['rew']:.3f}  "
              f"value {metrics['value']:.3f}  ret {metrics['ret']:.2f}  "
              f"entropy {metrics['entropy']:.2f}  scale {metrics['ret_scale']:.2f}")
        if self.run is not None:
            self.run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
