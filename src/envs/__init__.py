"""Environment construction from the config."""


def make_envs(cfg):
    """cfg.env.name is '<suite>:<task>', e.g. 'gym:Pendulum-v1' or 'dmc:walker_walk'.

    A name without a prefix is treated as a Gymnasium id.
    """
    suite, _, task = cfg.env.name.rpartition(":")
    if suite == "dmc":
        from src.envs.dmc_env import DMCEnvs
        return DMCEnvs(task, cfg.env.num_envs, seed=cfg.seed,
                       action_repeat=cfg.env.action_repeat)
    if suite in ("", "gym"):
        from src.envs.gym_env import GymEnvs
        return GymEnvs(task, cfg.env.num_envs, seed=cfg.seed)
    raise ValueError(f"unknown env suite '{suite}' in {cfg.env.name}")
