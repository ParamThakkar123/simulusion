"""Run DreamerV1 on a Procgen environment (small example).

Usage examples:
  python examples/run_dreamer_procgen.py env=coinrun total_steps=2000
  python examples/run_dreamer_procgen.py env=heist distribution_mode=hard
  python examples/run_dreamer_procgen.py list_envs=true
"""

import logging
from omegaconf import OmegaConf

from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.envs.procgen_env import list_procgen_envs
from simulusion.models.dreamer import DreamerAgent


def make_config(cli_cfg) -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.env_backend = "procgen"
    cfg.env = cli_cfg.env
    cfg.total_steps = int(cli_cfg.get("total_steps", 5000))
    cfg.seed = int(cli_cfg.get("seed", 1))
    cfg.no_gpu = cli_cfg.get("device", "auto") == "cpu"
    cfg.algo = "Dreamerv1"

    cfg.procgen_distribution_mode = cli_cfg.get("distribution_mode", "easy")
    cfg.procgen_num_levels = int(cli_cfg.get("num_levels", 0))
    cfg.procgen_start_level = cli_cfg.get("start_level", None)

    cfg.seed_steps = int(cli_cfg.get("seed_steps", 500))
    cfg.collect_steps = int(cli_cfg.get("collect_steps", 200))
    cfg.update_steps = int(cli_cfg.get("update_steps", 10))
    cfg.batch_size = int(cli_cfg.get("batch_size", 16))
    cfg.image_size = (64, 64)
    cfg.log_video_freq = -1
    cfg.test_interval = max(1, int(cli_cfg.get("test_interval", 1000)))
    cfg.test_episodes = int(cli_cfg.get("test_episodes", 5))
    return cfg


def main():
    cli_cfg = OmegaConf.from_cli()

    if cli_cfg.get("list_envs", False):
        for env_name in list_procgen_envs():
            print(env_name)
        return

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("simulusion").setLevel(logging.INFO)

    cfg = make_config(cli_cfg)
    logging.info(
        "Running DreamerV1 on procgen env='%s' (distribution=%s, levels=%s)",
        cfg.env,
        cfg.procgen_distribution_mode,
        cfg.procgen_num_levels,
    )

    agent = DreamerAgent(cfg, logdir=cli_cfg.get("logdir", None))
    if cli_cfg.get("evaluate", False):
        logging.info("Starting evaluation...")
        agent.evaluate()
    else:
        logging.info("Starting training...")
        agent.train(total_steps=cfg.total_steps)


if __name__ == "__main__":
    main()
