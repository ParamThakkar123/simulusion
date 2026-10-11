"""Run DreamerV1 on a Brax environment (small example).

Usage examples:
  python examples/run_dreamer_brax.py env=ant total_steps=2000
  python examples/run_dreamer_brax.py env=walker evaluate=true
"""

import logging
from omegaconf import OmegaConf

from simulusion.models.dreamer import DreamerAgent
from simulusion.configs.dreamer_config import DreamerConfig


def make_config(cli_cfg) -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.env_backend = "brax"
    cfg.env = cli_cfg.env
    cfg.total_steps = int(cli_cfg.get("total_steps", 5000))
    cfg.seed = int(cli_cfg.get("seed", 1))
    cfg.no_gpu = cli_cfg.get("device", "auto") == "cpu"
    cfg.brax_jit = cli_cfg.get("brax_jit", False)
    cfg.brax_auto_reset = cli_cfg.get("brax_auto_reset", False)
    cfg.algo = "Dreamerv1"
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
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("simulusion").setLevel(logging.INFO)

    cfg = make_config(cli_cfg)
    logging.info(f"Running DreamerV1 on brax env='{cfg.env}' (jit={cfg.brax_jit})")

    agent = DreamerAgent(cfg, logdir=cli_cfg.get("logdir", None))

    if cli_cfg.get("evaluate", False):
        logging.info("Starting evaluation...")
        agent.evaluate()
    else:
        logging.info("Starting training...")
        agent.train(total_steps=cfg.total_steps)


if __name__ == "__main__":
    main()
