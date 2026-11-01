"""Run DreamerV1 on a Gymnasium Robotics environment.

Usage examples:
  python examples/run_dreamer_robotics.py env=HalfCheetah-v2 total_steps=2000
  python examples/run_dreamer_robotics.py list_envs=true
  python examples/run_dreamer_robotics.py env=FetchReach-v3 evaluate=true
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)


def make_config(cli_cfg):
    from simulusion.configs.dreamer_config import DreamerConfig

    cfg = DreamerConfig()
    cfg.env_backend = "robotics"
    cfg.env = cli_cfg.env
    cfg.gym_render_mode = "rgb_array"
    cfg.total_steps = int(cli_cfg.get("total_steps", 5000))
    cfg.seed = int(cli_cfg.get("seed", 1))
    cfg.no_gpu = cli_cfg.get("device", "auto") == "cpu"
    cfg.algo = "Dreamerv1"

    cfg.seed_steps = int(cli_cfg.get("seed_steps", 500))
    cfg.collect_steps = int(cli_cfg.get("collect_steps", 200))
    cfg.update_steps = int(cli_cfg.get("update_steps", 10))
    cfg.batch_size = int(cli_cfg.get("batch_size", 16))
    img_size = int(cli_cfg.get("image_size", 64))
    cfg.image_size = (img_size, img_size)
    cfg.action_repeat = int(cli_cfg.get("action_repeat", 2))
    cfg.time_limit = int(cli_cfg.get("time_limit", 1000))
    cfg.log_video_freq = -1
    cfg.test_interval = max(1, int(cli_cfg.get("test_interval", 1000)))
    cfg.test_episodes = int(cli_cfg.get("test_episodes", 5))
    return cfg


def main() -> None:
    cli_cfg = OmegaConf.from_cli()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("simulusion").setLevel(logging.INFO)

    if cli_cfg.get("list_envs", False):
        from simulusion.envs.robotics_env import list_gymnasium_robotics_envs

        env_ids = list_gymnasium_robotics_envs()
        if not env_ids:
            raise SystemExit(
                "No Gymnasium Robotics environments found. Install with: "
                "pip install 'simulusion[robotics]'"
            )
        for env_id in env_ids:
            print(env_id)
        return

    from simulusion.models.dreamer import DreamerAgent

    cfg = make_config(cli_cfg)
    LOGGER.info("Running DreamerV1 on Gymnasium Robotics env='%s'", cfg.env)

    agent = DreamerAgent(cfg, logdir=cli_cfg.get("logdir", None))
    if cli_cfg.get("evaluate", False):
        LOGGER.info("Starting evaluation...")
        agent.evaluate()
    else:
        LOGGER.info("Starting training...")
        agent.train(total_steps=cfg.total_steps)


if __name__ == "__main__":
    main()
