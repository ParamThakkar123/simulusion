"""
Example script for running the RL harness with vectorized simulators.
This demonstrates the implementation of vectorized simulator workers and RL training harness.
"""

import logging
from functools import partial
from omegaconf import OmegaConf
from simulusion.training.rl_harness import PPOTrainer
from simulusion.envs.vector_env import TorchVectorizedEnv
from simulusion.envs.gym_env import make_gym_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    cli_cfg = OmegaConf.from_cli()

    env = cli_cfg.get("env", "CartPole-v1")
    num_workers = int(cli_cfg.get("num_workers", 2))
    envs_per_worker = int(cli_cfg.get("envs_per_worker", 4))
    total_timesteps = int(cli_cfg.get("total_timesteps", 50000))
    device = cli_cfg.get("device", "cpu")
    seed = int(cli_cfg.get("seed", 42))

    logger.info(f"Starting RL harness with env: {env}")
    logger.info(f"Workers: {num_workers}, Envs per worker: {envs_per_worker}")
    logger.info(f"Total envs: {num_workers * envs_per_worker}")
    logger.info(f"Total timesteps: {total_timesteps}")

    env_factory = partial(make_gym_env, env=env, size=(64, 64))
    vec_env = TorchVectorizedEnv(
        env_factory=env_factory,
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
        seed=seed,
    )

    trainer = PPOTrainer(vec_env, device=device)
    trainer.train(total_timesteps=total_timesteps)

    logger.info("Training completed. Cleaning up...")
    vec_env.close()


if __name__ == "__main__":
    main()
