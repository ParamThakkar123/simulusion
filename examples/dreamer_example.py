"""
Example script for training a Dreamer agent.

This demonstrates how to use the DreamerAgent class for end-to-end training
of a world model-based reinforcement learning agent.

DeepMind Control tasks use ``domain-task`` names and need ``pip install
simulusion[dmc]``. For a quick run without simulator downloads, use a Gymnasium
task instead::

    python examples/dreamer_example.py --env cartpole-balance
    python examples/dreamer_example.py --env Pendulum-v1 --env-backend gym
"""

import argparse
import logging

from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.models.dreamer import DreamerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dreamer agent")
    parser.add_argument("--env", default="cartpole-balance")
    parser.add_argument("--env-backend", default="dmc")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--seed-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="auto/gpu use CUDA or Apple MPS when available",
    )
    args = parser.parse_args()

    logger.info("Training Dreamer on %s (%s)", args.env, args.env_backend)
    logger.info("Total steps: %s", args.total_steps)

    config = DreamerConfig()
    config.env = args.env
    config.env_backend = args.env_backend
    config.total_steps = args.total_steps
    config.seed_steps = args.seed_steps
    config.seed = args.seed
    config.no_gpu = args.device == "cpu"

    agent = DreamerAgent(config, logdir=args.logdir)
    agent.train(total_steps=args.total_steps)

    logger.info("Training completed! Final checkpoint is in %s/ckpts", agent.logdir)


if __name__ == "__main__":
    main()
