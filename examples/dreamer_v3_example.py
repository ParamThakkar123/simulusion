"""
Example script for training a DreamerV3 agent.

DreamerV3 is designed to run with fixed hyperparameters across domains, so this
script deliberately exposes very few knobs: the environment, the budget, and the
model size. Everything else is left at the paper defaults.

Usage::

    # Runs on a base `pip install simulusion[gym]`.
    python examples/dreamer_v3_example.py env=Pendulum-v1 env_backend=gym

    # DeepMind Control, larger model (requires `pip install simulusion[dmc]`).
    python examples/dreamer_v3_example.py env=walker-walk model_size=200m

    # Atari, discrete actions handled automatically.
    python examples/dreamer_v3_example.py env=ALE/Pong-v5 env_backend=gym
"""

import logging

from omegaconf import OmegaConf

from simulusion.configs.dreamer_v3_config import MODEL_SIZES, DreamerV3Config
from simulusion.models.dreamer_v3 import DreamerV3Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    cli_cfg = OmegaConf.from_cli()

    env = cli_cfg.get("env", "Pendulum-v1")
    env_backend = cli_cfg.get("env_backend", "gym")
    total_steps = int(cli_cfg.get("total_steps", 100_000))
    model_size = str(cli_cfg.get("model_size", "12m"))
    seed = int(cli_cfg.get("seed", 0))
    logdir = cli_cfg.get("logdir", None)
    device = cli_cfg.get("device", "auto")

    if model_size not in MODEL_SIZES:
        raise SystemExit(
            f"Unknown model_size={model_size!r}. "
            f"Choose one of: {', '.join(MODEL_SIZES)}"
        )

    config = DreamerV3Config(
        env=env,
        env_backend=env_backend,
        total_steps=total_steps,
        model_size=model_size,
        seed=seed,
    )
    if device != "auto":
        config.no_gpu = device == "cpu"

    logger.info("Training DreamerV3 on %s (%s backend)", env, env_backend)
    logger.info(
        "Model size %s: hidden=%s recurrent=%s latent=%sx%s",
        model_size,
        config.resolved_hidden_size,
        config.resolved_recurrent_units,
        config.latent_dim,
        config.resolved_latent_classes,
    )
    logger.info(
        "Replay ratio %s -> %s gradient steps per %s collected steps",
        config.replay_ratio,
        config.update_steps,
        config.collect_steps,
    )

    agent = DreamerV3Agent(config, logdir=logdir)
    logger.info("Parameters: %s", f"{agent.parameter_count():,}")

    agent.train(total_steps=total_steps)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
