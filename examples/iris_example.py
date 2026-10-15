"""
Example script for training an IRIS agent on Atari.

IRIS (Micheli et al., ICLR 2023) learns a discrete autoencoder and an
autoregressive Transformer world model, then trains its actor-critic entirely
in imagination.

Requires the Gym extra for Atari: ``pip install simulusion[gym]``.

Usage::

    python examples/iris_example.py --game ALE/Pong-v5 --epochs 5
"""

import argparse
import logging

from simulusion.configs.iris_config import IRISConfig
from simulusion.training.train_iris import IRISTrainer
from simulusion.utils.device import default_device_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an IRIS agent on Atari")
    parser.add_argument("--game", default="ALE/Pong-v5", help="Atari environment id")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the autoencoder, transformer and actor-critic batch sizes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default=None, help="cuda, mps or cpu (default: best available)"
    )
    parser.add_argument("--save-dir", default="checkpoints/iris")
    args = parser.parse_args()

    config = IRISConfig(env=args.game, total_epochs=args.epochs)
    if args.batch_size is not None:
        config.autoencoder_batch_size = args.batch_size
        config.transformer_batch_size = args.batch_size
        config.actor_critic_batch_size = args.batch_size

    device = args.device or default_device_name()
    logger.info(
        "Training IRIS on %s for %d epochs (%s)", args.game, args.epochs, device
    )

    trainer = IRISTrainer(game=args.game, device=device, seed=args.seed, config=config)
    trainer.train(total_epochs=args.epochs, save_dir=args.save_dir)

    logger.info("Training completed! Checkpoints are in %s", args.save_dir)


if __name__ == "__main__":
    main()
