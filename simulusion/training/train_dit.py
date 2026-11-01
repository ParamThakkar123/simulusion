"""DiT training entrypoint for the Simulusion CLI.

``DiT.fit`` already carries the full training loop -- dataset construction,
the DDPM noise schedule, the EMA shadow model, checkpointing and a sample
grid. What was missing was the thin CLI layer every other model here has, so
``simulusion train dit`` had nothing to dispatch to and DiT was inference-only.

Usage:
    simulusion train dit epochs=50 batch=256
    python -m simulusion.training.train_dit EPOCHS=50 DATASET=cifar10
    python -m simulusion.training.train_dit --config my_dit.yaml --print-config

Overrides are accepted in either the config's own UPPER_CASE spelling or the
snake_case aliases (``epochs=50`` and ``EPOCHS=50`` are the same field).

CIFAR-10 downloads itself into ``ROOT_PATH``. ``imagenet`` and ``imagefolder``
read from ``ROOT_PATH`` instead and need the data to be there already.

Set ``EARLY_STOPPING=true`` to train until the held-out loss stops improving
rather than for a fixed number of epochs. ``EPOCHS`` then bounds the run
instead of defining it, and the checkpoint holds the best epoch's weights, not
the last:

    python -m simulusion.training.train_dit EPOCHS=2000 EARLY_STOPPING=true PATIENCE=20
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simulusion.configs.dit_config import DiTConfig, canonical_dit_key
from simulusion.experiments import (
    dump_config,
    instantiate_dataclass,
    parse_experiment_args,
)


def _canonicalize(overrides: Sequence[str]) -> list[str]:
    """Rewrite snake_case override keys to the config's UPPER_CASE fields."""
    canonical: list[str] = []
    for item in overrides:
        key, sep, value = item.partition("=")
        if sep:
            canonical.append(f"{canonical_dit_key(key.strip())}={value}")
        else:
            canonical.append(item)
    return canonical


def train_dit(config: DiTConfig | None = None, **kwargs: Any) -> DiTConfig:
    """Train a DiT on the configured dataset and write a checkpoint.

    Args:
        config: Training configuration. Defaults to :class:`DiTConfig`.
        **kwargs: Field overrides applied on top of ``config``, in either
            spelling.

    Returns:
        The configuration actually used, so callers can see the resolved values
        (and locate ``WORKDIR``) without re-deriving them.
    """
    # Imported here, not at module scope: the CLI lists trainers without
    # importing torch, and DiT pulls in the whole diffusion stack.
    from simulusion.models.diffusion.DiT import DiT

    if config is None:
        config = DiTConfig()
    for key, value in kwargs.items():
        setattr(config, canonical_dit_key(key), value)

    DiT.fit(
        epochs=config.EPOCHS,
        dataset=config.DATASET,
        batch_size=config.BATCH,
        lr=config.LR,
        img_size=config.IMG_SIZE,
        channels=config.CHANNELS,
        patch=config.PATCH,
        width=config.WIDTH,
        depth=config.DEPTH,
        heads=config.HEADS,
        drop=config.DROP,
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        ema=config.EMA,
        ema_decay=config.EMA_DECAY,
        num_classes=config.NUM_CLASSES,
        class_dropout_prob=config.CLASS_DROPOUT_PROB,
        learn_sigma=config.LEARN_SIGMA,
        workdir=config.WORKDIR,
        root_path=config.ROOT_PATH,
        val_split=config.VAL_SPLIT,
        crop_size=config.CROP_SIZE,
        num_workers=config.NUM_WORKERS,
        early_stopping=config.EARLY_STOPPING,
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        checkpoint_every=config.CHECKPOINT_EVERY,
    )
    return config


def main(argv: list[str] | None = None) -> DiTConfig:
    """Compose the DiT config from YAML/dot-list overrides and launch training."""
    args = parse_experiment_args(argv, description="Train DiT")
    config = instantiate_dataclass(
        DiTConfig, args.config, _canonicalize(args.overrides)
    )
    if args.print_config:
        print(dump_config(config.__dict__))
        return config
    train_dit(config=config)
    return config


if __name__ == "__main__":
    main()
