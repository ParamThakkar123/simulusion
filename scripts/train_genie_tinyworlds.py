#!/usr/bin/env python3
"""Training script for Genie on TinyWorlds HDF5 dataset."""

import dataclasses
import os
import torch
from omegaconf import OmegaConf

from simulusion.configs.genie_config import GenieSmallConfig
from simulusion.training.train_genie import create_genie_trainer
from simulusion.datasets import create_tinyworlds_dataloader


def main():
    cli_cfg = OmegaConf.from_cli()

    dataset = cli_cfg.get("dataset", "SONIC")
    num_frames = int(cli_cfg.get("num_frames", 16))
    image_size = int(cli_cfg.get("image_size", 64))
    batch_size = int(cli_cfg.get("batch_size", 2))
    num_workers = int(cli_cfg.get("num_workers", 4))
    max_steps = int(cli_cfg.get("max_steps", 50000))
    log_interval = int(cli_cfg.get("log_interval", 100))
    val_interval = int(cli_cfg.get("val_interval", 1000))
    learning_rate = float(cli_cfg.get("learning_rate", 1e-4))
    cache_dir = cli_cfg.get("cache_dir", None)
    data_file = cli_cfg.get("data_file", None)
    checkpoint_dir = cli_cfg.get("checkpoint_dir", "checkpoints")
    device_str = cli_cfg.get("device", None)

    device = (
        torch.device(device_str)
        if device_str
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Using device: {device}")

    early_stopping = bool(cli_cfg.get("early_stopping", False))
    patience = int(cli_cfg.get("patience", 10))
    min_delta = float(cli_cfg.get("min_delta", 1e-4))
    val_split = float(cli_cfg.get("val_split", 0.1 if early_stopping else 0.0))

    checkpoint_interval = int(cli_cfg.get("checkpoint_interval", 0))

    config = GenieSmallConfig()
    config.num_frames = num_frames
    config.image_size = image_size
    config.batch_size = batch_size
    config.max_steps = max_steps
    config.learning_rate = learning_rate
    config.early_stopping = early_stopping
    config.patience = patience
    config.min_delta = min_delta
    config.val_split = val_split

    # Every remaining GenieSmallConfig field is settable from the CLI, so the
    # architecture can be shrunk as well as the schedule. Without this the
    # smallest run possible was still the full 462M-parameter model, which does
    # not fit on a small GPU no matter how low batch_size goes.
    handled = {
        "dataset",
        "num_frames",
        "image_size",
        "batch_size",
        "num_workers",
        "max_steps",
        "log_interval",
        "val_interval",
        "learning_rate",
        "cache_dir",
        "data_file",
        "checkpoint_dir",
        "checkpoint_interval",
        "device",
        "early_stopping",
        "patience",
        "min_delta",
        "val_split",
    }
    fields = {f.name: f.type for f in dataclasses.fields(config)}
    for key, value in cli_cfg.items():
        if key in handled:
            continue
        if key not in fields:
            raise SystemExit(
                f"unknown option '{key}'. GenieSmallConfig fields: "
                + ", ".join(sorted(fields))
            )
        current = getattr(config, key)
        # OmegaConf hands back str for everything on the CLI; coerce to the
        # type the field already holds so the dataclass stays well typed.
        if isinstance(current, bool):
            value = str(value).lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(config, key, value)
        print(f"  config override: {key}={value}")

    print(f"Loading {dataset} dataset...")
    loader_kwargs = dict(
        dataset_name=dataset,
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        download=not data_file,
        data_file=data_file,
        val_split=val_split,
    )
    train_dataset, train_loader = create_tinyworlds_dataloader(
        shuffle=True, split="train", **loader_kwargs
    )

    val_loader = None
    if early_stopping:
        # Same val_split and seed, so this is the disjoint other half.
        _, val_loader = create_tinyworlds_dataloader(
            shuffle=False, split="val", **loader_kwargs
        )

    print(f"Dataset: {len(train_dataset)} samples, {len(train_loader)} batches")

    print("Creating Genie model and trainer...")
    trainer, model = create_genie_trainer(config=config, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_steps=max_steps,
        log_interval=log_interval,
        val_interval=val_interval,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/genie_{dataset.lower()}_final.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
