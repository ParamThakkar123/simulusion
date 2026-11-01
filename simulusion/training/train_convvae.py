"""Training script for Convolutional Variational Autoencoder (ConvVAE).

This module provides functions to train a ConvVAE model on observation data
for world model learning.
"""

from os.path import join

import torch
import torch.utils.data
from simulusion.configs.wm_config import WMVAEConfig


def train_convae(config: WMVAEConfig) -> None:
    """Train a Convolutional VAE model.

    Runs on the shared :class:`~simulusion.training.Trainer` through the
    ``vae`` task, on the rollouts in ``config.data_dir`` (``.npz`` files with
    ``observations``, or any episode folder). As before, everything goes to
    ``<logdir>/vae``:

        - ``checkpoint.tar`` every epoch and ``best.tar`` for the lowest
          held-out loss (the last 600 rollouts are held out when there are
          more), resumed from unless ``noreload`` is set
        - ``ReduceLROnPlateau`` and early stopping on that loss
        - ``samples/sample_epoch_<N>.png`` every ``sample_interval`` epochs
          unless ``nosamples``
        - ``convae_final.pth``, the final weights

    The trainer adds ``checkpoints/``, ``run.json``, ``run_config.yaml`` and
    ``train_metrics.jsonl``.

    Example:
        >>> config = WMVAEConfig(
        ...     height=64, width=64, latent_size=32, num_epochs=100, logdir='results'
        ... )
        >>> train_convae(config)
    """
    from simulusion.training.legacy import fit_legacy
    from simulusion.training.tasks.world_models import VAETask, VAETaskConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available, using CPU")
    vae_dir = join(config.logdir, "vae")
    task = VAETask(VAETaskConfig(model=config), legacy_dir=vae_dir)
    fit_legacy(
        task,
        vae_dir,
        device=device,
        # What use_amp selected here: fp16 with loss scaling on CUDA.
        precision="fp16" if device.type == "cuda" else "fp32",
        eval_every=1,
        checkpoint_every=1,
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
        patience=config.early_stopping_patience,
    )
    torch.save(task.vae.state_dict(), join(vae_dir, "convae_final.pth"))
