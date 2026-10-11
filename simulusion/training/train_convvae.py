"""Training script for Convolutional Variational Autoencoder (ConvVAE).

This module provides functions to train a ConvVAE model on observation data
for world model learning.
"""

import os
from os.path import join, exists
from typing import Any, Optional, cast

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
import albumentations as A
from simulusion.datasets.wm_dataset import ObservationDataset
from simulusion.vision.VAE.ConvVAE import ConvVAE
from simulusion.configs.wm_config import WMVAEConfig
from simulusion.losses.convae_loss import conv_vae_loss_fn
from simulusion.utils.train_utils import EarlyStopping, ReduceLROnPlateau


def save_checkpoint(
    state: dict, is_best: bool, filename: str, best_filename: str
) -> None:
    """Save model checkpoint.

    Args:
        state: Dictionary containing model state to save.
        is_best: If True, also save as best checkpoint.
        filename: Path to save checkpoint.
        best_filename: Path to save best checkpoint.
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def test_epoch(
    model: ConvVAE,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: Any,
) -> float:
    """Run one epoch of validation.

    Args:
        model: The VAE model to evaluate.
        test_loader: DataLoader for test/validation data.
        device: Device to run evaluation on.
        loss_fn: Loss function to use.

    Returns:
        float: Average test loss for the epoch.
    """
    model.eval()
    test_loss = 0.0
    total_batches = len(test_loader)
    print(
        f"Test epoch: {total_batches} batches, dataset size: {len(test_loader.dataset)}"  # type: ignore[arg-type]
    )

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = loss_fn(recon, data, mu, logvar)
            test_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Test batch {batch_idx}/{total_batches}, loss: {loss.item():.4f}"
                )

    test_loss /= len(test_loader.dataset)  # type: ignore[arg-type]
    print("---> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def train_epoch(
    epoch: int,
    model: Any,
    optimizer: Any,
    train_loader: Any,
    device: Any,
    train_dataset: Any,
    loss_fn: Any,
    use_amp: bool = False,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> float:
    """Run one epoch of training.

    Args:
        epoch: Current epoch number.
        model: The VAE model to train.
        optimizer: Optimizer for training.
        train_loader: DataLoader for training data.
        device: Device to run training on.
        train_dataset: Training dataset (used to load next buffer if applicable).
        loss_fn: Loss function to use.
        use_amp: Whether to use automatic mixed precision.
        scaler: GradScaler for mixed precision training.
    """
    model.train()
    train_loss = 0.0
    if hasattr(train_dataset, "load_next_buffer"):
        train_dataset.load_next_buffer()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        if use_amp:
            assert scaler is not None
            with torch.cuda.amp.autocast():
                reconst, mu, logvar = model(data)
                loss = loss_fn(reconst, data, mu, logvar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconst, mu, logvar = model(data)
            loss = loss_fn(reconst, data, mu, logvar)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

        if batch_idx % 20 == 0:
            print(
                f"train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tloss: {loss.item() / len(data):.6f}"
            )

    print(
        "---> Epoch: {} Average Loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )
    return train_loss


def train_convae(config: WMVAEConfig) -> None:
    """Train a Convolutional VAE model.

    This function trains a ConvVAE on observation data using the provided
    configuration. It handles data loading, model initialization, training
    loop, checkpointing, and sample generation.

    Args:
        config: WMVAEConfig object containing all training hyperparameters.

    The training process includes:
        - Loading pretrained VAE if available (unless noreload is True)
        - Training for specified number of epochs
        - Validating after each epoch
        - Learning rate scheduling with ReduceLROnPlateau
        - Early stopping based on validation loss
        - Checkpointing best and current models
        - Generating sample images at specified intervals

    Example:
        >>> config = WMVAEConfig(
        ...     height=64, width=64, latent_size=32, num_epochs=100, logdir='results'
        ... )
        >>> train_convae(config)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")

    print(f"Using device: {device}")
    model = ConvVAE(img_channels=3, latent_size=config.latent_size).to(device)

    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile for faster training...")
        model = cast(ConvVAE, torch.compile(model, mode="reduce-overhead"))

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    earlystopping = EarlyStopping("min", patience=config.early_stopping_patience)

    RED_SIZE = config.height

    vae_dir = join(config.logdir, "vae") if hasattr(config, "logdir") else "vae"
    if not exists(vae_dir):
        os.makedirs(vae_dir)
        os.makedirs(join(vae_dir, "samples"))

    if not config.noreload:
        reload_file = join(vae_dir, "best.tar")
        if exists(reload_file):
            state = torch.load(reload_file, weights_only=True)
            print(
                "Reloading model at epoch {}, with test error {}".format(
                    state["epoch"], state["precision"]
                )
            )
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            earlystopping.load_state_dict(state["earlystopping"])

    transform_train = A.Compose(
        [
            A.Resize(height=RED_SIZE, width=RED_SIZE),
            A.HorizontalFlip(p=0.5),
        ]
    )

    transform_test = A.Compose(
        [
            A.Resize(height=RED_SIZE, width=RED_SIZE),
        ]
    )

    train_dataset = ObservationDataset(
        root=config.data_dir, transform=transform_train, train=True
    )
    num_workers = 0
    pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataset = ObservationDataset(
        root=config.data_dir, transform=transform_test, train=False
    )
    if len(test_dataset) == 0:
        print("WARNING: Test dataset is empty! Using training data for validation.")
        test_loader = train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.train_batch_size,
            shuffle=False,
            drop_last=False,
        )
    print(f"Test dataset size: {len(test_dataset)}, batches: {len(test_loader)}")

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    cur_best = None

    for ep in range(1, config.num_epochs + 1):
        print(f"\n=== Starting Epoch {ep}/{config.num_epochs} ===")
        train_epoch(
            ep,
            model,
            optimizer,
            train_loader,
            device,
            train_dataset,
            conv_vae_loss_fn,
            use_amp=use_amp,
            scaler=scaler,
        )
        print("Training complete, starting validation...")
        test_loss = test_epoch(model, test_loader, device, conv_vae_loss_fn)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        best_filename = join(vae_dir, "best.tar")
        filename = join(vae_dir, "checkpoint.tar")
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint(
            {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "precision": test_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "earlystopping": earlystopping.state_dict(),
            },
            is_best,
            filename,
            best_filename,
        )

        if ep % config.sample_interval == 0 and not config.nosamples:
            with torch.no_grad():
                sample = torch.randn(16, config.latent_size).to(device)
                sample = model.decoder(sample).cpu()
                save_image(
                    sample.view(16, 3, RED_SIZE, RED_SIZE),
                    join(vae_dir, "samples", f"sample_epoch_{ep}.png"),
                )

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(ep))
            break

    torch.save(model.state_dict(), join(vae_dir, "convae_final.pth"))
