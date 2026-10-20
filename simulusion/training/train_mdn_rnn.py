"""Training script for Mixture Density Recurrent Neural Network (MDRNN).

This module provides functions to train an MDRNN model for sequence prediction
in world models. The MDRNN predicts future latent states using a Gaussian
Mixture Model (GMM) based on current latent states and actions.
"""

from typing import Any, Optional
import os
from os.path import join, exists
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from tqdm import tqdm

from simulusion.configs.wm_config import WMVAEConfig, WMMDNRNNConfig
from simulusion.models.mdrnn import MDRNN
from simulusion.vision.VAE.ConvVAE import ConvVAE
from simulusion.losses.gmm_loss import gmm_loss
from simulusion.datasets.wm_dataset import SequenceDataset, LatentSequenceDataset
from simulusion.training.precision import GradientUpdate, MixedPrecision
from simulusion.utils.train_utils import EarlyStopping, ReduceLROnPlateau


def precompute_latents(vae_config: WMVAEConfig, mdrnn_config: WMMDNRNNConfig) -> None:
    """Pre-compute and save VAE latents to disk for memory-efficient RNN training.

    This function encodes all observations using the VAE and saves the latent
    representations to disk. This allows RNN training without keeping the VAE
    in GPU memory.

    Args:
        vae_config: WMVAEConfig for loading pretrained VAE.
        mdrnn_config: WMMDNRNNConfig containing latent_size and device settings.
    """
    latent_dir = join(mdrnn_config.data_dir, "latents")
    latent_file = join(latent_dir, f"latents_{mdrnn_config.latent_size}.npz")

    if exists(latent_file):
        print(f"Found pre-computed latents at {latent_file}, skipping encoding.")
        return

    os.makedirs(latent_dir, exist_ok=True)
    device = torch.device(mdrnn_config.device)

    vae_file = join(vae_config.logdir, "vae", "best.tar")
    assert exists(vae_file), "No trained VAE found. Train VAE first."

    print("Loading VAE for latent encoding...")
    vae_state = torch.load(vae_file, map_location=device, weights_only=True)
    vae = ConvVAE(img_channels=3, latent_size=mdrnn_config.latent_size).to(device)
    vae.load_state_dict(vae_state["state_dict"])
    vae.eval()

    import glob as glob_lib

    rollout_files = glob_lib.glob(join(mdrnn_config.data_dir, "*.npz"))
    print(f"Encoding {len(rollout_files)} rollout files...")

    all_latents: Any = []
    all_rewards: Any = []
    all_actions: Any = []
    all_terminals: Any = []

    with torch.no_grad():
        for fpath in tqdm(rollout_files):
            data = np.load(fpath, allow_pickle=False)
            observations = data["observations"]
            actions = data["actions"]
            rewards = data["rewards"]
            terminals = data["terminals"]

            batch_size = 64
            latents: Any = []
            for i in range(0, len(observations), batch_size):
                obs_batch = observations[i : i + batch_size]
                obs_tensor = torch.tensor(obs_batch).float() / 255.0
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                obs_tensor = F.interpolate(
                    obs_tensor, size=64, mode="bilinear", align_corners=True
                )
                obs_tensor = obs_tensor.to(device)

                mu, logsigma = vae.encoder(obs_tensor)
                z = mu + logsigma.exp() * torch.randn_like(logsigma)
                latents.append(z.cpu().numpy())

            latents = np.concatenate(latents, axis=0)
            all_latents.append(latents)
            all_actions.append(actions.astype(np.float32))
            all_rewards.append(rewards.astype(np.float32))
            all_terminals.append(terminals.astype(np.float32))

    all_latents = np.concatenate(all_latents, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_terminals = np.concatenate(all_terminals, axis=0)

    np.savez(
        latent_file,
        latents=all_latents,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
    )

    del vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"Saved pre-computed latents to {latent_file}")


def save_checkpoint(
    state: Any, is_best: bool, filename: str, best_filename: str
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


def to_latent(
    vae: ConvVAE,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    device: torch.device,
    red_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform observations to latent space using VAE encoder.

    This function encodes observations into the latent space using the VAE's
    encoder network. It applies the reparameterization trick to sample from
    the learned latent distribution.

    Args:
        vae: Trained VAE model with encoder.
        obs: Batch of current observations.
        next_obs: Batch of next observations.
        device: Device to run encoding on.
        red_size: Target size for resizing images (default: 64).

    Returns:
        Tuple of (latent_obs, latent_next_obs) tensors in latent space.
    """
    with torch.no_grad():
        obs, next_obs = [
            F.interpolate(
                x.view(-1, 3, 96, 96),
                size=red_size,
                mode="bilinear",
                align_corners=True,
            )
            for x in (obs, next_obs)
        ]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)
        ]

        latent_obs = obs_mu + obs_logsigma.exp() * torch.randn_like(obs_logsigma)
        latent_next_obs = next_obs_mu + next_obs_logsigma.exp() * torch.randn_like(
            next_obs_logsigma
        )
    return latent_obs, latent_next_obs


def get_loss(
    mdrnn: MDRNN,
    latent_obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    terminal: torch.Tensor,
    latent_next_obs: torch.Tensor,
    include_reward: bool,
    latent_size: int,
) -> dict[str, torch.Tensor]:
    """Compute MDRNN loss.

    Computes the combined loss for the MDRNN model:
    - GMM loss for next latent state prediction
    - BCE loss for terminal state prediction
    - MSE loss for reward prediction (if include_reward is True)

    Args:
        mdrnn: MDRNN model.
        latent_obs: Current latent observations.
        action: Actions taken.
        reward: Rewards received.
        terminal: Terminal state flags.
        latent_next_obs: Next latent observations (target).
        include_reward: Whether to include reward prediction in loss.
        latent_size: Size of latent space.

    Returns:
        Dictionary containing gmm, bce, mse, and total loss values.
    """
    latent_obs, action, reward, terminal, latent_next_obs = [
        arr.transpose(1, 0)
        for arr in [latent_obs, action, reward, terminal, latent_next_obs]
    ]

    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, terminal)

    if include_reward:
        mse = F.mse_loss(rs, reward)
        scale = latent_size + 2
    else:
        mse = torch.tensor(0.0, device=reward.device)
        scale = latent_size + 1

    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(
    epoch: int,
    mdrnn: Any,
    vae: Any,
    train_loader: Any,
    test_loader: Any,
    optimizer: Any,
    device: Any,
    include_reward: bool,
    test_every: int = 10,
    epochs: int = 1,
    use_amp: bool = False,
    scaler: Any = None,
    latent_size: int = 32,
    max_seq_len: int = 50,
    log_wandb: bool = False,
    prev_val_loss: float = 1e6,
    early_stop: Any = None,
    lr_scheduler: Any = None,
    batch_size: int = 50,
    train: bool = True,
    update: Optional[GradientUpdate] = None,
) -> float:
    """Run one epoch of training or validation.

    Args:
        epoch: Current epoch number.
        mdrnn: MDRNN model.
        vae: VAE model for encoding observations (None if using precomputed latents).
        train_loader: Training data loader.
        test_loader: Test/validation data loader.
        optimizer: Optimizer (used only for training).
        device: Device to run on.
        include_reward: Whether to include reward in loss.
        latent_size: Size of latent space.
        batch_size: Batch size.
        train: If True, run training pass; otherwise run validation.
        use_amp: If True, use automatic mixed precision.
        scaler: GradScaler for mixed precision training.
        update: Steps ``optimizer`` at its own precision (the shared Trainer
            passes one); without it one is built from ``use_amp`` and
            ``scaler``.

    Returns:
        float: Average loss for the epoch.
    """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    cum_loss: float = 0.0
    cum_gmm: float = 0.0
    cum_bce: float = 0.0
    cum_mse: float = 0.0

    if hasattr(loader.dataset, "load_next_buffer"):
        loader.dataset.load_next_buffer()

    use_precomputed = vae is None
    if update is None:
        update = GradientUpdate(
            optimizer, mdrnn.parameters(), MixedPrecision.from_use_amp(use_amp, device)
        )
        if scaler is not None:
            update.scaler = scaler  # keep the caller's loss scale across epochs
    precision = update.precision

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        if use_precomputed:
            latent_obs, action, reward, terminal, latent_next_obs = [
                arr.to(device) for arr in data
            ]
            batch_size = latent_obs.shape[0]
            seq_len = latent_obs.shape[1]
        else:
            obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]
            batch_size_current, seq_len = obs.shape[:2]
            obs = obs.view(batch_size_current * seq_len, 3, 96, 96)
            next_obs = next_obs.view(batch_size_current * seq_len, 3, 96, 96)
            latent_obs, latent_next_obs = to_latent(vae, obs, next_obs, device)
            latent_obs = latent_obs.view(batch_size_current, seq_len, -1)
            latent_next_obs = latent_next_obs.view(batch_size_current, seq_len, -1)

        if train:
            with precision.autocast():
                losses = get_loss(
                    mdrnn,
                    latent_obs,
                    action,
                    reward,
                    terminal,
                    latent_next_obs,
                    include_reward,
                    latent_size,
                )
            update(losses["loss"])
        else:
            with torch.no_grad(), precision.autocast():
                losses = get_loss(
                    mdrnn,
                    latent_obs,
                    action,
                    reward,
                    terminal,
                    latent_next_obs,
                    include_reward,
                    latent_size,
                )

        cum_loss += losses["loss"].item()
        cum_gmm += losses["gmm"].item()
        cum_bce += losses["bce"].item()
        mse_val = float(
            losses["mse"].item() if hasattr(losses["mse"], "item") else losses["mse"]
        )
        cum_mse += mse_val

        pbar.set_postfix_str(
            "loss={loss:10.6f} bce={bce:10.6f} gmm={gmm:10.6f} mse={mse:10.6f}".format(
                loss=cum_loss / (i + 1),
                bce=cum_bce / (i + 1),
                gmm=cum_gmm / max(latent_size, 1) / (i + 1),
                mse=cum_mse / (i + 1),
            )
        )
        pbar.update(batch_size)

    pbar.close()
    return cum_loss * batch_size / len(loader.dataset)


def train_mdn_rnn(
    vae_config: WMVAEConfig,
    mdrnn_config: WMMDNRNNConfig,
    use_precomputed_latents: bool = True,
    use_amp: bool = True,
) -> None:
    """Train an MDRNN model.

    This function trains an MDRNN on sequence data using the provided
    configurations. It loads a pretrained VAE for encoding observations
    into latent space, then trains the MDRNN to predict future latent
    states given current latent states and actions.

    Args:
        vae_config: WMVAEConfig for loading pretrained VAE.
        mdrnn_config: WMMDNRNNConfig containing MDRNN training hyperparameters.
        use_precomputed_latents: If True, use pre-encoded latents from disk.
        use_amp: If True, use automatic mixed precision for memory efficiency.

    The training process includes:
        - Loading pretrained VAE from vae_config.logdir
        - Training for specified number of epochs
        - Validating after each epoch
        - Learning rate scheduling with ReduceLROnPlateau
        - Early stopping based on validation loss
        - Checkpointing best and current models

    Example:
        >>> vae_config = WMVAEConfig(
        ...     height=64, width=64, latent_size=32, logdir='results'
        ... )
        >>> mdrnn_config = WMMDNRNNConfig(
        ...     latent_size=32, action_size=3, hidden_size=256,
        ...     gmm_components=5, logdir='results'
        ... )
        >>> train_mdn_rnn(vae_config, mdrnn_config)
    """
    device = torch.device(mdrnn_config.device)
    use_amp = use_amp and device.type == "cuda"

    latent_dir = join(mdrnn_config.data_dir, "latents")
    latent_file = join(latent_dir, f"latents_{mdrnn_config.latent_size}.npz")

    vae = None
    if use_precomputed_latents and exists(latent_file):
        print(f"Loading pre-computed latents from {latent_file}")
        latent_data = np.load(latent_file, allow_pickle=False)
        latents = latent_data["latents"]
        all_actions = latent_data["actions"]
        all_rewards = latent_data["rewards"]
        all_terminals = latent_data["terminals"]
    else:
        if use_precomputed_latents:
            print("Pre-computing latents...")
            precompute_latents(vae_config, mdrnn_config)
            latent_data = np.load(latent_file, allow_pickle=False)
            latents = latent_data["latents"]
            all_actions = latent_data["actions"]
            all_rewards = latent_data["rewards"]
            all_terminals = latent_data["terminals"]
        else:
            vae_file = join(vae_config.logdir, "vae", "best.tar")
            assert exists(vae_file), "No trained VAE in the logdir..."
            vae_state = torch.load(vae_file, map_location=device, weights_only=True)
            print(
                "Loading VAE at epoch {} with test error {}".format(
                    vae_state["epoch"], vae_state["precision"]
                )
            )
            vae = ConvVAE(img_channels=3, latent_size=mdrnn_config.latent_size).to(
                device
            )
            vae.load_state_dict(vae_state["state_dict"])
            vae.eval()

    rnn_dir = join(mdrnn_config.logdir, "mdrnn")
    if not exists(rnn_dir):
        os.makedirs(rnn_dir)

    mdrnn = MDRNN(
        latents=mdrnn_config.latent_size,
        actions=mdrnn_config.action_size,
        hiddens=mdrnn_config.hidden_size,
        gaussians=mdrnn_config.gmm_components,
    ).to(device)

    optimizer = optim.RMSprop(
        mdrnn.parameters(), lr=mdrnn_config.learning_rate, alpha=0.9
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=mdrnn_config.scheduler_factor,
        patience=mdrnn_config.scheduler_patience,
    )
    earlystopping = EarlyStopping("min", patience=mdrnn_config.early_stopping_patience)

    rnn_file = join(rnn_dir, "best.tar")
    if not mdrnn_config.noreload and exists(rnn_file):
        rnn_state = torch.load(rnn_file, map_location=device, weights_only=True)
        print(
            "Loading MDRNN at epoch {} with test error {}".format(
                rnn_state["epoch"], rnn_state["precision"]
            )
        )
        mdrnn.load_state_dict(rnn_state["state_dict"])
        optimizer.load_state_dict(rnn_state["optimizer"])
        scheduler.load_state_dict(rnn_state.get("scheduler", {}))
        earlystopping.load_state_dict(rnn_state.get("earlystopping", {}))

    if use_precomputed_latents and exists(latent_file):
        train_dataset: LatentSequenceDataset | SequenceDataset = LatentSequenceDataset(
            latents_arr=latents,
            actions=all_actions,
            rewards=all_rewards,
            terminals=all_terminals,
            train=True,
            buffer_size=30,
            num_test_files=600,
            seq_len=mdrnn_config.seq_len,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=mdrnn_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        test_dataset: LatentSequenceDataset | SequenceDataset = LatentSequenceDataset(
            latents_arr=latents,
            actions=all_actions,
            rewards=all_rewards,
            terminals=all_terminals,
            train=False,
            buffer_size=10,
            num_test_files=600,
            seq_len=mdrnn_config.seq_len,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=mdrnn_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    else:

        def transform(x: Any) -> torch.Tensor:
            return torch.tensor(x).float() / 255.0

        train_dataset = SequenceDataset(
            root=mdrnn_config.data_dir,
            transform=transform,
            train=True,
            buffer_size=30,
            num_test_files=600,
            seq_len=mdrnn_config.seq_len,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=mdrnn_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        test_dataset = SequenceDataset(
            root=mdrnn_config.data_dir,
            transform=transform,
            train=False,
            buffer_size=10,
            num_test_files=600,
            seq_len=mdrnn_config.seq_len,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=mdrnn_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    train_fn = partial(
        data_pass,
        mdrnn=mdrnn,
        vae=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        include_reward=mdrnn_config.include_reward,
        latent_size=mdrnn_config.latent_size,
        batch_size=mdrnn_config.batch_size,
        train=True,
        use_amp=use_amp,
        scaler=scaler,
    )

    test_fn = partial(
        data_pass,
        mdrnn=mdrnn,
        vae=vae,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        include_reward=mdrnn_config.include_reward,
        latent_size=mdrnn_config.latent_size,
        batch_size=mdrnn_config.batch_size,
        train=False,
        use_amp=use_amp,
        scaler=scaler,
    )

    cur_best = None

    for e in range(1, mdrnn_config.num_epochs + 1):
        train_fn(e)
        test_loss = test_fn(e)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        checkpoint_fname = join(rnn_dir, "checkpoint.tar")
        save_checkpoint(
            {
                "state_dict": mdrnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "earlystopping": earlystopping.state_dict(),
                "precision": test_loss,
                "epoch": e,
            },
            is_best,
            checkpoint_fname,
            rnn_file,
        )

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break

    torch.save(mdrnn.state_dict(), join(rnn_dir, "mdrnn_final.pth"))
