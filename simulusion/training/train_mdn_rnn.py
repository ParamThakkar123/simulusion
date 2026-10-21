"""Training script for Mixture Density Recurrent Neural Network (MDRNN).

This module provides functions to train an MDRNN model for sequence prediction
in world models. The MDRNN predicts future latent states using a Gaussian
Mixture Model (GMM) based on current latent states and actions.
"""

from os.path import join, exists

import torch
import torch.nn.functional as F
import torch.utils.data

from simulusion.configs.wm_config import WMVAEConfig, WMMDNRNNConfig
from simulusion.models.mdrnn import MDRNN
from simulusion.losses.gmm_loss import gmm_loss


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


def train_mdn_rnn(
    vae_config: WMVAEConfig,
    mdrnn_config: WMMDNRNNConfig,
    use_precomputed_latents: bool = True,
    use_amp: bool = True,
) -> None:
    """Train an MDRNN model.

    Runs on the shared :class:`~simulusion.training.Trainer` through the
    ``mdrnn`` task, on the rollouts in ``mdrnn_config.data_dir``, encoded by
    the VAE in ``<vae_config.logdir>/vae/best.tar``. As before, everything
    goes to ``<mdrnn_config.logdir>/mdrnn``: ``checkpoint.tar`` every epoch,
    ``best.tar`` for the lowest held-out loss (resumed from unless
    ``noreload``), ``ReduceLROnPlateau`` and early stopping on that loss, and
    ``mdrnn_final.pth``.

    Args:
        vae_config: WMVAEConfig for loading pretrained VAE.
        mdrnn_config: WMMDNRNNConfig with the MDN-RNN's settings.
        use_precomputed_latents: Accepted for compatibility. Latents are
            sampled from the VAE as each batch is drawn, so every epoch sees
            fresh samples instead of one fixed draw stored on disk.
        use_amp: fp16 with loss scaling, on CUDA.

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
    del use_precomputed_latents
    from simulusion.training.legacy import fit_legacy
    from simulusion.training.tasks.world_models import MDRNNTask, MDRNNTaskConfig

    device = torch.device(mdrnn_config.device)
    vae_file = join(vae_config.logdir, "vae", "best.tar")
    assert exists(vae_file), "No trained VAE in the logdir..."
    rnn_dir = join(mdrnn_config.logdir, "mdrnn")
    task = MDRNNTask(
        MDRNNTaskConfig(model=mdrnn_config, vae=vae_file), legacy_dir=rnn_dir
    )
    fit_legacy(
        task,
        rnn_dir,
        device=device,
        precision="fp16" if use_amp and device.type == "cuda" else "fp32",
        eval_every=1,
        checkpoint_every=1,
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
        patience=mdrnn_config.early_stopping_patience,
    )
    torch.save(
        task.trainer.unwrap(task.mdrnn).state_dict(), join(rnn_dir, "mdrnn_final.pth")
    )
