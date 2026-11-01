"""Gaussian Mixture Model (GMM) loss for MDRNN training.

This module provides the GMM loss function used in the Mixture Density
Recurrent Neural Network (MDRNN) for world model training.
"""

from typing import Any

import torch
from torch.distributions.normal import Normal


def gmm_loss(
    latent_next_obs: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    logpi: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Compute the negative log-likelihood of a batch under a Gaussian Mixture Model.

    This function computes minus the log probability of the batch under the GMM
    model described by mus, sigmas, and pi:

    .. math::

        p(x) = \\sum_k \\pi_k \\cdot \\mathcal{N}(x \\mid \\mu_k, \\sigma_k)

    This is the loss function used in the MDRNN paper for predicting
    the next latent state.

    Args:
        latent_next_obs: (bs1, bs2, ..., fs) Tensor containing the batch of target data.
        mus: (bs1, bs2, ..., gs, fs) Tensor of mixture means.
        sigmas: (bs1, bs2, ..., gs, fs) Tensor of mixture standard deviations.
        logpi: (bs1, bs2, ..., gs) Tensor of log mixture weights (log pi_k).
        reduce: If True, mean over batch dimensions; otherwise return per-sample loss.

    Returns:
        If reduce is True: scalar tensor with mean negative log-likelihood.
        If reduce is False: tensor with per-sample negative log-likelihoods.

    Reference:
        Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.

    Example:
        >>> batch = torch.randn(32, 10)
        >>> mus = torch.randn(32, 10, 5, 10)
        >>> sigmas = torch.randn(32, 10, 5, 10).exp()
        >>> logpi = torch.randn(32, 10, 5).log_softmax(dim=-1)
        >>> loss = gmm_loss(batch, mus, sigmas, logpi)
    """
    latent_next_obs = latent_next_obs.unsqueeze(-2)
    normal_dist: Any = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(latent_next_obs)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs
    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob
