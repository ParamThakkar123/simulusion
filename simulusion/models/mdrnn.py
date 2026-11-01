"""Mixture Density Recurrent Neural Network (MDRNN) model implementation.

This module provides implementations of MDRNN models for world modeling.
The MDRNN is used to predict future latent states given current latent
states and actions, using a Gaussian Mixture Model (GMM) for the output.

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple


class _MDRNNBase(nn.Module):
    """Base class for MDRNN models.

    This base class provides the shared GMM (Gaussian Mixture Model) output layer
    used by both MDRNN (multi-step) and MDRNNCell (single-step) implementations.

    Args:
        latents: Dimensionality of latent space input.
        actions: Dimensionality of action space.
        hiddens: Number of hidden units in RNN.
        gaussians: Number of Gaussian components in GMM output.
    """

    def __init__(
        self, latents: int, actions: int, hiddens: int, gaussians: int
    ) -> None:
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs: Any) -> Any:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError


class MDRNN(_MDRNNBase):
    """MDRNN model for multi-step sequence prediction.

    This model processes entire sequences of latent states and actions,
    predicting the next latent state using a Gaussian Mixture Model (GMM).
    It also predicts rewards and terminal states.

    Args:
        latents: Dimensionality of latent space (input and output).
        actions: Dimensionality of action space.
        hiddens: Number of hidden units in LSTM.
        gaussians: Number of Gaussian components in GMM output.

    Example:
        >>> mdrnn = MDRNN(latents=32, actions=3, hiddens=256, gaussians=5)
        >>> actions = torch.randn(10, 4, 3)  # seq_len, batch, action
        >>> latents = torch.randn(10, 4, 32)  # seq_len, batch, latent
        >>> mus, sigmas, logpi, rs, ds = mdrnn(actions, latents)
        >>> # mus.shape = (10, 4, 5, 32)
    """

    def __init__(
        self, latents: int, actions: int, hiddens: int, gaussians: int
    ) -> None:
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(
        self, actions: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-step forward pass through the MDRNN.

        Args:
            actions: (SEQ_LEN, BSIZE, ASIZE) Tensor of actions.
            latents: (SEQ_LEN, BSIZE, LSIZE) Tensor of latent states.

        Returns:
            Tuple of:
                - mus: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) GMM means
                - sigmas: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) GMM standard deviations
                - logpi: (SEQ_LEN, BSIZE, N_GAUSS) log GMM weights
                - rs: (SEQ_LEN, BSIZE) predicted rewards
                - ds: (SEQ_LEN, BSIZE) predicted terminal state logits
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride : 2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride : 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

    def get_init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial hidden state for the LSTM.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Tuple of (h, c) with shapes (batch_size, hiddens).
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hiddens, device=device)
        c = torch.zeros(batch_size, self.hiddens, device=device)
        return h, c


class MDRNNCell(_MDRNNBase):
    """MDRNN model for single-step forward prediction.

    This model processes a single step of latent state and action,

    This model processes a single step of latent state and action,
    predicting the next latent state using a Gaussian Mixture Model (GMM).
    It also predicts rewards and terminal states. Useful for real-time inference.

    Args:
        latents: Dimensionality of latent space (input and output).
        actions: Dimensionality of action space.
        hiddens: Number of hidden units in LSTMCell.
        gaussians: Number of Gaussian components in GMM output.

    Example:
        >>> cell = MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)
        >>> action = torch.randn(4, 3)  # batch, action
        >>> latent = torch.randn(4, 32)  # batch, latent
        >>> hidden = (torch.randn(4, 256), torch.randn(4, 256))
        >>> mus, sigmas, logpi, r, d, next_hidden = cell(action, latent, hidden)
    """

    def __init__(
        self, latents: int, actions: int, hiddens: int, gaussians: int
    ) -> None:
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(
        self,
        action: torch.Tensor,
        latent: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Single-step forward pass through the MDRNN cell.

        Args:
            action: (BSIZE, ASIZE) Tensor of actions for current batch.
            latent: (BSIZE, LSIZE) Tensor of latent states for current batch.
            hidden: Tuple of (h, c) hidden states for LSTMCell.

        Returns:
            Tuple of:
                - mus: (BSIZE, N_GAUSS, LSIZE) GMM means
                - sigmas: (BSIZE, N_GAUSS, LSIZE) GMM standard deviations
                - logpi: (BSIZE, N_GAUSS) log GMM weights
                - r: (BSIZE,) predicted rewards
                - d: (BSIZE,) predicted terminal state logits
                - next_hidden: Tuple of (h, c) next hidden states
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride : 2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride : 2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden

    def get_init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hiddens, device=device)
        c = torch.zeros(batch_size, self.hiddens, device=device)
        return h, c
