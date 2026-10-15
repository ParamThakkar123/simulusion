"""Linear Controller for World Models.

This module provides a simple linear controller that maps latent states
and recurrent hidden states to actions. The controller is trained using
CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

import torch
import torch.nn as nn


class Controller(nn.Module):
    """Linear controller that maps latent + hidden state to actions.

    This is a simple linear controller that takes the latent state and
    recurrent hidden state as input and outputs actions. It is trained
    separately from the world model using black-box optimization (CMA-ES).

    Attributes:
        latent_size: Dimensionality of latent state from VAE.
        hidden_size: Dimensionality of RSSM hidden state.
        action_size: Dimensionality of action space.

    Example:
        >>> controller = Controller(latent_size=32, hidden_size=200, action_size=3)
        >>> state = torch.cat([latent, hidden], dim=-1)
        >>> action = controller(state)
    """

    def __init__(self, latent_size: int, hidden_size: int, action_size: int):
        """Initialize the linear controller.

        Args:
            latent_size: Dimensionality of latent state from VAE.
            hidden_size: Dimensionality of RSSM hidden state.
            action_size: Dimensionality of action space.
        """
        super(Controller, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute actions from latent and hidden states.

        Args:
            state: Concatenated [latent, hidden] state tensor.

        Returns:
            Action tensor of shape (batch, action_size).
        """
        return self.fc(state)
