import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicObservationModel(nn.Module):
    """Observation decoder for low-dimensional (symbolic) environments.

    Maps latent belief/state features to vector observations through an MLP.
    """

    def __init__(
        self,
        observation_size: int,
        belief_size: int,
        state_size: int,
        embedding_size: int,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)

    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(nn.Module):
    """Observation decoder for pixel observations in Dreamer-style models.

    Uses transposed convolutions to reconstruct RGB-like frames from latent
    belief/state features.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        embedding_size: int,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(
    symbolic: bool,
    observation_size: int,
    belief_size: int,
    state_size: int,
    embedding_size: int,
    activation_function: str = "relu",
) -> nn.Module:
    """Factory that returns symbolic or visual observation decoder variants.

    Selects the concrete model class based on the `symbolic` flag.
    """
    if symbolic:
        return SymbolicObservationModel(
            observation_size,
            belief_size,
            state_size,
            embedding_size,
            activation_function,
        )
    else:
        return VisualObservationModel(
            belief_size, state_size, embedding_size, activation_function
        )
