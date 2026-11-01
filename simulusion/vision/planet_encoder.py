import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


class CNNEncoder(ExportableAgentMixin, nn.Module):
    """A Convolutional Neural Network (CNN) encoder for processing image inputs."""

    def __init__(self, embedding_size: int, activation_function: str = "relu") -> None:
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc: nn.Linear | nn.Identity = (
            nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.reshape(hidden.size(0), -1)  # (N, 256 * 2 * 2)
        hidden = self.fc(hidden)
        return hidden
