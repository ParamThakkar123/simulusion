import torch
import torch.nn as nn
import torch.nn.functional as F
from simulusion.export import ExportableAgentMixin


class RewardModel(ExportableAgentMixin, nn.Module):
    """Predict scalar rewards from Dreamer latent belief and state vectors.

    Implemented as an MLP used for model-based reward supervision and
    imagined rollout return estimation.
    """

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        activation_function: str = "relu",
    ) -> None:
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([belief, state], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden)
        reward = reward.squeeze(dim=-1)
        return reward
