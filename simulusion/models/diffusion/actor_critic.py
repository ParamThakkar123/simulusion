import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, cast

from simulusion.models.diffusion.reward_termination import ResidualBlock
from simulusion.export import ExportableAgentMixin


class ActorCriticNetwork(ExportableAgentMixin, nn.Module):
    """
    Actor-Critic network for DIAMOND RL training.
    Shared CNN-LSTM trunk with separate policy and value heads.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        action_dim: int = 18,
        channels: Tuple[int, ...] = (32, 32, 64, 64),
        lstm_dim: int = 512,
        res_blocks: int = 1,
        frame_size: int = 64,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim

        # Convolutional trunk of residual blocks with 2x2 max-pool downsampling
        # (DIAMOND Appendix D). The actor-critic takes a single frame (no action
        # conditioning), so the residual blocks use plain group normalization.
        self.stages = nn.ModuleList()
        in_ch = obs_channels
        for out_ch in channels:
            blocks = nn.ModuleList()
            for _ in range(res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            self.stages.append(blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Appendix D: "convolutional trunk followed by an LSTM cell" -- the trunk
        # output is flattened into the cell, not spatially averaged. Pooling it
        # away leaves the policy blind to *where* things are on screen, which for
        # Breakout or Pong is the entire state.
        spatial = max(1, frame_size // (2 ** len(channels)))
        self.feature_size = channels[-1] * spatial * spatial

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
        )

        self.policy_head = nn.Linear(lstm_dim, action_dim)
        self.value_head = nn.Linear(lstm_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of actor-critic network.

        Args:
            obs: Observations [B, T, C, H, W]
            hidden_state: Optional (h, c) hidden states

        Returns:
            policy_logits: [B, T, action_dim]
            values: [B, T, 1]
            hidden_state: (h, c)
        """
        B, T, C, H, W = obs.shape

        obs_flat = obs.reshape(B * T, C, H, W)

        h = obs_flat
        for stage in self.stages:
            for block in cast(nn.ModuleList, stage):
                h = block(h)
            h = self.downsample(h)

        h = h.reshape(B, T, -1)

        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(h)
        else:
            lstm_out, hidden_state = self.lstm(h, hidden_state)

        policy_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out)

        return policy_logits, values, hidden_state

    def get_action(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get action from a single observation.

        Args:
            obs: Single observation [B, C, H, W]
            hidden_state: Optional (h, c) hidden states
            deterministic: If True, take argmax; else sample

        Returns:
            action: Selected action [B]
            hidden_state: (h, c)
        """
        # delegate to the batched interface for a single-sample convenience
        actions, hidden_state = self.get_actions(
            obs.unsqueeze(0), hidden_state, deterministic=deterministic
        )
        # get scalar action for the single sample
        return int(actions[0].item()), hidden_state

    def get_actions(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Batched version of get_action.

        Args:
            obs: Tensor of shape [B, C, H, W]
            hidden_state: Optional LSTM hidden state tuple matching batch size
            deterministic: If True, take argmax; else sample from policy

        Returns:
            actions: LongTensor of shape [B]
            hidden_state: updated LSTM hidden state tuple
        """
        # convert to [B, T=1, C, H, W] expected by forward
        if obs.ndim == 4:
            obs_in = obs.unsqueeze(1)
        elif obs.ndim == 5:
            obs_in = obs
        else:
            raise ValueError(f"Unexpected obs ndim {obs.ndim}")

        policy_logits, values, hidden_state = self.forward(obs_in, hidden_state)
        # policy_logits: [B, T, A] -> take last timestep
        logits = policy_logits[:, -1, :]

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            # torch.multinomial expects 2D probs and returns [B, k]
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return actions.long(), hidden_state

    def get_value(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get value for a single observation."""
        obs = obs.unsqueeze(1)
        _, values, hidden_state = self.forward(obs, hidden_state)
        return values.squeeze(1), hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return (h, c)

    def get_hidden_size(self) -> int:
        """Get LSTM hidden size."""
        return self.lstm.hidden_size


class RLLoss(nn.Module):
    """
    RL loss functions for DIAMOND.
    Implements REINFORCE with value baseline and λ-returns.
    """

    def __init__(
        self,
        discount_factor: float = 0.985,
        lambda_returns: float = 0.95,
        entropy_weight: float = 0.001,
    ):
        super().__init__()
        self.discount_factor = discount_factor
        self.lambda_returns = lambda_returns
        self.entropy_weight = entropy_weight

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute λ-returns.

        Args:
            rewards: [B, T]
            values: [B, T+1]
            dones: [B, T]

        Returns:
            lambda_returns: [B, T]
        """
        B, T = rewards.shape

        lambda_returns = torch.zeros_like(rewards)

        returns = values[:, -1]

        for t in reversed(range(T)):
            returns = rewards[:, t] + self.discount_factor * (
                1 - dones[:, t].float()
            ) * (
                (1 - self.lambda_returns) * values[:, t + 1]
                + self.lambda_returns * returns
            )
            lambda_returns[:, t] = returns

        return lambda_returns

    def policy_loss(
        self,
        policy_logits: torch.Tensor,
        actions: torch.Tensor,
        lambda_returns: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy loss with REINFORCE and entropy regularization.

        Args:
            policy_logits: [B, T, A]
            actions: [B, T]
            lambda_returns: [B, T]
            values: [B, T+1]

        Returns:
            policy_loss: scalar
        """
        log_probs = F.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        advantages = lambda_returns - values[:, :-1]

        policy_loss = -(action_log_probs * advantages.detach()).mean()

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy

        return policy_loss + entropy_loss

    def value_loss(
        self,
        values: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value loss (MSE between value and lambda returns)."""
        target = lambda_returns.detach()
        value_pred = values[:, :-1].squeeze(-1)
        return F.mse_loss(value_pred, target)
