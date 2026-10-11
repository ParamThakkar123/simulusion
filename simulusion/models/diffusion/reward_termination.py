import torch
import torch.nn as nn
from typing import Tuple, Optional, cast
from simulusion.export import ExportableAgentMixin


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Pick a GroupNorm group count that divides ``channels`` (<= ``max_groups``)."""
    for g in (max_groups, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    """Residual block following DIAMOND Appendix D.

    The main path is GroupNorm -> SiLU -> 3x3 convolution (stride 1, padding 1),
    added to a (optionally projected) skip connection. When ``cond_dim`` is
    provided the group normalization is made *adaptive*, i.e. its scale/shift are
    predicted from a conditioning vector (the action embedding) as used by the
    reward/termination model. The actor-critic omits conditioning and uses a
    plain group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        groups = _num_groups(in_channels)
        self.conditioned = cond_dim is not None
        self.cond_embed: Optional[nn.Linear]
        if cond_dim is not None:
            # affine=False: the affine parameters are supplied by ``cond_embed``
            self.norm = nn.GroupNorm(groups, in_channels, affine=False)
            self.cond_embed = nn.Linear(cond_dim, in_channels * 2)
        else:
            self.norm = nn.GroupNorm(groups, in_channels)
            self.cond_embed = None

        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.skip: nn.Module
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.norm(x)
        if self.cond_embed is not None and cond is not None:
            scale, bias = self.cond_embed(cond).chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + bias.unsqueeze(
                -1
            ).unsqueeze(-1)
        h = self.act(h)
        h = self.conv(h)
        return h + self.skip(x)


class RewardTerminationModel(ExportableAgentMixin, nn.Module):
    """
    Reward and termination prediction model.
    CNN + LSTM architecture following DIAMOND paper specifications.

    Args:
        obs_channels: Number of observation channels (3 for RGB)
        action_dim: Number of possible actions
        channels: List of channel sizes for conv blocks
        lstm_dim: LSTM hidden dimension
        cond_dim: Conditioning dimension for adaptive norm
    """

    def __init__(
        self,
        obs_channels: int = 3,
        action_dim: int = 18,
        channels: Tuple[int, ...] = (32, 32, 32, 32),
        lstm_dim: int = 512,
        cond_dim: int = 128,
        res_blocks: int = 2,
        frame_size: int = 64,
    ):
        super().__init__()
        self.obs_channels = obs_channels
        self.action_dim = action_dim
        self.lstm_dim = lstm_dim

        self.action_embed = nn.Embedding(action_dim, cond_dim)

        # Convolutional trunk of residual blocks with 2x2 max-pool downsampling
        # (DIAMOND Appendix D). Each stage holds ``res_blocks`` action-conditioned
        # residual blocks (adaptive group norm) followed by a 2x2 stride-2 pool.
        self.stages = nn.ModuleList()
        in_ch = obs_channels
        for out_ch in channels:
            blocks = nn.ModuleList()
            for _ in range(res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, cond_dim=cond_dim))
                in_ch = out_ch
            self.stages.append(blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Appendix D feeds the convolutional trunk straight into the LSTM cell,
        # i.e. the whole feature map. Global-average-pooling it to `channels[-1]`
        # numbers first -- as this did -- discards every bit of spatial layout,
        # so the reward head could no longer tell *where* on the screen an event
        # happened. That is precisely the visual detail the paper argues for.
        spatial = max(1, frame_size // (2 ** len(channels)))
        self.feature_size = channels[-1] * spatial * spatial

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
        )

        self.reward_head = nn.Linear(lstm_dim, 3)
        self.termination_head = nn.Linear(lstm_dim, 2)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of reward/termination model.

        Args:
            obs: Observations [B, T, C, H, W]
            actions: Actions [B, T]
            hidden_state: Optional (h, c) hidden states

        Returns:
            reward_logits: Reward predictions [B, T, 3] (for -1, 0, 1)
            termination_logits: Termination predictions [B, T, 2]
            hidden_state: Updated (h, c) hidden states
        """
        B, T, C, H, W = obs.shape

        obs_flat = obs.reshape(B * T, C, H, W)
        actions_flat = actions.reshape(B * T)

        action_emb = self.action_embed(actions_flat)

        h = obs_flat
        for stage in self.stages:
            for block in cast(nn.ModuleList, stage):
                h = block(h, action_emb)
            h = self.downsample(h)

        h = h.reshape(B, T, -1)

        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(h)
        else:
            lstm_out, hidden_state = self.lstm(h, hidden_state)

        reward_logits = self.reward_head(lstm_out)
        termination_logits = self.termination_head(lstm_out)

        return reward_logits, termination_logits, hidden_state

    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Predict reward and termination for a single step.

        Args:
            obs: Single observation [B, C, H, W]
            actions: Single action [B]
            hidden_state: Optional (h, c) hidden states

        Returns:
            reward: Predicted reward classes as tensor (values -1,0,1)
            terminated: Predicted termination tensor (bool tensor)
            hidden_state: Updated (h, c) hidden states
        """
        obs = obs.unsqueeze(1)
        actions = actions.unsqueeze(1)

        reward_logits, term_logits, hidden_state = self.forward(
            obs, actions, hidden_state
        )

        reward = reward_logits.argmax(dim=-1) - 1
        terminated = term_logits.argmax(dim=-1).bool()

        return reward.squeeze(-1).float(), terminated.squeeze(-1), hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        h = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_dim, device=device)
        return (h, c)


class RewardTerminationLoss(nn.Module):
    """Loss function for reward and termination prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.reward_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.termination_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        reward_logits: torch.Tensor,
        termination_logits: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss for reward and termination predictions.

        Args:
            reward_logits: [B, T, 3]
            termination_logits: [B, T, 2]
            rewards: Rewards [B, T]. Mapped to class indices via sign(r) + 1,
                i.e. {-1, 0, +1} reward signs -> classes {0, 1, 2}.
            terminated: Termination flags [B, T]

        Returns:
            total_loss, reward_loss, termination_loss
        """
        # Paper (Algorithm 1) trains the reward head with CE(r_hat, sign(r)).
        # Using sign() here keeps the target correct even if a reward is not
        # already clipped to {-1, 0, 1} (e.g. a fractional value would otherwise
        # be truncated toward zero by the direct `rewards + 1` mapping).
        reward_targets = (torch.sign(rewards) + 1).long()

        # use reshape to avoid issues when tensors are non-contiguous
        reward_loss = self.reward_criterion(
            reward_logits.reshape(-1, 3), reward_targets.view(-1)
        )
        termination_loss = self.termination_criterion(
            termination_logits.reshape(-1, 2), terminated.long().view(-1)
        )

        total_loss = reward_loss + termination_loss

        return total_loss, reward_loss, termination_loss
