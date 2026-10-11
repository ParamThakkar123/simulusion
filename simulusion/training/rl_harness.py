from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, List
import logging

from simulusion.envs.vector_env import TorchVectorizedEnv

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Simple actor-critic network for RL harness."""

    def __init__(self, obs_shape: tuple, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # CNN for image observations
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            cnn_out = self.cnn(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(cnn_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CNN, then actor and critic heads."""
        features = self.cnn(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        logits, value = self(obs)
        dist: Any = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class PPOTrainer:
    """Simple PPO trainer for testing vectorized environments."""

    def __init__(
        self,
        vec_env: TorchVectorizedEnv,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        num_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
    ):
        self.vec_env = vec_env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        # Get action dim from env
        if hasattr(vec_env.action_space, "n"):
            action_dim = vec_env.action_space.n
        else:
            action_dim = vec_env.action_space.shape[0]

        # Assume image obs for now
        obs_shape = vec_env.observation_space["image"].shape

        self.policy = ActorCritic(obs_shape, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_trajectories(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect trajectories using the vectorized environment."""
        obs_batch: Any = self.vec_env.reset_batch()
        obs = obs_batch["obs"]["image"].to(self.device)

        # Collect lists of tensors, convert to stacked tensors before returning
        trajectories: Dict[str, List[torch.Tensor]] = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

        for _ in range(num_steps // self.vec_env.total_envs):
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action(obs)

            # Step environment
            actions_np = actions.cpu().numpy()
            step_result = self.vec_env.step_batch(
                torch.from_numpy(actions_np).to(self.device)
            )

            next_obs = step_result["obs"]["image"].to(self.device)
            rewards = step_result["reward"].to(self.device)
            dones = step_result["done"].to(self.device)

            # Store trajectory data
            trajectories["obs"].append(obs)
            trajectories["actions"].append(actions)
            trajectories["log_probs"].append(log_probs)
            trajectories["rewards"].append(rewards)
            trajectories["values"].append(values.squeeze(-1))
            trajectories["dones"].append(dones)

            obs = next_obs

        # Convert lists to stacked tensors for return
        result: Dict[str, torch.Tensor] = {}
        for key, lst in trajectories.items():
            result[key] = torch.stack(lst)

        return result

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        # ensure last_gae is a tensor matching per-step value shape
        # values[t] has the same shape as rewards[t]
        last_gae = torch.zeros_like(values[0])

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = torch.zeros_like(values[t])
            else:
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value * (1 - dones[t].float())
                - values[t]
            )
            last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * last_gae
            )
            advantages[t] = last_gae

        return advantages

    def train_step(self, trajectories: Dict[str, torch.Tensor]) -> None:
        """Perform one training step using PPO."""
        obs = trajectories["obs"].view(
            -1, *self.vec_env.observation_space["image"].shape
        )
        actions = trajectories["actions"].view(-1)
        old_log_probs = trajectories["log_probs"].view(-1)
        rewards = trajectories["rewards"].view(-1)
        values = trajectories["values"].view(-1)
        dones = trajectories["dones"].view(-1)

        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.num_epochs):
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                new_logits, new_values = self.policy(batch_obs)
                new_values = new_values.squeeze(-1)

                # Policy loss
                dist = Categorical(logits=new_logits)
                new_log_probs: Any = dist.log_prob(batch_actions)  # type: ignore[no-untyped-call]
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                # Entropy bonus
                entropy: Any = dist.entropy().mean()  # type: ignore[no-untyped-call]

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

    def train(self, total_timesteps: int, log_interval: int = 1000) -> None:
        """Main training loop."""
        logger.info(f"Starting training for {total_timesteps} timesteps")

        timestep = 0
        while timestep < total_timesteps:
            # Collect trajectories
            trajectories = self.collect_trajectories(
                min(2048, total_timesteps - timestep)
            )
            # trajectories["obs"] is a tensor; ensure int arithmetic for timestep
            timestep += int(len(trajectories["obs"])) * self.vec_env.total_envs

            # Train
            self.train_step(trajectories)

            if timestep % log_interval == 0:
                logger.info(f"Timestep {timestep}: Training step completed")


def create_rl_harness_example() -> None:
    """
    Example function to create and run the RL harness.
    Usage: Call this with your environment factory.
    """
    from simulusion.envs.gym_env import make_gym_env

    def env_factory() -> Any:
        return make_gym_env("CartPole-v1", size=(64, 64))

    vec_env = TorchVectorizedEnv(
        env_factory=env_factory, num_workers=2, envs_per_worker=4, seed=42
    )

    # Create trainer
    trainer = PPOTrainer(vec_env, device="cpu")

    # Train
    trainer.train(total_timesteps=10000)

    # Cleanup
    vec_env.close()


if __name__ == "__main__":
    create_rl_harness_example()
