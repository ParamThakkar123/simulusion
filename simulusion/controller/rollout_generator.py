"""Rollout generation utilities for World Models.

This module provides the RolloutGenerator class for collecting episode
experience using trained policies in environments.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Any

from tqdm import trange
from torchvision.utils import make_grid

from simulusion.memory.planet_memory import Episode
from simulusion.utils.utils import StreamingVideoWriter


class RolloutGenerator:
    """Generator for collecting environment rollouts.

    This class handles environment interactions and rollout collection,
    supporting both random and policy-based action selection.

    Attributes:
        env: The environment to interact with.
        device: Device to run computations on.
        policy: The policy to use for action selection (optional).
        episode_gen: Factory for creating episode objects.
        name: Name identifier for the generator.
        max_episode_steps: Maximum steps per episode.

    Example:
        >>> generator = RolloutGenerator(
        ...     env=env,
        ...     device='cuda',
        ...     policy=policy,
        ...     max_episode_steps=1000
        ... )
        >>> episode = generator.rollout_once()
    """

    def __init__(
        self,
        env: Any,
        device: torch.device | str,
        policy: Any = None,
        max_episode_steps: int | None = None,
        episode_gen: Any = None,
        name: str = "",
        enable_streaming_video: bool = False,
        streaming_video_path: str | None = None,
        streaming_video_fps: int = 20,
        streaming_video_format: str = "mp4",
    ) -> None:
        """Initialize the RolloutGenerator.

        Args:
            env: The environment to interact with.
            device: Device for tensor operations.
            policy: Policy to use for action selection.
            max_episode_steps: Maximum steps per episode.
            episode_gen: Factory function for creating episodes.
            name: Name identifier for this generator.
        """
        self.env = env
        self.device = device
        self.policy = policy
        self.episode_gen = episode_gen or Episode
        self.name = name or "Rollout Generator"
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps is None:
            self.max_episode_steps = self.env.max_episode_steps
        self.enable_streaming_video = enable_streaming_video
        self.streaming_video_path = streaming_video_path
        self.streaming_video_fps = streaming_video_fps
        self.streaming_video_format = streaming_video_format
        self.video_writer: Any = None

    def rollout_once(
        self, random_policy: bool = False, explore: bool = False
    ) -> Episode:
        """Perform a single rollout of the environment.

        Args:
            random_policy: If True, use random actions instead of policy.
            explore: If True, add exploration noise to policy actions.

        Returns:
            Episode object containing the rollout experience.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print("Policy is None. Using random policy instead!!")
        if not random_policy:
            self.policy.reset()
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f"{self.name} Ts"
        for _ in trange(self.max_episode_steps, desc=des, leave=False):
            if random_policy:
                act = self.env.sample_random_action()
            else:
                act = self.policy.poll(obs.to(self.device), explore).flatten()
            nobs, reward, terminal, _ = self.env.step(act)
            eps.append(obs, act, reward, terminal)
            obs = nobs
        eps.terminate(nobs)
        return eps

    def rollout_n(self, n: int = 1, random_policy: bool = False) -> list:
        """Perform multiple rollouts.

        Args:
            n: Number of rollouts to perform.
            random_policy: If True, use random actions.

        Returns:
            List of Episode objects.
        """
        if self.policy is None and not random_policy:
            random_policy = True
            print("Policy is None. Using random policy instead!!")
        des = f"{self.name} EPS"
        ret = []
        for _ in trange(n, desc=des, leave=False):
            ret.append(self.rollout_once(random_policy=random_policy))
        return ret

    def rollout_eval_n(self, n: int) -> tuple:
        """Perform multiple evaluation rollouts with metrics.

        Args:
            n: Number of evaluation rollouts.

        Returns:
            Tuple of (episodes, frames, metrics).
        """
        metrics = defaultdict(list)
        episodes, frames = [], []
        for _ in range(n):
            # rollout_eval returns four values; the trailing latents are None
            # unless collect_latents was asked for, and are not aggregated here.
            e, f, m, _ = self.rollout_eval()
            episodes.append(e)
            frames.append(f)
            for k, v in m.items():
                metrics[k].append(v)
        return episodes, frames, metrics

    def rollout_eval(self, collect_latents: bool = False) -> tuple:
        assert self.policy is not None, "Policy is None!!"
        self.policy.reset()
        eps = self.episode_gen()
        obs = self.env.reset()
        des = f"{self.name} Eval Ts"
        frames: list = []
        latents_list: list | None = [] if collect_latents else None
        if self.enable_streaming_video and self.streaming_video_path:
            self.video_writer = StreamingVideoWriter(
                self.streaming_video_path,
                fps=self.streaming_video_fps,
                format=self.streaming_video_format,
            )
        metrics: dict[str, Any] = {}
        rec_losses = []
        pred_r, act_r = [], []
        eps_reward = 0
        for _ in trange(self.max_episode_steps, desc=des, leave=False):
            with torch.no_grad():
                act = self.policy.poll(obs.to(self.device)).flatten()
                dec = (
                    self.policy.rssm.decoder(self.policy.h, self.policy.s)
                    .squeeze()
                    .cpu()
                    .clamp_(-0.5, 0.5)
                )
                rec_losses.append(((obs - dec).abs()).sum().item())
                frame = make_grid([obs + 0.5, dec + 0.5], nrow=2).numpy()
                frames.append(frame)
                if self.video_writer:
                    self.video_writer.write_frame(frame)
                if collect_latents and latents_list is not None:
                    latents_list.append(
                        torch.cat([self.policy.h, self.policy.s], dim=-1).cpu().numpy()
                    )
                pred_r.append(
                    self.policy.rssm.pred_reward(self.policy.h, self.policy.s)
                    .cpu()
                    .flatten()
                    .item()
                )
            nobs, reward, terminal, _ = self.env.step(act)
            eps.append(obs, act, reward, terminal)
            act_r.append(reward)
            eps_reward += reward
            obs = nobs
        eps.terminate(nobs)
        if self.video_writer:
            self.video_writer.close()
        metrics["eval/episode_reward"] = eps_reward
        metrics["eval/reconstruction_loss"] = rec_losses
        metrics["eval/reward_pred_loss"] = abs(
            np.array(act_r)[:-1] - np.array(pred_r)[1:]
        )
        return eps, np.stack(frames), metrics, latents_list
