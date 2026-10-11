"""Utility functions for world model evaluation.

Handles trajectory generation from a DIAMOND world model and
data loading from environments or replay buffers.
"""

import torch
import numpy as np
from typing import Callable, Optional, Dict, List

from simulusion.models.diffusion.diamond_diffusion import (
    DiffusionUNet,
    EulerSampler,
)


@torch.no_grad()
def generate_trajectories(
    diffusion_model: DiffusionUNet,
    sampler: EulerSampler,
    real_trajectories: Dict[str, torch.Tensor],
    num_conditioning_frames: int = 4,
    horizon: int = 16,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 16,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate imagined trajectories conditioned on real trajectories.

    For each real trajectory, uses the first ``num_conditioning_frames``
    frames + actions as conditioning, then autoregressively generates
    the remaining frames following the real action sequence.

    Args:
        diffusion_model: Trained DIAMOND diffusion model.
        sampler: Noise scheduler / sampler (e.g. EulerSampler).
        real_trajectories: Dict with keys:
            - "observations": [B, T, C, H, W] float in [0, 1]
            - "actions": [B, T] long tensor
        num_conditioning_frames: Number of frames used to condition the model.
        horizon: Number of frames to generate per trajectory.
        device: Target device.
        batch_size: Batch size for generation.
        progress_callback: Optional callback (current, total) for progress.

    Returns:
        Dict with keys:
            - "observations": [B, horizon, C, H, W] generated frames.
            - "real_observations": [B, horizon, C, H, W] corresponding real frames.
    """
    diffusion_model.eval()
    obs = real_trajectories["observations"].to(device)
    actions = real_trajectories["actions"].to(device)
    B, T, C, H, W = obs.shape

    gen_frames_list: List[torch.Tensor] = []
    real_frames_list: List[torch.Tensor] = []

    for start_idx in range(0, B, batch_size):
        end_idx = min(start_idx + batch_size, B)
        batch_obs = obs[start_idx:end_idx]
        batch_act = actions[start_idx:end_idx]
        batch_B = end_idx - start_idx

        # Initial conditioning: first num_conditioning_frames
        obs_history = batch_obs[:, :num_conditioning_frames, :, :, :]
        action_history = batch_act[:, :num_conditioning_frames]

        traj_gen = []

        for t in range(horizon):
            # Sample next frame from the diffusion model
            sampled = sampler.sample(
                model=diffusion_model,
                shape=(batch_B, C, H, W),
                device=device,
                obs_history=obs_history,
                actions=action_history,
            )

            traj_gen.append(sampled)

            # Advance conditioning window
            next_obs_seq = sampled.unsqueeze(1)
            obs_history = torch.cat([obs_history[:, 1:], next_obs_seq], dim=1)

            # Use the real action corresponding to the NEXT timestep
            # (the action that was taken to get from current to next state)
            if num_conditioning_frames + t < T:
                next_action = batch_act[:, num_conditioning_frames + t].unsqueeze(1)
                action_history = torch.cat([action_history[:, 1:], next_action], dim=1)
            else:
                # Pad with last known action if we exceed trajectory length
                action_history = torch.cat(
                    [action_history[:, 1:], action_history[:, -1:]], dim=1
                )

        gen_frames_list.append(torch.stack(traj_gen, dim=1))
        # Corresponding real frames (starting after conditioning)
        real_frames_list.append(
            batch_obs[
                :,
                num_conditioning_frames : num_conditioning_frames + horizon,
                :,
                :,
                :,
            ]
        )

        if progress_callback:
            progress_callback(end_idx, B)

    return {
        "observations": torch.cat(gen_frames_list, dim=0),
        "real_observations": torch.cat(real_frames_list, dim=0),
    }


def collect_real_trajectories_from_env(
    env_fn: Callable,
    action_dim: int,
    num_trajectories: int = 1024,
    trajectory_length: int = 20,
    num_conditioning_frames: int = 4,
    device: torch.device = torch.device("cpu"),
    policy_fn: Optional[Callable] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Collect real trajectories from an environment.

    Args:
        env_fn: Callable that returns a new environment instance.
        action_dim: Number of discrete actions.
        num_trajectories: Number of trajectories to collect.
        trajectory_length: Length of each trajectory in frames.
        num_conditioning_frames: Frames to reserve for conditioning.
        device: Target device.
        policy_fn: Optional policy for action selection (default: random).
        seed: Random seed.

    Returns:
        Dict with "observations" [N, T, C, H, W] and "actions" [N, T].
    """
    np.random.seed(seed)
    all_obs: List[np.ndarray] = []
    all_act: List[np.ndarray] = []
    total_len = trajectory_length

    collected = 0
    while collected < num_trajectories:
        env = env_fn()
        obs, _ = env.reset()
        obs = obs.astype(np.float32) / 255.0
        if obs.ndim == 3:
            pass  # (H, W, C)
        elif obs.ndim == 2:
            obs = np.stack([obs] * 3, axis=-1)

        traj_obs = [obs]
        traj_act: List[int] = []

        for _ in range(total_len - 1):
            if policy_fn is not None:
                action = policy_fn(obs)
            else:
                action = np.random.randint(0, action_dim)

            next_obs, _, done, _ = env.step(action)
            next_obs = next_obs.astype(np.float32) / 255.0
            if next_obs.ndim == 2:
                next_obs = np.stack([next_obs] * 3, axis=-1)

            traj_obs.append(next_obs)
            traj_act.append(action)

            if done:
                break

        obs_arr = np.stack(traj_obs[:total_len])
        act_arr = np.array(traj_act[: total_len - 1], dtype=np.int64)

        # Pad if trajectory was cut short by termination
        if len(obs_arr) < total_len:
            pad_len = total_len - len(obs_arr)
            obs_arr = np.pad(
                obs_arr,
                ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )
            act_arr = np.pad(act_arr, (0, pad_len), mode="edge")

        # Actions need to be same length as observations: duplicate last action
        act_arr = np.concatenate([act_arr, [act_arr[-1]]])

        all_obs.append(obs_arr)
        all_act.append(act_arr)
        collected += 1
        env.close()

    # Stack and convert to tensors
    obs_full = torch.from_numpy(np.stack(all_obs)).float().to(device) / 255.0
    # (N, T, H, W, C) -> (N, T, C, H, W)
    obs_full = obs_full.permute(0, 1, 4, 2, 3)
    act_full = torch.from_numpy(np.stack(all_act)).long().to(device)

    return {
        "observations": obs_full,
        "actions": act_full,
    }


def load_trajectories_from_replay_buffer(
    replay_buffer_path: str,
    trajectory_length: int = 20,
    num_trajectories: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Load trajectories from a saved replay buffer npz file.

    Args:
        replay_buffer_path: Path to .npz file with replay buffer data.
        trajectory_length: Number of frames per trajectory.
        num_trajectories: Maximum number of trajectories to load.
        device: Target device.

    Returns:
        Dict with "observations" [N, T, C, H, W] and "actions" [N, T].
    """
    data = np.load(replay_buffer_path, allow_pickle=False)
    obs_arr = data["observations"]  # (N, H, W, C)
    act_arr = data["actions"]  # (N,)

    n_total = len(obs_arr)
    n_traj = min(num_trajectories, n_total // trajectory_length)

    all_obs = []
    all_act = []

    for i in range(n_traj):
        start = i * trajectory_length
        end = start + trajectory_length
        traj_obs = obs_arr[start:end]
        traj_act = act_arr[start:end]

        all_obs.append(traj_obs)
        all_act.append(traj_act)

    obs_full = torch.from_numpy(np.stack(all_obs)).float().to(device) / 255.0
    obs_full = obs_full.permute(0, 1, 4, 2, 3)
    act_full = torch.from_numpy(np.stack(all_act)).long().to(device).squeeze(-1)

    return {"observations": obs_full, "actions": act_full}
