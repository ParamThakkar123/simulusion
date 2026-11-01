#!/usr/bin/env python3
"""Interactively play inside a trained DIAMOND world model.

Two modes (toggle with TAB):
  REAL  - agent or human drives the real Atari environment
  DREAM - agent or human drives inside the diffusion model's imagination

Controls:
  TAB        - toggle REAL / DREAM mode
  R          - reset episode
  Arrow keys - steer (UP / DOWN / LEFT / RIGHT)
  W/A/S/D    - steer (UP / LEFT / DOWN / RIGHT)
  SPACE / X  - FIRE
  Z          - NOOP
  Q / ESC    - quit

When no human key is pressed the actor-critic policy takes over (AGENT control).
Hold a mapped key to take control (HUMAN control).

Usage:
    python -m simulusion.inference.play_diamond --checkpoint path/to/checkpoint.pt --game Breakout-v5
    python -m simulusion.inference.play_diamond --checkpoint path/to/checkpoint.pt --record gameplay.mp4
"""

import argparse
import time
from typing import Any

import cv2
import numpy as np
import torch

from simulusion.configs.diamond_config import DiamondConfig
from simulusion.inference.play_base import (
    get_action_from_key,
    init_video_recorder,
    resolve_checkpoint_path,
)
from simulusion.training.train_diamond import DiamondAgent, _normalize_frame
from simulusion.utils.device import default_device_name

ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


def _as_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def to_display_frame(frame: np.ndarray) -> np.ndarray:
    """Model-domain frame in [-1, 1] -> float RGB in [0, 1] for display/recording."""
    return np.clip((frame + 1.0) * 0.5, 0.0, 1.0)


def imagine_next_frame(
    agent: DiamondAgent, obs_history: list[np.ndarray], action_history: list[int]
) -> np.ndarray:
    """Sample the world model's next frame, in the model's [-1, 1] domain.

    ``obs_history`` holds HWC frames in [-1, 1], the domain the diffusion model
    and policy are trained in (``_normalize_frame``). ``action_history[-1]``
    must be the action taken at ``obs_history[-1]``: training conditions on
    ``actions[t-L+1..t]`` to predict ``obs[t+1]`` (``SequenceDataset``), so
    leaving the current action out conditions on a window one step stale.
    """
    cfg = agent.config
    n = cfg.num_conditioning_frames
    obs_np = np.stack(obs_history[-n:]).transpose(0, 3, 1, 2)
    obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(agent.device)

    act_hist = [_as_int(a) for a in action_history[-n:]]
    act_hist = [0] * (n - len(act_hist)) + act_hist
    act_tensor = torch.tensor(act_hist, device=agent.device).unsqueeze(0)

    with torch.no_grad():
        generated = agent.sampler.sample(
            model=agent.diffusion_model,
            shape=(1, obs_np.shape[1], cfg.obs_size, cfg.obs_size),
            device=agent.device,
            obs_history=obs_tensor,
            actions=act_tensor,
        )
    return generated.squeeze(0).permute(1, 2, 0).cpu().numpy()


def make_agent(
    checkpoint: str,
    game: str,
    device: str | None = None,
    seed: int = 42,
    sampling_steps: int | None = None,
) -> DiamondAgent:
    """Build a DIAMOND agent from a checkpoint for inference.

    ``sampling_steps`` overrides the checkpoint's ``num_sampling_steps``, the
    number of Euler denoising steps per imagined frame (the paper uses 3).
    """
    ckpt_path = resolve_checkpoint_path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict, dict):
        config = DiamondConfig(**cfg_dict)
    else:
        config = cfg_dict

    config.game = game
    config.seed = seed
    if device is not None:
        config.device = device
    config.device = config.device or default_device_name()
    config.terminate_on_life_loss = False
    if sampling_steps is not None:
        config.num_sampling_steps = sampling_steps

    agent = DiamondAgent(config)
    agent.load_checkpoint(ckpt_path)
    agent.actor_critic.eval()
    agent.diffusion_model.eval()
    agent.reward_model.eval()

    return agent


def run_play(
    checkpoint: str,
    game: str = "Breakout-v5",
    device: str | None = None,
    seed: int = 42,
    deterministic: bool = True,
    record: str | None = None,
    record_fps: int = 20,
    control: str = "assist",
    sampling_steps: int | None = None,
) -> None:
    agent = make_agent(checkpoint, game, device, seed, sampling_steps)
    device_obj = agent.device
    cfg = agent.config

    raw_obs, _ = agent.env.reset()
    norm_obs = _normalize_frame(raw_obs)
    obs_history = [norm_obs] * cfg.num_conditioning_frames
    action_history: list[int] = []

    policy_hidden: tuple[torch.Tensor, torch.Tensor] | None = (
        agent.actor_critic.init_hidden(1, device_obj)
    )
    dream_mode = False
    running = True
    episode_reward = 0.0
    step_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0
    control_mode = "AGENT"

    video_recorder = init_video_recorder(record, fps=record_fps)
    cv2.namedWindow("DIAMOND Play", cv2.WINDOW_NORMAL)

    def build_obs_tensor() -> torch.Tensor:
        obs_np = np.stack(obs_history[-cfg.num_conditioning_frames :])
        obs_np = obs_np.transpose(0, 3, 1, 2)
        return torch.from_numpy(obs_np).unsqueeze(0).to(device_obj)

    def reset_episode() -> None:
        nonlocal raw_obs, norm_obs, obs_history, action_history
        nonlocal policy_hidden, episode_reward, step_count
        raw_obs, _ = agent.env.reset()
        norm_obs = _normalize_frame(raw_obs)
        obs_history = [norm_obs] * cfg.num_conditioning_frames
        action_history = []
        policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
        episode_reward = 0.0
        step_count = 0

    while running:
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q") or key == 27:
            running = False
            continue

        if key == ord("\t"):
            dream_mode = not dream_mode
            print(f"Switched to {'DREAM' if dream_mode else 'REAL'} mode")
            policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
        if key == ord("r"):
            reset_episode()
            print("Reset episode")

        obs_tensor = build_obs_tensor()

        human_action = get_action_from_key(key)
        agent_action, policy_hidden = agent.actor_critic.get_action(
            obs_tensor[:, -1], policy_hidden, deterministic=deterministic
        )
        if control == "human":
            action = 0 if human_action is None else human_action
            control_mode = "HUMAN"
        elif control == "versus":
            # You take the env; the policy's choice is shown so you can play
            # against what the model would have done.
            action = 0 if human_action is None else human_action
            control_mode = "HUMAN vs AGENT"
        elif human_action is not None:
            action = human_action
            control_mode = "HUMAN"
            policy_hidden = agent.actor_critic.init_hidden(1, device_obj)
        else:
            control_mode = "AGENT"
            action = agent_action

        # Record the action before sampling: the world model conditions on the
        # action taken at the latest frame (see imagine_next_frame).
        action_history.append(action)

        if dream_mode:
            gen_np = imagine_next_frame(agent, obs_history, action_history)
            display_rgb = to_display_frame(gen_np)
            gen_u8 = (display_rgb * 255).astype(np.uint8)
            display_bgr = cv2.cvtColor(gen_u8, cv2.COLOR_RGB2BGR)

            obs_history.append(gen_np)

        else:
            next_raw, reward, done, _ = agent.env.step(action)
            next_norm = _normalize_frame(next_raw)

            display_rgb = next_raw.astype(np.float32) / 255.0
            display_bgr = next_raw
            if display_bgr.ndim == 3 and display_bgr.shape[2] == 3:
                display_bgr = cv2.cvtColor(display_bgr, cv2.COLOR_RGB2BGR)

            episode_reward += reward
            obs_history.append(next_norm)

            if done:
                print(
                    f"Episode finished. Reward: {episode_reward:.1f}, Steps: {step_count}"
                )
                reset_episode()

        step_count += 1

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        action_name = ACTION_NAMES.get(_as_int(action), str(action))
        agent_name = ACTION_NAMES.get(_as_int(agent_action), str(agent_action))
        mode_label = "DREAM" if dream_mode else "REAL"
        info_lines = [
            f"{mode_label}  {control_mode}  R: {episode_reward:.1f}  Step: {step_count}  FPS: {fps_display}",
            f"Action: {action_name} ({action})"
            + (f"  |  AGENT: {agent_name}" if control == "versus" else ""),
            "[TAB] toggle  [R] reset  [arrows/WASD] drive  [Q] quit",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                display_bgr,
                line,
                (5, 15 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        cv2.imshow("DIAMOND Play", display_bgr)

        if video_recorder is not None:
            rec_frame = (display_rgb * 255).astype(np.uint8)
            video_recorder.write_frame(rec_frame)

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    agent.env.close()
    print("Exited.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play inside a trained DIAMOND world model"
    )
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--game", "-g", default="Breakout-v5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--record", default=None, help="Path to save gameplay video (e.g. gameplay.mp4)"
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=20,
        help="FPS for recorded video (default: 20)",
    )
    parser.add_argument(
        "--control",
        choices=("assist", "human", "versus"),
        default="assist",
        help="assist: keys override the policy. human: you always drive. "
        "versus: you drive, the policy's action is shown as the opponent.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Euler denoising steps per dream frame (default: the checkpoint's).",
    )
    parser.add_argument(
        "--versus",
        action="store_true",
        help="Shortcut for --control versus.",
    )
    args = parser.parse_args()
    run_play(
        checkpoint=args.checkpoint,
        game=args.game,
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
        record=args.record,
        record_fps=args.record_fps,
        control="versus" if args.versus else args.control,
        sampling_steps=args.sampling_steps,
    )


if __name__ == "__main__":
    main()
