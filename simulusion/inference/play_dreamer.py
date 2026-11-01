#!/usr/bin/env python3
"""Interactively play inside a trained Dreamer world model.

Two modes (toggle with TAB):
  REAL  - agent or human drives the real environment
  DREAM - agent or human drives inside the RSSM's imagination, with frames
          reconstructed by the observation decoder

Controls:
  TAB        - toggle REAL / DREAM mode
  R          - reset episode
  Arrow keys - steer (first two action dimensions)
  W/A/S/D    - steer (first two action dimensions)
  SPACE      - zero action (hold still)
  Q / ESC    - quit

When no human key is pressed the learned actor takes over (AGENT control).
Hold a mapped key to take control (HUMAN control).

Dreamer environments use continuous actions, so keys nudge the first two
action dimensions to their limits rather than selecting a discrete action.

Usage:
    python -m simulusion.inference.play_dreamer --checkpoint path/to/checkpoint.pt --game walker-walk
    python -m simulusion.inference.play_dreamer --checkpoint path/to/checkpoint.pt --record dream.mp4
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from simulusion.utils.device import default_device_name

from simulusion.inference.play_base import init_video_recorder, resolve_checkpoint_path
from simulusion.configs import DreamerConfig
from simulusion.models.dreamer import make_env, preprocess_obs
from simulusion.models.dreamer_rssm import RSSM
from simulusion.vision import ActionDecoder, ConvDecoder, ConvEncoder

# Arrow key codes, matching simulusion/inference/play_base.py.
_ARROW_UP = 0x26
_ARROW_DOWN = 0x28
_ARROW_LEFT = 0x25
_ARROW_RIGHT = 0x27

# (action dimension, value) - continuous controls, so keys drive an axis.
KEY_TO_AXIS: dict[int, tuple[int, float]] = {
    ord("w"): (0, 1.0),
    ord("s"): (0, -1.0),
    ord("a"): (1, -1.0),
    ord("d"): (1, 1.0),
    _ARROW_UP: (0, 1.0),
    _ARROW_DOWN: (0, -1.0),
    _ARROW_LEFT: (1, -1.0),
    _ARROW_RIGHT: (1, 1.0),
}


def get_human_action(key: int, action_size: int) -> Optional[np.ndarray]:
    """Map a key to a continuous action vector, or None if unmapped."""

    if key == -1:
        return None
    axis = KEY_TO_AXIS.get(key)
    if axis is None:
        masked = key & 0xFF
        if masked != key:
            axis = KEY_TO_AXIS.get(masked)
        if axis is None:
            # SPACE means "hold still", which is a real action, not a no-op.
            if key in (ord(" "), ord(" ") & 0xFF):
                return np.zeros(action_size, dtype=np.float32)
            return None

    dim, value = axis
    action = np.zeros(action_size, dtype=np.float32)
    if dim < action_size:
        action[dim] = value
    return action


class DreamerPlayer:
    """The subset of a trained Dreamer needed to act and to dream.

    Rebuilt directly from the checkpoint rather than through ``Dreamer``, which
    would allocate a full replay buffer just to play back a policy.
    """

    def __init__(
        self,
        checkpoint: str,
        env_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        ckpt_path = resolve_checkpoint_path(checkpoint, model_dir="checkpoints/dreamer")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        config = ckpt.get("config", {})
        self.args = DreamerConfig(**config) if isinstance(config, dict) else config
        if env_name:
            self.args.env = env_name
        self.args.seed = seed

        resolved = device or default_device_name()
        self.device = torch.device(resolved)

        self.env = make_env(self.args)
        obs_shape = tuple(
            ckpt.get("obs_shape") or self.env.observation_space["image"].shape
        )
        self.action_size = int(
            ckpt.get("action_size") or self.env.action_space.shape[0]
        )

        self.rssm = RSSM(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            hidden_size=self.args.deter_size,
            obs_embed_size=self.args.obs_embed_size,
            activation=self.args.dense_activation_function,
        ).to(self.device)
        self.actor = ActionDecoder(
            action_size=self.action_size,
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            units=self.args.num_units,
            n_layers=4,
            activation=self.args.dense_activation_function,
        ).to(self.device)
        self.obs_encoder = ConvEncoder(
            input_shape=obs_shape,
            embed_size=self.args.obs_embed_size,
            activation=self.args.cnn_activation_function,
        ).to(self.device)
        self.obs_decoder = ConvDecoder(
            stoch_size=self.args.stoch_size,
            deter_size=self.args.deter_size,
            output_shape=obs_shape,
            activation=self.args.cnn_activation_function,
        ).to(self.device)

        missing = [
            key
            for key in ("rssm", "actor", "obs_encoder", "obs_decoder")
            if key not in ckpt
        ]
        if missing:
            raise ValueError(
                f"{ckpt_path} is not a Dreamer checkpoint - missing {missing}. "
                "Expected a file written by Dreamer.save()."
            )

        self.rssm.load_state_dict(ckpt["rssm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.obs_decoder.load_state_dict(ckpt["obs_decoder"])

        for module in (self.rssm, self.actor, self.obs_encoder, self.obs_decoder):
            module.eval()

    def features(self, state: dict) -> torch.Tensor:
        return torch.cat([state["stoch"], state["deter"]], dim=-1)

    def encode(self, obs: Any) -> torch.Tensor:
        image = obs["image"] if isinstance(obs, dict) else obs
        tensor = torch.tensor(np.array(image), dtype=torch.float32)
        tensor = tensor.to(self.device).unsqueeze(0)
        return self.obs_encoder(preprocess_obs(tensor))

    def decode(self, state: dict) -> np.ndarray:
        """Reconstruct the RGB frame a latent state corresponds to."""

        image = self.obs_decoder(self.features(state)).mean
        image = image.squeeze(0).detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        # preprocess_obs maps [0, 255] to [-0.5, 0.5]; undo that for display.
        return np.clip(image + 0.5, 0.0, 1.0)


def _to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    frame = (np.clip(frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def _observation_frame(obs: Any) -> np.ndarray:
    image = obs["image"] if isinstance(obs, dict) else obs
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    return image / 255.0


def run_play(
    checkpoint: str,
    game: str = "walker-walk",
    device: Optional[str] = None,
    seed: int = 42,
    deterministic: bool = True,
    record: Optional[str] = None,
    record_fps: int = 20,
    control: str = "assist",
) -> None:
    player = DreamerPlayer(checkpoint, env_name=game, device=device, seed=seed)
    env = player.env

    obs = env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)

    dream_mode = False
    running = True
    episode_reward = 0.0
    step_count = 0
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0
    control_mode = "AGENT"

    video_recorder = init_video_recorder(record, fps=record_fps)
    cv2.namedWindow("Dreamer Play", cv2.WINDOW_NORMAL)

    def reset_episode() -> tuple[Any, dict, torch.Tensor]:
        nonlocal episode_reward, step_count
        episode_reward = 0.0
        step_count = 0
        return (
            env.reset(),
            player.rssm.init_state(1, player.device),
            torch.zeros(1, player.action_size, device=player.device),
        )

    while running:
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            running = False
            continue

        if key == ord("\t"):
            dream_mode = not dream_mode
            print(f"Switched to {'DREAM' if dream_mode else 'REAL'} mode")
        if key == ord("r"):
            obs, state, prev_action = reset_episode()
            print("Reset episode")
            continue

        human_action = get_human_action(key, player.action_size)

        with torch.no_grad():
            agent_action = player.actor(player.features(state), deter=deterministic)
            if control == "human":
                control_mode = "HUMAN"
                chosen = (
                    human_action
                    if human_action is not None
                    else np.zeros(player.action_size, dtype=np.float32)
                )
            elif control == "versus":
                control_mode = "HUMAN vs AGENT"
                chosen = (
                    human_action
                    if human_action is not None
                    else np.zeros(player.action_size, dtype=np.float32)
                )
            elif human_action is not None:
                control_mode = "HUMAN"
                chosen = human_action
            else:
                control_mode = "AGENT"
                chosen = None

            if dream_mode:
                # No environment step: the RSSM predicts the next latent, and
                # the decoder turns it back into a frame.
                if chosen is None:
                    action = agent_action
                else:
                    action = torch.tensor(chosen, device=player.device).unsqueeze(0)
                state = player.rssm.imagine_step(state, action)
                display_rgb = player.decode(state)
                prev_action = action
            else:
                state, _ = player.rssm.observe_step(
                    state, prev_action, player.encode(obs)
                )
                if chosen is None:
                    action = agent_action
                    action_np = action[0].cpu().numpy()
                else:
                    action_np = chosen
                    action = torch.tensor(chosen, device=player.device).unsqueeze(0)

                next_obs, reward, done, info = env.step(action_np)
                executed = (
                    info["action"]
                    if isinstance(info, dict) and "action" in info
                    else action_np
                )
                prev_action = torch.tensor(
                    np.asarray(executed, dtype=np.float32), device=player.device
                ).unsqueeze(0)
                episode_reward += float(reward)
                display_rgb = _observation_frame(next_obs)
                obs = next_obs

                if done:
                    print(
                        f"Episode finished. Reward: {episode_reward:.1f}, "
                        f"Steps: {step_count}"
                    )
                    obs, state, prev_action = reset_episode()

        step_count += 1
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        display_bgr = _to_bgr(display_rgb)
        mode_label = "DREAM" if dream_mode else "REAL"
        info_lines = [
            f"{mode_label}  {control_mode}  R: {episode_reward:.1f}  "
            f"Step: {step_count}  FPS: {fps_display}",
            f"Action: {np.round(prev_action[0].cpu().numpy(), 2).tolist()}",
            "[TAB] toggle  [R] reset  [arrows/WASD] drive  [Q] quit",
        ]
        if control == "versus":
            info_lines.insert(
                2,
                "AGENT: "
                + str(np.round(agent_action[0].detach().cpu().numpy(), 2).tolist()),
            )
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

        cv2.imshow("Dreamer Play", display_bgr)

        if video_recorder is not None:
            video_recorder.write_frame(
                (np.clip(display_rgb, 0, 1) * 255).astype(np.uint8)
            )

    if video_recorder is not None:
        video_recorder.close()
    cv2.destroyAllWindows()
    env.close()
    print("Exited.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play inside a trained Dreamer world model"
    )
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--game", "-g", default="walker-walk")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--record", default=None, help="Path to save gameplay video (e.g. dream.mp4)"
    )
    parser.add_argument(
        "--record-fps", type=int, default=20, help="FPS for recorded video"
    )
    parser.add_argument(
        "--control",
        choices=("assist", "human", "versus"),
        default="assist",
        help="assist: keys override the policy. human: you always drive. "
        "versus: you drive, the policy's action is shown as the opponent.",
    )
    parser.add_argument(
        "--versus", action="store_true", help="Shortcut for --control versus."
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
    )


if __name__ == "__main__":
    main()
