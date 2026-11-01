#!/usr/bin/env python3
"""Record a fixed-length DIAMOND demo video without a display.

``scripts/play_diamond.py`` is interactive: it opens a ``cv2`` window and runs
until you press Q. That makes it unusable over SSH on a headless GPU box and
awkward for producing a reproducible clip. This script drives the same agent
non-interactively for a set number of steps and writes the videos to disk.

It produces up to three files:
  diamond_real.mp4         - the policy acting in the real environment
  diamond_dream.mp4        - the diffusion world model imagining forward from the same
                    conditioning frames, driven by the same policy
  diamond_side_by_side.mp4 - both of the above stitched horizontally (real | dream)

Usage:
    python demos/record_diamond.py -c checkpoints/diamond/checkpoint_0.pt
    python demos/record_diamond.py -c ckpt.pt --steps 400 --dream-steps 200 --scale 6
    python demos/record_diamond.py -c ckpt.pt --device cpu --out-dir demos/out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running this file directly puts demos/ on sys.path, not the repo root, so
# `simulusion` would not resolve from a checkout that has not been installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from simulusion.inference.play_diamond import (
    imagine_next_frame,
    make_agent,
    to_display_frame,
)
from simulusion.training.train_diamond import _normalize_frame
from simulusion.utils.utils import StreamingVideoWriter


def upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale so 64x64 frames are legible in a demo."""
    if scale <= 1:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(
        frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST
    )


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a float RGB frame in [0, 1] to uint8 without clipping surprises."""
    return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption into an RGB uint8 frame."""
    out = frame.copy()
    cv2.putText(
        out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> Path | None:
    """Write RGB uint8 frames to ``path``. Returns the path, or None if empty."""
    if not frames:
        return None
    writer = StreamingVideoWriter(str(path), fps=fps)
    for frame in frames:
        writer.write_frame(frame)
    writer.close()
    return path


def rollout(
    agent,
    steps: int,
    dream: bool,
    deterministic: bool,
    scale: int,
) -> tuple[list[np.ndarray], float]:
    """Roll the policy forward in the real env or inside the world model.

    Returns the captured frames and the accumulated reward (always 0.0 in dream
    mode, where there is no environment to score against).
    """
    cfg = agent.config
    device = agent.device

    # Frames stay in the model's [-1, 1] domain, which both the policy and the
    # diffusion model are trained on; they are mapped to [0, 1] only for video.
    raw_obs, _ = agent.env.reset()
    obs_history = [_normalize_frame(raw_obs)] * cfg.num_conditioning_frames
    action_history: list[int] = []
    policy_hidden = agent.actor_critic.init_hidden(1, device)

    frames: list[np.ndarray] = []
    episode_reward = 0.0
    started = time.time()

    for step in range(steps):
        obs_np = np.stack(obs_history[-cfg.num_conditioning_frames :]).transpose(
            0, 3, 1, 2
        )
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)

        with torch.no_grad():
            action, policy_hidden = agent.actor_critic.get_action(
                obs_tensor[:, -1], policy_hidden, deterministic=deterministic
            )

        # The world model conditions on the action taken at the latest frame,
        # so it goes into the history before sampling.
        action_history.append(int(action))

        if dream:
            next_frame = imagine_next_frame(agent, obs_history, action_history)
            obs_history.append(next_frame)
        else:
            next_raw, reward, done, _ = agent.env.step(action)
            next_frame = _normalize_frame(next_raw)
            episode_reward += float(reward)
            obs_history.append(next_frame)
            if done:
                # Keep recording across episode boundaries so the clip stays the
                # requested length rather than ending early.
                raw_obs, _ = agent.env.reset()
                reset_frame = _normalize_frame(raw_obs)
                obs_history = [reset_frame] * cfg.num_conditioning_frames
                action_history = []
                policy_hidden = agent.actor_critic.init_hidden(1, device)

        frames.append(upscale(to_uint8(to_display_frame(next_frame)), scale))

        if (step + 1) % 50 == 0:
            rate = (step + 1) / max(1e-6, time.time() - started)
            print(
                f"  {'dream' if dream else 'real'}: {step + 1}/{steps} ({rate:.1f} fps)"
            )

    return frames, episode_reward


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record DIAMOND real-env and imagination videos headlessly"
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to a .pt file.")
    parser.add_argument("--game", "-g", default="Breakout-v5", help="Atari game id.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300, help="Real env frames.")
    parser.add_argument(
        "--dream-steps",
        type=int,
        default=100,
        help="Imagination frames. Diffusion sampling is slow on CPU; set 0 to skip.",
    )
    parser.add_argument(
        "--out-dir", default="demos/out", help="Directory for the written videos."
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--scale", type=int, default=4, help="Nearest-neighbour upscale factor."
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Euler denoising steps per dream frame (default: the checkpoint's; "
        "the paper uses 3).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of taking the argmax.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = make_agent(
        args.checkpoint, args.game, args.device, args.seed, args.sampling_steps
    )
    print(
        f"Device: {agent.device}  preset: {agent.config.preset}  game: {args.game}  "
        f"sampling steps: {agent.config.num_sampling_steps}"
    )

    deterministic = not args.stochastic
    written: list[Path] = []

    real_frames: list[np.ndarray] = []
    if args.steps > 0:
        print(f"Recording {args.steps} real-environment frames...")
        real_frames, reward = rollout(
            agent, args.steps, False, deterministic, args.scale
        )
        print(f"  episode reward across clip: {reward:.1f}")
        path = write_video(
            out_dir / "diamond_real.mp4",
            [label(f, "REAL") for f in real_frames],
            args.fps,
        )
        if path:
            written.append(path)

    dream_frames: list[np.ndarray] = []
    if args.dream_steps > 0:
        print(f"Recording {args.dream_steps} imagined frames...")
        dream_frames, _ = rollout(
            agent, args.dream_steps, True, deterministic, args.scale
        )
        path = write_video(
            out_dir / "diamond_dream.mp4",
            [label(f, "DREAM") for f in dream_frames],
            args.fps,
        )
        if path:
            written.append(path)

    if real_frames and dream_frames:
        pairs = min(len(real_frames), len(dream_frames))
        combined = [
            np.concatenate(
                [label(real_frames[i], "REAL"), label(dream_frames[i], "DREAM")], axis=1
            )
            for i in range(pairs)
        ]
        path = write_video(out_dir / "diamond_side_by_side.mp4", combined, args.fps)
        if path:
            written.append(path)

    agent.env.close()
    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
