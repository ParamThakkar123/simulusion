#!/usr/bin/env python3
"""Record an IRIS policy playing Atari, headlessly.

Why this exists separately from ``IRISAgent.load``: the checkpoints in this repo
predate a change to ``IRISDecoder``, which gained an ``index_to_embedding``
parameter. ``IRISAgent.load`` loads every component strictly, so it raises
``RuntimeError: Missing key(s) ... index_to_embedding.weight`` and nothing can be
demoed from those files.

The policy path (``forward_actor_critic`` -> ``cnn`` -> ``lstm`` ->
``actor_head``) does not touch the decoder or the tokenizer at all, so this
script loads *only* the components the policy needs. That is enough to record
the agent playing the real game. It is NOT enough for imagination/reconstruction
demos, which do need the decoder — those require a checkpoint from current code.

Usage:
    python demos/record_iris.py -c checkpoints/iris/checkpoint_0.pt
    python demos/record_iris.py -c checkpoints/iris/checkpoint_0.pt --game ALE/Pong-v5 \
        --episodes 3 --out demos/out/iris_pong.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from simulusion.configs.iris_config import IRISConfig
from simulusion.envs.ale_atari_env import make_atari_env
from simulusion.models.iris_agent import IRISAgent
from simulusion.utils.utils import StreamingVideoWriter

# Components the actor-critic path actually reads. Everything else in the
# checkpoint (encoder/decoder/transformer, optimizers) is deliberately skipped.
POLICY_COMPONENTS = ("cnn", "lstm", "actor_head", "critic_head")


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB frame to the model's input size and scale to [0, 1] CHW."""
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)


def load_policy(agent: IRISAgent, ckpt: dict[str, Any]) -> dict[str, Any]:
    """Load only the policy components. Returns a report of what was loaded."""
    report: dict[str, Any] = {"loaded": [], "missing": [], "failed": {}}
    for name in POLICY_COMPONENTS:
        if name not in ckpt:
            report["missing"].append(name)
            continue
        try:
            getattr(agent, name).load_state_dict(ckpt[name])
            report["loaded"].append(name)
        except RuntimeError as exc:
            report["failed"][name] = str(exc).splitlines()[0]

    report["epoch"] = ckpt.get("epoch")
    report["global_step"] = ckpt.get("global_step")
    return report


def read_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True``.

    Unpickling arbitrary objects can execute code, so an older file that embeds
    a pickled config is rejected rather than silently loaded unsafely. Re-save
    it from a trusted environment with a plain-dict config.
    """
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def infer_architecture(ckpt: dict[str, Any]) -> dict[str, int]:
    """Recover the architecture the checkpoint was trained with.

    A checkpoint does not carry the IRISConfig that produced it, and the current
    defaults have moved (``actor_layers`` is now 1, while these checkpoints were
    trained with 4). Building the agent from stock defaults therefore fails with
    an LSTM size mismatch. Read the real values off the tensors instead.
    """
    shape: dict[str, int] = {}

    actor_head = ckpt.get("actor_head", {})
    if "weight" in actor_head:
        shape["action_size"] = int(actor_head["weight"].shape[0])

    lstm = ckpt.get("lstm", {})
    layers = sum(1 for key in lstm if key.startswith("weight_ih_l"))
    if layers:
        shape["actor_layers"] = layers
    if "weight_hh_l0" in lstm:
        # weight_hh is (4 * hidden, hidden) for a torch LSTM.
        shape["actor_hidden_size"] = int(lstm["weight_hh_l0"].shape[1])

    return shape


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record an IRIS policy playing Atari, headlessly"
    )
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument(
        "--game",
        "-g",
        default="ALE/Pong-v5",
        help="Must match the checkpoint's action space.",
    )
    parser.add_argument("--episodes", "-n", type=int, default=2)
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="Safety cap per episode."
    )
    parser.add_argument("--out", default="demos/out/iris.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sticky-actions",
        type=float,
        default=IRISConfig.repeat_action_probability,
        help="Probability of repeating the previous action. Defaults to the "
        "value IRIS trains under (0.0, the Atari 100k protocol); make_atari_env "
        "itself defaults to 0.25, which is a different task.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # frameskip and sticky actions have to match training. make_atari_env
    # defaults to repeat_action_probability=0.25; IRIS trains at 0.0, and
    # recording under sticky actions shows the policy losing to dynamics it
    # never saw rather than to its own limits.
    env = make_atari_env(
        args.game,
        obs_type="rgb",
        frameskip=IRISConfig.action_repeat,
        repeat_action_probability=args.sticky_actions,
    )
    n_actions = int(env.action_space.n)

    ckpt = read_checkpoint(args.checkpoint)
    arch = infer_architecture(ckpt)

    ckpt_actions = arch.get("action_size")
    if ckpt_actions is not None and ckpt_actions != n_actions:
        print(
            f"Action-space mismatch: checkpoint has {ckpt_actions} actions, "
            f"{args.game} has {n_actions}. Pick the game this policy was trained on."
        )
        return 1

    config = IRISConfig()
    for field in ("actor_layers", "actor_hidden_size"):
        if field in arch and getattr(config, field) != arch[field]:
            print(
                f"  adjusting {field}: {getattr(config, field)} -> {arch[field]} "
                "(from checkpoint)"
            )
            setattr(config, field, arch[field])

    agent = IRISAgent(config=config, action_size=n_actions, device=device)

    print(f"Loading policy components from {args.checkpoint}")
    report = load_policy(agent, ckpt)
    print(f"  loaded:  {', '.join(report['loaded']) or 'nothing'}")
    if report["missing"]:
        print(f"  missing: {', '.join(report['missing'])}")
    for name, err in report["failed"].items():
        print(f"  FAILED {name}: {err}")
    if report["failed"] or not report["loaded"]:
        print("Policy did not load cleanly; refusing to record a meaningless demo.")
        return 1
    print(f"  checkpoint epoch={report['epoch']} step={report['global_step']}")

    agent.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = StreamingVideoWriter(str(out_path), fps=args.fps)

    total_frames = 0
    rewards: list[float] = []

    for episode in range(args.episodes):
        # Seed the first reset only; later episodes continue the same stream, so
        # --episodes 3 gives three different games rather than three identical ones.
        reset_kwargs = (
            {"seed": args.seed} if args.seed is not None and episode == 0 else {}
        )
        obs, _ = env.reset(**reset_kwargs)
        episode_reward = 0.0
        terminated = truncated = False
        for step in range(args.max_steps):
            # Record the full-resolution frame; the model sees the downscaled one.
            writer.write_frame(obs.astype(np.uint8))
            total_frames += 1

            frame = preprocess(obs, config.frame_height)
            tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
            action = agent.act(
                tensor, epsilon=args.epsilon, temperature=args.temperature
            )

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            episode_reward += float(reward)
            if terminated or truncated:
                break

        rewards.append(episode_reward)
        # Distinguish a finished game from one --max-steps cut short: the
        # reward of a truncated episode is not the policy's score.
        truncated_by_cap = not (terminated or truncated)
        suffix = f" (cut at --max-steps {args.max_steps})" if truncated_by_cap else ""
        print(
            f"  episode {episode + 1}/{args.episodes}: "
            f"reward={episode_reward:.1f}{suffix}"
        )

    writer.close()
    env.close()

    print(f"\nWrote {out_path} ({total_frames} frames)")
    print(f"Mean reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
