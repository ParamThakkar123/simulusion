"""DIAMOND: a diffusion world model of Atari, and an agent trained inside it.

    python demos/diamond_demo.py train  --epochs 21
    python demos/diamond_demo.py record --run demos/runs/diamond

``record`` writes into ``<run>/media``:

* ``dream.mp4`` -- the real game beside the diffusion model's open-loop
  prediction. Both start from the same four real frames and receive the same
  actions; every frame on the right is sampled by the denoiser from the
  previous four *generated* frames.
* ``play_in_dream.mp4`` -- the policy playing entirely inside the world model:
  it acts on generated frames, and the model generates the response.
* ``curve.png`` -- diffusion loss over training (parsed from ``train.log``).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import hstack, label, plot_curves, upscale, write_mp4  # noqa: E402


def _checkpoints(run: Path) -> list[tuple[int, Path]]:
    """Every checkpoint of the run as (global epoch, path), oldest first.

    The trainer counts epochs from 0 in each segment, so a continued run lives
    in ``continue_<epoch>/`` and its epochs are offset by where it started.
    """
    found = []
    for ckpt in run.glob("checkpoint_*.pt"):
        found.append((int(ckpt.stem.split("_")[1]), ckpt))
    for seg in run.glob("continue_*"):
        offset = int(seg.name.split("_")[1])
        for ckpt in seg.glob("checkpoint_*.pt"):
            found.append((offset + int(ckpt.stem.split("_")[1]), ckpt))
    return sorted(found)


def train(args: argparse.Namespace) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if args.resume and _checkpoints(run):
        done, ckpt = _checkpoints(run)[-1]
        # Start one past the checkpoint's epoch; the continuation's epoch 0 is it.
        start = done + 1
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "continue",
            "--checkpoint",
            str(ckpt),
            "--out",
            str(run / f"continue_{start}"),
            "--game",
            args.game,
            "--epochs",
            str(args.epochs),
            "--steps-per-epoch",
            str(args.steps_per_epoch),
            "--save-interval",
            str(args.save_interval),
        ]
        print(f"Continuing from {ckpt} (epoch {done})", flush=True)
        with (run / "train.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=_HERE.parent
            )
        return
    cmd = [
        sys.executable,
        "-m",
        "simulusion.training.train_diamond",
        "--config",
        "simulusion/configs/experiments/diamond.yaml",
        f"game={args.game}",
        "preset=small",
        f"batch_size={args.batch_size}",
        f"num_epochs={args.epochs}",
        f"training_steps_per_epoch={args.steps_per_epoch}",
        "environment_steps_per_epoch=100",
        # Checkpoints are written when epoch % save_interval == 0, so with
        # epochs = k * interval + 1 the final epoch is saved.
        f"save_interval={args.save_interval}",
        f"eval_interval={args.save_interval}",
        "log_interval=1",
        "max_episode_steps=2000",
        # Windows spawns DataLoader workers, each pickling the whole replay buffer.
        "data_loader_num_workers=0",
        "pin_memory=false",
        "persistent_workers=false",
        f"checkpoint_dir={run.as_posix()}",
        f"seed={args.seed}",
    ]
    print(" ".join(cmd), flush=True)
    with (run / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=_HERE.parent
        )


def continue_training(args: argparse.Namespace) -> None:
    """Load a checkpoint (models, optimizers, replay buffer) and keep training."""
    import torch

    from simulusion.configs.diamond_config import DiamondConfig
    from simulusion.training.train_diamond import DiamondAgent

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = DiamondConfig(**ckpt["config"])
    config.game = args.game
    config.num_epochs = args.epochs
    config.training_steps_per_epoch = args.steps_per_epoch
    config.save_interval = args.save_interval
    config.eval_interval = args.save_interval
    config.checkpoint_dir = args.out
    if args.device:
        config.device = args.device
    agent = DiamondAgent(config)
    agent.load_checkpoint(args.checkpoint)
    print(
        f"Loaded {args.checkpoint}; replay holds {agent.replay_buffer.size} steps",
        flush=True,
    )
    agent.train()


def _latest_checkpoint(run: Path) -> Path:
    ckpts = _checkpoints(run)
    if not ckpts:
        raise SystemExit(f"No checkpoints in {run}")
    return ckpts[-1][1]


def record(args: argparse.Namespace) -> None:
    import torch

    from simulusion.inference.play_diamond import (
        imagine_next_frame,
        make_agent,
        to_display_frame,
    )
    from simulusion.training.train_diamond import _normalize_frame

    run = Path(args.run)
    out = run / "media"
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run)
    agent = make_agent(
        str(ckpt), args.game, seed=args.seed, sampling_steps=args.sampling_steps
    )
    device, n = agent.device, agent.config.num_conditioning_frames
    print(f"Loaded {ckpt}")

    def act(frame: np.ndarray, hidden):
        x = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            return agent.actor_critic.get_action(x, hidden, deterministic=False)

    # 1. Play the real game, recording frames and the actions taken at them.
    raw, _ = agent.env.reset()
    hidden = agent.actor_critic.init_hidden(1, device)
    real, actions, score = [_normalize_frame(raw)], [], 0.0
    for _ in range(args.warmup + n + args.horizon):
        action, hidden = act(real[-1], hidden)
        actions.append(action)
        raw, reward, done, _ = agent.env.step(action)
        score += reward
        real.append(_normalize_frame(raw))
        if done:
            break
    print(f"real rollout: {len(actions)} steps, score {score:.0f}")

    # 2. Open loop: start from four real frames, then feed back generated frames
    # while following the actions that were taken in the real game.
    start = min(args.warmup, max(0, len(actions) - n - 1))
    dream = list(real[start : start + n])
    frames = []
    for i in range(start + n - 1, min(len(actions), start + n - 1 + args.horizon)):
        nxt = imagine_next_frame(agent, dream, actions[start : i + 1])
        dream.append(nxt)
        step = i - (start + n - 1) + 1
        frames.append(
            hstack(
                [
                    label(upscale(to_display_frame(real[i + 1]), 5), "Real Atari"),
                    label(
                        upscale(to_display_frame(nxt), 5),
                        f"Diffusion world model t+{step}",
                        accent=True,
                    ),
                ]
            )
        )
    print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    # 3. The policy plays inside its own dream: no emulator involved at all.
    hidden = agent.actor_critic.init_hidden(1, device)
    for frame in real[start : start + n - 1]:
        _, hidden = act(frame, hidden)
    dream = list(real[start : start + n])
    taken = list(actions[start : start + n - 1])
    frames = []
    for step in range(args.dream_steps):
        action, hidden = act(dream[-1], hidden)
        taken.append(action)
        dream.append(imagine_next_frame(agent, dream, taken))
        frames.append(
            label(
                upscale(to_display_frame(dream[-1]), 5),
                f"Agent inside the dream {step + 1}",
                accent=True,
            )
        )
    print("wrote", write_mp4(frames, out / "play_in_dream.mp4", fps=args.fps))

    log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    epochs, offset, previous = [], 0, -1
    for m in re.findall(r"^Epoch (\d+):", log, flags=re.M):
        epoch = int(m)
        if epoch <= previous:  # a continued segment restarts its count
            offset += previous + 1
        epochs.append(offset + epoch)
        previous = epoch
    losses = [float(m) for m in re.findall(r"Diffusion loss: ([\d.]+)", log)]
    if losses:
        plot_curves(
            {"diffusion loss": (epochs[: len(losses)], losses)},
            out / "curve.png",
            title=f"DIAMOND on {args.game}",
            xlabel="epoch",
            ylabel="denoising loss",
        )
        print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--game", default="Breakout-v5")
    t.add_argument("--epochs", type=int, default=21)
    t.add_argument("--steps-per-epoch", type=int, default=100)
    t.add_argument("--save-interval", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/diamond")

    t.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint (models, optimizers, replay buffer)",
    )

    c = sub.add_parser("continue", help=argparse.SUPPRESS)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--game", default="Breakout-v5")
    c.add_argument("--epochs", type=int, default=21)
    c.add_argument("--steps-per-epoch", type=int, default=100)
    c.add_argument("--save-interval", type=int, default=5)
    c.add_argument("--device", default=None)

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/diamond")
    r.add_argument("--game", default="Breakout-v5")
    r.add_argument("--checkpoint", default=None)
    r.add_argument(
        "--warmup", type=int, default=30, help="real steps before the dream starts"
    )
    r.add_argument("--horizon", type=int, default=60)
    r.add_argument("--dream-steps", type=int, default=90)
    r.add_argument("--fps", type=int, default=12)
    r.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help="Euler denoising steps per frame (training used 3)",
    )
    r.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    {"train": train, "continue": continue_training, "record": record}[args.command](
        args
    )


if __name__ == "__main__":
    main()
