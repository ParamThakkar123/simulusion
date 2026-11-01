"""Dreamer: train on a DeepMind Control task, then record what it sees and dreams.

    python demos/dreamer_demo.py train  --env cartpole-swingup --steps 60000
    python demos/dreamer_demo.py record --run demos/runs/dreamer

    # DreamerV2 (two-hot symlog reward/value heads, balanced KL)
    python demos/dreamer_demo.py train  --algo dreamer-v2 --run demos/runs/dreamer_v2
    python demos/dreamer_demo.py record --run demos/runs/dreamer_v2

    # Interrupted? Continue from the newest checkpoint.
    python demos/dreamer_demo.py train --resume ...same arguments...

``record`` writes into ``<run>/media``:

* ``dream.mp4`` -- the real environment beside the world model's open-loop
  prediction. The model observes a few real frames, then its observations are
  cut off and it imagines the rest of the episode from the same actions the
  policy takes in the real environment. Where the two halves drift apart is
  exactly where the world model is wrong.
* ``policy.mp4`` -- the trained policy controlling the real environment.
* ``curve.png`` -- evaluation return over training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# Run from a clone without installing: the repo root provides `simulusion`.
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import hstack, label, plot_curves, read_metrics, upscale, write_mp4  # noqa: E402


def _segments(run: Path) -> list[dict]:
    """The run's training segments: the first run, then one per ``--resume``.

    Each segment is its own Dreamer log folder whose step counter starts at 0,
    so ``offset`` is the global environment step it continued from.
    """
    path = run / "segments.json"
    if path.exists():
        return json.loads(path.read_text())
    return [{"dir": ".", "offset": 0}]


def _checkpoints(run: Path) -> list[tuple[int, Path]]:
    """Every checkpoint of the run as (global step, path), oldest first."""
    found = []
    for seg in _segments(run):
        for ckpt in (run / seg["dir"]).glob("ckpts/*_ckpt.pt"):
            found.append((seg["offset"] + int(ckpt.stem.split("_")[0]), ckpt))
    return sorted(found)


def _latest_checkpoint(run: Path) -> Path:
    ckpts = _checkpoints(run)
    if not ckpts:
        raise SystemExit(f"No checkpoints under {run}")
    return ckpts[-1][1]


def train(args: argparse.Namespace) -> None:
    import simulusion

    run = Path(args.run).resolve()
    steps, logdir, restore = args.steps, run, {}
    if args.resume and _checkpoints(run):
        # Continue from the newest checkpoint: weights and optimizers are
        # restored, the replay buffer (not checkpointed) refills from fresh
        # experience, and training covers the remaining steps.
        done, ckpt = _checkpoints(run)[-1]
        if done >= args.steps:
            print(f"{run} already reached {done} steps")
            return
        segments = [s for s in _segments(run) if s["offset"] < done]
        segments.append({"dir": f"resume_{done}", "offset": done})
        (run / "segments.json").write_text(json.dumps(segments, indent=2))
        steps, logdir = args.steps - done, run / f"resume_{done}"
        restore = {"restore": True, "checkpoint_path": str(ckpt)}
        print(f"Resuming from {ckpt} ({done} steps done, {steps} to go)")

    cfg = simulusion.create_config(
        args.algo,
        env_backend=args.backend,
        env=args.env,
        action_repeat=args.action_repeat,
        image_size=(64, 64),
        batch_size=args.batch_size,
        train_seq_len=50,
        total_steps=steps,
        # global_step advances by action_repeat per env step, so every interval
        # below must be a multiple of it or its `% interval == 0` check never fires.
        seed_steps=args.action_repeat * 500,
        collect_steps=1000,
        update_steps=100,
        # The run stores steps // action_repeat transitions; the 800k default
        # would reserve gigabytes of RAM for an empty buffer.
        buffer_size=steps // args.action_repeat + 1000,
        checkpoint_interval=10_000,
        test_interval=5_000,
        test_episodes=1,
        seed=args.seed,
        exp_name="demo",
        logdir=str(logdir),
        **restore,
    )
    agent = simulusion.create_model(args.algo, config=cfg)
    agent.train()


def _eval_curve(run: Path) -> tuple[list[float], list[float]]:
    """Evaluation return against global step, stitched across segments."""
    segments = _segments(run)
    steps, returns = [], []
    for i, seg in enumerate(segments):
        end = segments[i + 1]["offset"] if i + 1 < len(segments) else float("inf")
        path = run / seg["dir"] / "metrics.jsonl"
        if not path.exists():
            continue
        for row in read_metrics(path):
            step = seg["offset"] + row.get("step", 0)
            if "eval_avg_reward" in row and step < end:
                steps.append(step)
                returns.append(row["eval_avg_reward"])
    return steps, returns


def _frame(obs) -> np.ndarray:
    image = obs["image"] if isinstance(obs, dict) else obs
    return np.asarray(image, dtype=np.float32).transpose(1, 2, 0) / 255.0


def record(args: argparse.Namespace) -> None:
    import torch

    from simulusion.inference.play_dreamer import DreamerPlayer

    run = Path(args.run)
    out = run / "media"
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run)
    player = DreamerPlayer(str(ckpt), device=args.device, seed=args.seed)
    print(f"Loaded {ckpt}")

    def step_env(action: torch.Tensor):
        action_np = action[0].cpu().numpy()
        obs, reward, done, info = player.env.step(action_np)
        executed = (
            info.get("action", action_np) if isinstance(info, dict) else action_np
        )
        prev = torch.tensor(
            np.asarray(executed, dtype=np.float32), device=player.device
        )
        return obs, float(reward), done, prev.unsqueeze(0)

    # Open-loop imagination beside the real environment.
    obs = player.env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)
    frames = []
    for t in range(args.context + args.horizon):
        observed = t < args.context
        with torch.no_grad():
            if observed:
                state, _ = player.rssm.observe_step(
                    state, prev_action, player.encode(obs)
                )
            else:
                state = player.rssm.imagine_step(state, prev_action)
            action = player.actor(player.features(state), deter=True)
            dream = player.decode(state)
        caption = "observed" if observed else f"imagined t+{t - args.context + 1}"
        frames.append(
            hstack(
                [
                    label(upscale(_frame(obs), 4), "Real environment"),
                    label(
                        upscale(dream, 4),
                        f"World model: {caption}",
                        accent=not observed,
                    ),
                ]
            )
        )
        obs, _, done, prev_action = step_env(action)
        if done:
            break
    print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    # The policy acting in the real environment, closed loop.
    obs = player.env.reset()
    state = player.rssm.init_state(1, player.device)
    prev_action = torch.zeros(1, player.action_size, device=player.device)
    frames, total = [], 0.0
    for _ in range(args.policy_steps):
        with torch.no_grad():
            state, _ = player.rssm.observe_step(state, prev_action, player.encode(obs))
            action = player.actor(player.features(state), deter=True)
        frames.append(label(upscale(_frame(obs), 4), f"return {total:6.1f}"))
        obs, reward, done, prev_action = step_env(action)
        total += reward
        if done:
            break
    print(f"policy return over {len(frames)} steps: {total:.1f}")
    print("wrote", write_mp4(frames, out / "policy.mp4", fps=args.fps))

    steps, returns = _eval_curve(run)
    if steps:
        plot_curves(
            {"eval return": (steps, returns)},
            out / "curve.png",
            title=f"{player.args.algo} on {player.args.env}",
            xlabel="environment steps",
            ylabel="episode return",
        )
        print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument(
        "--algo", choices=["dreamer", "dreamer-v1", "dreamer-v2"], default="dreamer"
    )
    t.add_argument("--env", default="cartpole-swingup")
    t.add_argument("--backend", default="dmc")
    t.add_argument("--action-repeat", type=int, default=8)
    t.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="sequences per update; lower it if MuJoCo rendering runs out of GPU memory",
    )
    t.add_argument("--steps", type=int, default=60_000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/dreamer")
    t.add_argument(
        "--resume",
        action="store_true",
        help="continue from the run's newest checkpoint instead of starting over",
    )

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/dreamer")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--context", type=int, default=5)
    r.add_argument("--horizon", type=int, default=45)
    r.add_argument("--policy-steps", type=int, default=125)
    r.add_argument("--fps", type=int, default=10)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--device", default=None)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
