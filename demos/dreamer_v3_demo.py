"""DreamerV3: train with a benchmark preset from the paper, then record it.

    # Visual Control: pixels in, the paper's 12M model and replay ratio 512.
    python demos/dreamer_v3_demo.py train  --benchmark dmc-visual --task cartpole-swingup
    python demos/dreamer_v3_demo.py record --run demos/runs/dreamer_v3

    # Proprioceptive Control: joint states in, no pixels.
    python demos/dreamer_v3_demo.py train --benchmark dmc-proprio \\
        --task cartpole-swingup --run demos/runs/dreamer_v3_proprio

    # Interrupted? Continue from the newest checkpoint.
    python demos/dreamer_v3_demo.py train --resume ...same arguments...

Every protocol setting comes from
:func:`simulusion.configs.dreamer_v3_benchmarks.benchmark_config`; the flags
only shorten the run (``--steps``) or shrink the model to fit a small GPU
(``--model-size``, ``--replay-ratio``).

``record`` writes into ``<run>/media``:

* ``dream.mp4`` (pixel runs) -- the real environment beside the world model's
  open-loop prediction from the same actions, after a few observed frames.
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
from _media import hstack, label, plot_curves, upscale, write_mp4  # noqa: E402
from dreamer_demo import _checkpoints, _eval_curve, _segments  # noqa: E402


def train(args: argparse.Namespace) -> None:
    import simulusion
    from simulusion.configs.dreamer_v3_benchmarks import benchmark_config

    run = Path(args.run).resolve()
    steps, logdir, restore = args.steps, run, {}
    if args.resume and _checkpoints(run):
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

    overrides = {}
    if args.model_size:
        overrides["model_size"] = args.model_size
    if args.replay_ratio:
        overrides["replay_ratio"] = args.replay_ratio
    if args.benchmark == "bsuite":
        # BSuite scores a run from its own CSV log. A resumed run cannot add to
        # it (the episode budget restarts), so BSuite refuses an existing log.
        overrides["bsuite_results_dir"] = str(run / "bsuite_csv")
    cfg = benchmark_config(
        args.benchmark,
        args.task,
        total_steps=steps,
        # Only what the run stores; the 5M default reserves memory up front.
        buffer_size=steps // 2 + 1000,
        checkpoint_interval=args.eval_every,
        test_interval=args.eval_every,
        test_episodes=args.eval_episodes,
        seed=args.seed,
        exp_name="demo",
        logdir=str(logdir),
        **overrides,
        **restore,
    )
    print(
        f"{args.benchmark}/{cfg.env}: model {cfg.model_size}, replay ratio "
        f"{cfg.replay_ratio} ({cfg.update_steps} updates per {cfg.collect_steps} "
        f"frames), action repeat {cfg.action_repeat}, {cfg.total_steps} frames"
    )
    agent = simulusion.create_model("dreamer-v3", config=cfg)
    agent.train()


def _frame(obs) -> np.ndarray:
    return np.asarray(obs["image"], dtype=np.float32).transpose(1, 2, 0) / 255.0


def _render(env, obs) -> np.ndarray:
    from simulusion.models.dreamer_v3 import _video_frame

    frame = _video_frame(env, obs)
    return np.asarray(frame)


def record(args: argparse.Namespace) -> None:
    import torch

    from simulusion.models.dreamer import make_env
    from simulusion.models.dreamer_v3 import DreamerV3

    run = Path(args.run)
    out = run / "media"
    ckpt = Path(args.checkpoint) if args.checkpoint else _checkpoints(run)[-1][1]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DreamerV3.from_pretrained(
        ckpt.parent, checkpoint_filename=ckpt.name, map_location=device
    )
    model.args.seed = args.seed
    env = make_env(model.args)
    print(f"Loaded {ckpt} ({model.args.env})")

    def step(action: torch.Tensor):
        action_np = action[0].cpu().numpy()
        obs, reward, done, info = env.step(action_np)
        executed = (
            info.get("action", action_np) if isinstance(info, dict) else action_np
        )
        prev = torch.as_tensor(np.asarray(executed, np.float32), device=model.device)
        return obs, float(reward), done, prev.reshape(1, -1)

    if "image" in model.obs_space:
        # Open-loop imagination beside the real environment.
        obs = env.reset()
        state = model.rssm.init_state(1, model.device)
        prev_action = model._initial_action()
        frames = []
        for t in range(args.context + args.horizon):
            observed = t < args.context
            with torch.no_grad():
                if observed:
                    state = model.observe(obs, state, prev_action, is_first=t == 0)
                else:
                    state = model.rssm.imagine_step(state, prev_action)
                features = model.rssm.get_feat(state)
                action = model.actor(features, deterministic=True)
                dream = model.obs_decoder(features)["image"][0]
            dream_np = dream.float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            caption = "observed" if observed else f"imagined t+{t - args.context + 1}"
            frames.append(
                hstack(
                    [
                        label(upscale(_frame(obs), 4), "Real environment"),
                        label(
                            upscale(dream_np, 4),
                            f"World model: {caption}",
                            accent=not observed,
                        ),
                    ]
                )
            )
            obs, _, done, prev_action = step(action)
            if done:
                break
        print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    # The policy acting in the real environment, closed loop.
    obs = env.reset()
    state = model.rssm.init_state(1, model.device)
    prev_action = model._initial_action()
    frames, total = [], 0.0
    for t in range(args.policy_steps):
        with torch.no_grad():
            state, action = model.act_with_world_model(
                obs, state, prev_action, explore=False, is_first=t == 0
            )
        frame = _render(env, obs)
        scale = max(1, 256 // max(1, frame.shape[0]))
        frames.append(label(upscale(frame, scale), f"return {total:6.1f}"))
        obs, reward, done, prev_action = step(action)
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
            title=f"DreamerV3 on {model.args.env}",
            xlabel="environment steps",
            ylabel="episode return",
        )
        print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--benchmark", default="dmc-visual")
    t.add_argument("--task", default="cartpole-swingup")
    t.add_argument("--steps", type=int, default=200_000, help="environment frames")
    t.add_argument("--model-size", default=None, help="override the preset's size")
    t.add_argument(
        "--replay-ratio",
        type=int,
        default=None,
        help="override the preset's replay ratio to train faster",
    )
    t.add_argument("--eval-every", type=int, default=10_000)
    t.add_argument("--eval-episodes", type=int, default=3)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/dreamer_v3")
    t.add_argument(
        "--resume",
        action="store_true",
        help="continue from the run's newest checkpoint",
    )

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/dreamer_v3")
    r.add_argument("--checkpoint", default=None)
    r.add_argument("--context", type=int, default=5)
    r.add_argument("--horizon", type=int, default=45)
    r.add_argument("--policy-steps", type=int, default=500)
    r.add_argument("--fps", type=int, default=15)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--device", default=None)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
