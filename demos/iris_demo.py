"""IRIS: a discrete autoencoder + Transformer world model, and a policy trained in it.

    python demos/iris_demo.py train  --minutes 600
    python demos/iris_demo.py train  --minutes 600 --resume   # continue a stopped run
    python demos/iris_demo.py train  --until-epoch 200 --resume  # stop at a fixed epoch
    python demos/iris_demo.py record --run demos/runs/iris

The paper configuration is used unchanged, including its schedule: the
autoencoder trains alone until epoch 25, the Transformer joins it, and the
policy starts at epoch 50. Only the batch sizes (to fit a 4 GB GPU) and, if
asked, the gradient steps per epoch differ; the values used are written to
``<run>/args.json``.

Every few epochs the run saves the agent (weights and optimizers), its replay
buffer and its progress, so ``--resume`` continues where it stopped rather than
starting over.

``record`` writes into ``<run>/media``:

* ``tokens.mp4`` -- real frames beside their reconstruction from 16 discrete
  tokens, the only thing the world model ever sees.
* ``dream.mp4`` -- the policy acting inside the Transformer's imagination,
  starting from a real frame.
* ``policy.mp4`` -- the policy playing the real game.
* ``curve.png`` -- autoencoder and world-model losses over training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import hstack, label, plot_curves, read_metrics, upscale, write_mp4  # noqa: E402

# Batch sizes that fit a 4 GB GPU. Architecture, tokenizer, horizon, losses and
# the training schedule stay at the paper's values.
BATCH_OVERRIDES = dict(
    autoencoder_batch_size=64,
    transformer_batch_size=16,
    actor_critic_batch_size=16,
)
BUFFER_FIELDS = ("observations", "actions", "rewards", "terminals")


def _save_state(trainer, run: Path, epoch: int) -> None:
    """Checkpoint everything --resume needs; each file is replaced atomically."""
    buffer = trainer.replay_buffer
    filled = buffer.size if buffer.full else buffer.idx
    tmp = run / "replay.tmp.npz"
    np.savez(tmp, **{f: getattr(buffer, f)[:filled] for f in BUFFER_FIELDS})
    os.replace(tmp, run / "replay.npz")
    trainer.agent.save(str(run / "iris.tmp.pt"))
    os.replace(run / "iris.tmp.pt", run / "iris.pt")
    state = {
        "epoch": epoch,
        "env_steps": trainer.env_steps,
        "buffer": {k: getattr(buffer, k) for k in ("idx", "full", "steps", "episodes")},
    }
    (run / "state.json").write_text(json.dumps(state, indent=2))


def _load_state(trainer, run: Path) -> int:
    """Restore a run saved by _save_state; returns the epoch to continue from."""
    state = json.loads((run / "state.json").read_text())
    trainer.agent.load(str(run / "iris.pt"))
    data = np.load(run / "replay.npz")
    buffer = trainer.replay_buffer
    for field in BUFFER_FIELDS:
        values = data[field]
        getattr(buffer, field)[: len(values)] = values
    for key, value in state["buffer"].items():
        setattr(buffer, key, value)
    trainer.env_steps = state["env_steps"]
    return int(state["epoch"])


def _snapshot(run: Path, epoch: int, keep: int = 3) -> None:
    """Copy the latest checkpoint to ``snapshots/`` and keep the newest few.

    ``iris.pt`` is overwritten every save, so without snapshots a run whose
    policy collapses has nothing earlier to go back to.
    """
    import shutil

    folder = run / "snapshots"
    folder.mkdir(exist_ok=True)
    shutil.copy2(run / "iris.pt", folder / f"iris_epoch{epoch:04d}.pt")
    for old in sorted(folder.glob("iris_epoch*.pt"))[:-keep]:
        old.unlink()


def _reset_policy(agent) -> None:
    """Re-initialise the actor-critic and its optimizer; the world model stays."""
    for module in (agent.cnn, agent.lstm, agent.actor_head, agent.critic_head):
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    agent.ac_opt.state.clear()


def _release_cache_between_phases(agent) -> None:
    """Free PyTorch's cached GPU memory whenever training switches phase.

    The autoencoder, Transformer and imagination phases each leave their own
    peak cached, which together fill a 4 GB card; Windows then backs the
    overflow with system memory (8 GB of commit on the reference laptop) until
    the machine runs short. Training itself is unchanged.
    """
    import torch

    last: list[str | None] = [None]
    for name in ("update_autoencoder", "update_transformer", "imagine_rollout"):
        method = getattr(agent, name)

        def wrapped(*args, _method=method, _name=name, **kwargs):
            if last[0] != _name:
                torch.cuda.empty_cache()
                last[0] = _name
            return _method(*args, **kwargs)

        setattr(agent, name, wrapped)


def train(args: argparse.Namespace) -> None:
    from simulusion.configs.iris_config import IRISConfig
    from simulusion.training.train_iris import IRISTrainer

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    requested = {
        key: value
        for key, value in (
            ("training_steps_per_epoch", args.autoencoder_steps),
            ("transformer_steps_per_epoch", args.transformer_steps),
            ("actor_critic_steps_per_epoch", args.actor_critic_steps),
            ("actor_critic_batch_size", args.actor_critic_batch),
        )
        if value is not None
    }
    resuming = args.resume and (run / "state.json").exists()
    if resuming:
        # Keep the original run's settings, except those changed explicitly on
        # this command line (for example more policy work once the world model
        # has converged). Every change is recorded in args.json.
        saved = json.loads((run / "args.json").read_text())
        overrides = {**saved["overrides"], **requested}
        if requested:
            changes = saved.setdefault("changes", [])
            changes.append(
                {
                    "from_epoch": json.loads((run / "state.json").read_text())["epoch"],
                    **requested,
                }
            )
            saved["overrides"] = overrides
            (run / "args.json").write_text(json.dumps(saved, indent=2))
    else:
        overrides = {**BATCH_OVERRIDES, **requested}
        (run / "args.json").write_text(
            json.dumps({**vars(args), "overrides": overrides}, indent=2)
        )

    config = IRISConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    trainer = IRISTrainer(game=args.game, device="cuda", seed=args.seed, config=config)
    _release_cache_between_phases(trainer.agent)
    epoch = _load_state(trainer, run) if resuming else 0
    if resuming:
        print(f"Resuming at epoch {epoch} ({trainer.env_steps} env steps)", flush=True)
        if args.reset_policy:
            _reset_policy(trainer.agent)
            print("Policy re-initialised; the world model is kept", flush=True)

    deadline = time.time() + args.minutes * 60
    # --until-epoch is an absolute target, so a resumed run stops at the same
    # epoch however many times it was restarted; --minutes restarts its clock
    # on every launch.
    last_epoch = min(config.total_epochs, args.until_epoch or config.total_epochs)
    while time.time() < deadline and epoch < last_epoch:
        started = time.time()
        metrics = trainer.train_epoch(epoch)
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        epoch += 1
        if epoch % args.save_every == 0:
            _save_state(trainer, run, epoch)
        if args.snapshot_every and epoch % args.snapshot_every == 0:
            _snapshot(run, epoch)
    _save_state(trainer, run, epoch)


def record(args: argparse.Namespace) -> None:
    import cv2
    import torch

    from simulusion.configs.iris_config import IRISConfig
    from simulusion.envs.ale_atari_env import make_atari_env
    from simulusion.models.iris_agent import IRISAgent

    run = Path(args.run)
    out = run / "media"
    saved = json.loads((run / "args.json").read_text())
    ckpt = torch.load(run / "iris.pt", map_location="cpu", weights_only=True)
    config = IRISConfig.from_dict(ckpt["config"])
    config.perceptual_weight = 0.0  # skip building VGG16; it holds no saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = IRISAgent(config, action_size=ckpt["action_size"], device=device)
    agent.load(str(run / "iris.pt"))
    agent.eval()

    # The same emulator settings the trainer collects with.
    env = make_atari_env(
        saved["game"],
        obs_type="rgb",
        frameskip=config.action_repeat,
        repeat_action_probability=config.repeat_action_probability,
        max_episode_steps=config.max_episode_steps,
    )

    def to_input(raw: np.ndarray) -> torch.Tensor:
        size = (config.frame_width, config.frame_height)
        small = cv2.resize(np.asarray(raw), size, interpolation=cv2.INTER_LINEAR)
        return (
            torch.from_numpy(small).permute(2, 0, 1).float().div(255.0)[None].to(device)
        )

    def show(x: torch.Tensor) -> np.ndarray:
        return x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    raw, _ = env.reset(seed=args.seed)
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    score = 0.0
    policy_frames, token_frames, history = [], [], []
    for step in range(args.policy_steps):
        x = to_input(raw)
        with torch.no_grad():
            recon = agent.reconstruct(x)
            out_act = agent.act(recon, hidden=hidden, return_hidden=True)
            assert isinstance(out_act, tuple)
            action, hidden = out_act
        history.append(recon)
        policy_frames.append(label(upscale(np.asarray(raw), 2), f"return {score:+.0f}"))
        if step < args.token_steps:
            token_frames.append(
                hstack(
                    [
                        label(upscale(show(x), 5), "Real frame (64x64)"),
                        label(
                            upscale(show(recon), 5),
                            "Decoded from 16 tokens",
                            accent=True,
                        ),
                    ]
                )
            )
        raw, reward, terminated, truncated, _ = env.step(int(action.item()))
        score += float(reward)
        if terminated or truncated:
            break
    print(f"real game: {len(policy_frames)} steps, return {score:+.0f}")
    print("wrote", write_mp4(policy_frames, out / "policy.mp4", fps=args.fps))
    print("wrote", write_mp4(token_frames, out / "tokens.mp4", fps=args.fps))

    # Imagination from a real frame partway through the episode, with the
    # preceding 20 reconstructed frames as burn-in for the policy's LSTM.
    start = min(len(history) - 1, args.dream_start)
    burn_in = (
        torch.cat(history[max(0, start - 20) : start], dim=0)[None] if start else None
    )
    torch.manual_seed(args.seed)
    with torch.no_grad():
        traj = agent.imagine_rollout(
            history[start],
            horizon=args.dream_steps,
            burn_in_frames=burn_in,
            stop_on_termination=False,
        )
    dream = traj["frames"][0]
    rewards = traj["rewards"][0].cpu().numpy()
    frames, total = [], 0.0
    for t in range(dream.shape[0]):
        if t:
            total += float(rewards[t - 1])
        frames.append(
            label(
                upscale(dream[t].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), 5),
                f"Imagined step {t}  reward {total:+.1f}",
                accent=True,
            )
        )
    print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    # A resumed run re-logs the epochs after its last checkpoint; keep the latest.
    rows = list({r["epoch"]: r for r in read_metrics(run / "metrics.jsonl")}.values())
    series = {}
    for key, name in (
        ("recon_loss", "autoencoder recon"),
        ("token_loss", "next-token loss"),
    ):
        mine = [r for r in rows if r.get(key)]
        if mine:
            series[name] = ([r["epoch"] for r in mine], [r[key] for r in mine])
    plot_curves(
        series,
        out / "curve.png",
        title=f"IRIS on {saved['game']}",
        xlabel="epoch",
        ylabel="loss",
    )
    print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--game", default="ALE/Pong-v5")
    t.add_argument(
        "--minutes", type=float, default=600, help="time limit for this launch"
    )
    t.add_argument(
        "--until-epoch",
        type=int,
        default=None,
        help="stop once this epoch is reached; unlike --minutes, survives --resume",
    )
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/iris")
    t.add_argument("--resume", action="store_true", help="continue a stopped run")
    t.add_argument(
        "--save-every", type=int, default=1, help="epochs between checkpoints"
    )
    t.add_argument("--autoencoder-steps", type=int, default=None, help="paper: 200")
    t.add_argument("--transformer-steps", type=int, default=None, help="paper: 200")
    t.add_argument("--actor-critic-steps", type=int, default=None, help="paper: 200")
    t.add_argument(
        "--actor-critic-batch",
        type=int,
        default=None,
        help="imagined trajectories per policy update (default 16; paper: 64)",
    )
    t.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="epochs between kept snapshots in <run>/snapshots (newest 3; 0: off)",
    )
    t.add_argument(
        "--reset-policy",
        action="store_true",
        help="with --resume: start the policy afresh on the trained world model",
    )

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/iris")
    r.add_argument("--policy-steps", type=int, default=600)
    r.add_argument("--token-steps", type=int, default=120)
    r.add_argument("--dream-start", type=int, default=60)
    r.add_argument("--dream-steps", type=int, default=80)
    r.add_argument("--fps", type=int, default=15)
    r.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
