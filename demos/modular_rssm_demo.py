"""ModularRSSM: swap the recurrent backbone of a world model and compare them.

    python demos/modular_rssm_demo.py train  --minutes-per-backbone 30
    python demos/modular_rssm_demo.py record --run demos/runs/modular_rssm

``ModularRSSM`` assembles a world model from an encoder, a decoder and a
backbone. This demo trains the same encoder/decoder with a GRU and with an
LSTM backbone on identical data (random-action DeepMind Control episodes),
then shows both predicting the future open loop next to the real frames.

``record`` writes into ``<run>/media``:

* ``dream.mp4`` -- real frames | GRU prediction | LSTM prediction.
* ``curve.png`` -- reconstruction loss for both backbones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import hstack, label, plot_curves, read_metrics, upscale, write_mp4  # noqa: E402

BACKBONES = ("gru", "lstm")
FREE_NATS = 3.0
_NLL_CONSTANT = 0.5 * 3 * 64 * 64 * float(np.log(2 * np.pi))


def collect(env_name: str, episodes: int, action_repeat: int, seed: int):
    """Random-action episodes as uint8 frames (E, T+1, 3, 64, 64) and actions (E, T, A)."""
    from dm_control import suite

    domain, task = env_name.split("-", 1)
    env = suite.load(domain, task, task_kwargs={"random": seed})
    spec = env.action_spec()
    rng = np.random.default_rng(seed)
    all_frames, all_actions = [], []
    for _ in range(episodes):
        env.reset()
        frames = [env.physics.render(64, 64, camera_id=0)]
        actions = []
        for _ in range(1000 // action_repeat):
            action = rng.uniform(spec.minimum, spec.maximum).astype(np.float32)
            for _ in range(action_repeat):
                env.step(action)
            frames.append(env.physics.render(64, 64, camera_id=0))
            actions.append(action)
        all_frames.append(np.stack(frames).transpose(0, 3, 1, 2))
        all_actions.append(np.stack(actions))
    return np.stack(all_frames), np.stack(all_actions)


def _build(backbone: str, action_size: int):
    import simulusion

    return simulusion.create_model(
        "modular-rssm",
        obs_shape=(3, 64, 64),
        action_size=action_size,
        backbone_type=backbone,
        stoch_size=30,
        deter_size=200,
        embed_size=1024,
        hidden_size=200,
    )


def _to_input(frames):
    """uint8 frames -> float in [-0.5, 0.5], the range the decoder is trained on."""
    return frames.float() / 255.0 - 0.5


def _loss(model, frames, actions):
    """ELBO on a (B, L+1) clip: reconstruct every frame, KL(posterior || prior).

    Equivalent to calling ``model.observe_step`` per timestep, but the encoder
    and decoder each run once over the whole clip instead of once per step.
    """
    import torch
    from torch.distributions import kl_divergence

    B, L = actions.shape[:2]
    obs = _to_input(frames).transpose(0, 1)  # (L+1, B, 3, 64, 64)
    acts = actions.transpose(0, 1)
    embeds = model.encoder(obs.flatten(0, 1)).view(L + 1, B, -1)
    state = model.init_state(B, obs.device)
    _, state = model.backbone(state, torch.zeros_like(acts[0]), embeds[0])
    features, kl = [], torch.zeros((), device=obs.device)
    for t in range(L):
        prior, state = model.backbone(state, acts[t], embeds[t + 1])
        features.append(torch.cat([state["stoch"], state["deter"]], dim=-1))
        kl_t = kl_divergence(
            model.get_dist(state["mean"], state["std"]),
            model.get_dist(prior["mean"], prior["std"]),
        )
        kl = kl + kl_t.clamp(min=FREE_NATS).mean()
    decoded = model.decode_observation(torch.cat(features))
    recon = -decoded.log_prob(obs[1:].flatten(0, 1)).mean()
    return recon, kl / L


def train(args: argparse.Namespace) -> None:
    import torch

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Collecting {args.episodes} random episodes of {args.env}...", flush=True)
    frames, actions = collect(args.env, args.episodes, args.action_repeat, args.seed)
    np.savez_compressed(run / "data.npz", frames=frames[-4:], actions=actions[-4:])
    train_frames, train_actions = frames[:-4], actions[:-4]  # hold out 4 episodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    E, T = train_actions.shape[:2]

    for backbone in BACKBONES:
        torch.manual_seed(args.seed)
        model = _build(backbone, actions.shape[-1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=6e-4, eps=1e-4)
        deadline = time.time() + args.minutes_per_backbone * 60
        step = 0
        while time.time() < deadline:
            ep = rng.integers(0, E, size=args.batch_size)
            start = rng.integers(0, T - args.seq_len + 1, size=args.batch_size)
            idx = start[:, None] + np.arange(args.seq_len + 1)[None]
            clip = torch.from_numpy(train_frames[ep[:, None], idx]).to(device)
            acts = torch.from_numpy(train_actions[ep[:, None], idx[:, :-1]]).to(device)
            recon, kl = _loss(model, clip, acts)
            opt.zero_grad(set_to_none=True)
            (recon + kl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                row = {
                    "backbone": backbone,
                    "step": step,
                    "recon": recon.item(),
                    "kl": kl.item(),
                }
                with (run / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(row, flush=True)
        torch.save(model.state_dict(), run / f"{backbone}.pt")


def record(args: argparse.Namespace) -> None:
    import torch

    run = Path(args.run)
    saved = json.loads((run / "args.json").read_text())
    data = np.load(run / "data.npz")
    frames, actions = data["frames"], data["actions"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for backbone in BACKBONES:
        model = _build(backbone, actions.shape[-1]).to(device).eval()
        model.load_state_dict(torch.load(run / f"{backbone}.pt", map_location=device))
        models[backbone] = model

    ep = args.episode
    real = torch.from_numpy(frames[ep]).to(device)
    acts = torch.from_numpy(actions[ep]).to(device)
    torch.manual_seed(args.seed)
    dreams = {}
    with torch.no_grad():
        for name, model in models.items():
            state = model.init_state(1, device)
            prev = torch.zeros(1, acts.shape[-1], device=device)
            out = []
            for t in range(args.context + args.horizon):
                if t < args.context:
                    _, state = model.observe_step(
                        state, prev, _to_input(real[t : t + 1])
                    )
                else:
                    state = model.imagine_step(state, prev)
                    state = {**state, "stoch": state["mean"]}  # the most likely future
                feat = torch.cat([state["stoch"], state["deter"]], dim=-1)
                image = model.decode_observation(feat).mean[0] + 0.5
                out.append(image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                prev = acts[t : t + 1]
            dreams[name] = out

    video = []
    for t in range(args.context + args.horizon):
        observed = t < args.context
        caption = "observed" if observed else f"imagined t+{t - args.context + 1}"
        panels = [label(upscale(frames[ep][t], 4), "Real environment")]
        for name in BACKBONES:
            panels.append(
                label(
                    upscale(dreams[name][t], 4),
                    f"{name.upper()}: {caption}",
                    accent=not observed,
                )
            )
        video.append(hstack(panels))
    out = run / "media"
    print("wrote", write_mp4(video, out / "dream.mp4", fps=args.fps))

    if not (run / "metrics.jsonl").exists():
        return
    rows = read_metrics(run / "metrics.jsonl")
    series = {}
    for name in BACKBONES:
        mine = [r for r in rows if r["backbone"] == name]
        # The NLL of a unit-variance Gaussian over 3x64x64 pixels includes the
        # constant 0.5 * 12288 * ln(2*pi) ~= 11291, which flattens the curve;
        # what remains is half the summed squared error.
        series[name.upper()] = (
            [r["step"] for r in mine],
            [r["recon"] - _NLL_CONSTANT for r in mine],
        )
    plot_curves(
        series,
        out / "curve.png",
        title=f"ModularRSSM backbones on {saved['env']}",
        xlabel="gradient steps",
        ylabel="reconstruction error (SSE / 2)",
    )
    print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--env", default="cartpole-swingup")
    t.add_argument("--action-repeat", type=int, default=4)
    t.add_argument("--episodes", type=int, default=60)
    t.add_argument("--minutes-per-backbone", type=float, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--seq-len", type=int, default=32)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/modular_rssm")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/modular_rssm")
    r.add_argument(
        "--episode", type=int, default=0, help="held-out episode index (0-3)"
    )
    r.add_argument("--context", type=int, default=5)
    r.add_argument("--horizon", type=int, default=45)
    r.add_argument("--fps", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
