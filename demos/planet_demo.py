"""PlaNet: learn a latent dynamics model from pixels and plan in it with CEM.

    python demos/planet_demo.py train  --minutes 75
    python demos/planet_demo.py record --run demos/runs/planet

PlaNet has no policy network: every action comes from the cross-entropy method
searching over action sequences inside the learned model. The demo uses the
same DeepMind Control task as the Dreamer demo, so the two are comparable.

``record`` writes into ``<run>/media``:

* ``dream.mp4`` -- the real environment beside the model's open-loop
  prediction, given the planner's actions.
* ``policy.mp4`` -- the CEM planner controlling the real environment.
* ``curve.png`` -- evaluation return over training.
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

BIT_DEPTH = 5
# Smaller than PlaNet's default 1000 x 10 so planning runs in real time on a laptop GPU.
POLICY_CFG = dict(
    planning_horizon=12, num_candidates=300, num_iterations=5, top_candidates=30
)


class DMCPixelEnv:
    """DeepMind Control task as the (3, 64, 64) [-0.5, 0.5] tensors PlaNet expects."""

    def __init__(self, domain: str, task: str, action_repeat: int, seed: int) -> None:
        from dm_control import suite

        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self.action_repeat = action_repeat
        spec = self._env.action_spec()
        self.action_size = int(spec.shape[0])
        self._low, self._high = spec.minimum, spec.maximum
        self.max_episode_steps = 1000 // action_repeat
        self.last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def _obs(self):
        import torch

        from simulusion.utils.utils import preprocess_img

        frame = self._env.physics.render(64, 64, camera_id=0)
        self.last_frame = frame
        x = torch.from_numpy(frame.copy()).float().permute(2, 0, 1)
        preprocess_img(x, BIT_DEPTH)
        return x

    def reset(self):
        self._env.reset()
        return self._obs()

    def step(self, action):
        action = np.asarray(
            action.detach().cpu().numpy() if hasattr(action, "detach") else action,
            dtype=np.float64,
        ).reshape(-1)
        action = np.clip(action, self._low, self._high)
        reward, done = 0.0, False
        for _ in range(self.action_repeat):
            step = self._env.step(action)
            reward += step.reward or 0.0
            done = step.last()
            if done:
                break
        return self._obs(), reward, done, {}

    def sample_random_action(self):
        import torch

        return torch.from_numpy(
            np.random.uniform(self._low, self._high).astype(np.float32)
        )


def _make(args: argparse.Namespace, seed: int):
    from simulusion.models.planet import Planet

    domain, task = args.env.split("-", 1)
    env = DMCPixelEnv(domain, task, args.action_repeat, seed)
    planet = Planet(
        env=env,
        bit_depth=BIT_DEPTH,
        policy_cfg=POLICY_CFG,
        max_episode_steps=env.max_episode_steps,
        results_dir=str(args.run),
    )
    return planet, env


def train(args: argparse.Namespace) -> None:
    import torch

    from simulusion.training.train_planet import train as planet_step

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    planet, env = _make(args, args.seed)

    planet.warmup(n_episodes=args.seed_episodes, random_policy=True)
    deadline = time.time() + args.minutes * 60
    metrics_path = run / "metrics.jsonl"
    epoch = 0
    while time.time() < deadline:
        epoch += 1
        losses: dict[str, list[float]] = {}
        for _ in range(args.updates_per_epoch):
            out = planet_step(
                planet.memory,
                planet.rssm.train(),
                planet.optimizer,
                planet.device,
                N=32,
                H=50,
            )
            for key, value in out.items():
                if not isinstance(value, dict):
                    losses.setdefault(key, []).append(float(value))
        planet.rssm.eval()
        planet.memory.append([planet.rollout_gen.rollout_once(explore=True)])
        episode, _, eval_metrics, _ = planet.rollout_gen.rollout_eval()
        planet.memory.append([episode])
        row = {
            "epoch": epoch,
            "episodes": len(planet.memory.episodes),
            "env_steps": len(planet.memory.episodes)
            * env.max_episode_steps
            * env.action_repeat,
            "eval_return": float(eval_metrics["eval/episode_reward"]),
            "time": time.time(),
            **{k: float(np.mean(v)) for k, v in losses.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch}: eval return {row['eval_return']:.1f}", flush=True)
        torch.save(planet.rssm.state_dict(), run / "rssm.pth")


def record(args: argparse.Namespace) -> None:
    import torch

    run = Path(args.run)
    saved = json.loads((run / "args.json").read_text())
    for key in ("env", "action_repeat"):
        setattr(args, key, saved[key])
    planet, env = _make(args, args.seed)
    planet.rssm.load_state_dict(
        torch.load(run / "rssm.pth", map_location=planet.device)
    )
    planet.rssm.eval()
    rssm, policy, device = planet.rssm, planet.policy, planet.device
    out = run / "media"

    def show(x: torch.Tensor) -> np.ndarray:
        return (x.squeeze(0).clamp(-0.5, 0.5) + 0.5).permute(1, 2, 0).cpu().numpy()

    # Open loop: the planner acts on the real environment from real observations
    # (that is how it chooses actions), while a second copy of the model state
    # stops observing after `context` frames and only imagines.
    policy.reset()
    obs = env.reset()
    h = torch.zeros(1, rssm.state_size, device=device)
    s = torch.zeros(1, rssm.latent_size, device=device)
    a = torch.zeros(1, env.action_size, device=device)
    frames = []
    for t in range(args.context + args.horizon):
        observed = t < args.context
        with torch.no_grad():
            if observed:
                h, s = rssm.get_init_state(rssm.encoder(obs[None].to(device)), h, s, a)
            else:
                h = rssm.deterministic_state_fwd(h, s, a)
                s = rssm.state_prior(h)[0]
            dream = show(rssm.decoder(h, s))
            action = policy.poll(obs.to(device)).flatten()
        caption = "observed" if observed else f"imagined t+{t - args.context + 1}"
        frames.append(
            hstack(
                [
                    label(upscale(env.last_frame, 4), "Real environment"),
                    label(
                        upscale(dream, 4),
                        f"World model: {caption}",
                        accent=not observed,
                    ),
                ]
            )
        )
        obs, _, done, _ = env.step(action)
        a = action.view(1, -1).to(device)
        if done:
            break
    print("wrote", write_mp4(frames, out / "dream.mp4", fps=args.fps))

    policy.reset()
    obs, frames, total = env.reset(), [], 0.0
    for _ in range(env.max_episode_steps):
        with torch.no_grad():
            action = policy.poll(obs.to(device)).flatten()
        frames.append(label(upscale(env.last_frame, 4), f"return {total:6.1f}"))
        obs, reward, done, _ = env.step(action)
        total += reward
        if done:
            break
    print(f"planner return: {total:.1f}")
    print("wrote", write_mp4(frames, out / "policy.mp4", fps=args.fps))

    rows = read_metrics(run / "metrics.jsonl")
    plot_curves(
        {
            "eval return": (
                [r["env_steps"] for r in rows],
                [r["eval_return"] for r in rows],
            )
        },
        out / "curve.png",
        title=f"PlaNet on {args.env}",
        xlabel="environment steps",
        ylabel="episode return",
    )
    print("wrote", out / "curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--env", default="cartpole-swingup")
    t.add_argument("--action-repeat", type=int, default=8)
    t.add_argument("--minutes", type=float, default=75)
    t.add_argument("--seed-episodes", type=int, default=5)
    t.add_argument("--updates-per-epoch", type=int, default=100)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--run", default="demos/runs/planet")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/planet")
    r.add_argument("--context", type=int, default=5)
    r.add_argument("--horizon", type=int, default=45)
    r.add_argument("--fps", type=int, default=10)
    r.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
