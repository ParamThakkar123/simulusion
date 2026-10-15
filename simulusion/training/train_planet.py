from typing import Any
import argparse
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from functools import partial
import os

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from simulusion.utils.utils import (
    preprocess_img,
    bottle,
    TensorBoardMetrics,
    save_video,
    flatten_dict,
    postprocess_img,
    normalize_frames_for_saving,
)
from simulusion.memory.planet_memory import Memory, Episode
from simulusion.models.rssm import RecurrentStateSpaceModel
from simulusion.controller.rssm_policy import RSSMPolicy
from simulusion.controller.rollout_generator import RolloutGenerator


def train(
    memory: Any,
    rssm: Any,
    optimizer: Any,
    device: torch.device,
    N: int = 32,
    H: int = 50,
    beta: float = 1.0,
    grads: bool = False,
) -> dict:
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Variational Bound Method
        using only single step predictions.
    """
    free_nats = torch.ones(1, device=device) * 3.0
    (x, u, r, t), lengths = memory.sample(N, H, time_first=True)
    x, u, r, t = [
        torch.from_numpy(np.ascontiguousarray(arr)).to(device).float()
        for arr in (x, u, r, t)
    ]
    preprocess_img(x, depth=5)

    e_t = bottle(rssm.encoder, x)
    h_t, s_t = rssm.get_init_state(e_t[0])
    states, priors, posteriors, posterior_samples = [], [], [], []

    for i, a_t in enumerate(torch.unbind(u, dim=0)):
        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        m, s = posteriors[-1]
        posterior_samples.append(Normal(m, s).rsample())
        s_t = posterior_samples[-1]

    prior_mean = torch.stack([p[0] for p in priors])
    prior_std = torch.stack([p[1] for p in priors])
    posterior_mean = torch.stack([p[0] for p in posteriors])
    posterior_std = torch.stack([p[1] for p in posteriors])
    prior_dist = Normal(prior_mean, prior_std)
    posterior_dist = Normal(posterior_mean, posterior_std)
    states_stacked = torch.stack(states)
    posterior_samples_stacked = torch.stack(posterior_samples)

    rec_loss = (
        F.mse_loss(
            bottle(rssm.decoder, states_stacked, posterior_samples_stacked),
            x[1:],
            reduction="none",
        )
        .sum((2, 3, 4))
        .mean()
    )

    kld_loss = (
        kl_divergence(posterior_dist, prior_dist).sum(-1).clamp(min=free_nats).mean()
    )

    rew_loss = F.mse_loss(
        bottle(rssm.pred_reward, states_stacked, posterior_samples_stacked), r
    )

    optimizer.zero_grad()
    loss = beta * kld_loss + rec_loss + rew_loss
    loss.backward()  # type: ignore[no-untyped-call]
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000.0, norm_type=2)
    optimizer.step()

    metrics = {
        "losses": {
            "kl": kld_loss.item(),
            "reconstruction": rec_loss.item(),
            "reward_pred": rew_loss.item(),
        },
    }
    if grads:
        metrics["grad_norms"] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """CLI for the PlaNet trainer.

    Every default below is the value this script previously hard-coded, so a
    bare `python -m simulusion.training.train_planet` behaves as before. They are
    flags because nothing else could reach them: sweeps had no way to shorten a
    run or redirect its output.
    """
    parser = argparse.ArgumentParser(description="Train PlaNet/RSSM on pixels")
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--iters", type=int, default=150, help="Gradient steps per epoch."
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Must stay at or above the 50-step training trace, or the replay "
        "buffer has nothing long enough to sample and training raises.",
    )
    parser.add_argument("--bit-depth", type=int, default=5)
    parser.add_argument("--outdir", default="results/")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Epochs between checkpoints. This used to be a fixed 25 while the "
        "loop only ran 2 epochs, so no checkpoint was ever written.",
    )
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Example PlaNet/RSSM training script with rollout collection and evaluation.

    Builds environment/model/policy objects, iteratively trains on replayed
    episodes, and periodically saves videos and checkpoints.
    """
    args = build_parser().parse_args(argv)

    from simulusion.utils.utils import TorchImageEnvWrapper

    env = TorchImageEnvWrapper(args.env, bit_depth=args.bit_depth)
    from simulusion.utils.device import resolve_device

    device = resolve_device(args.device or None)
    if device.type == "cpu":
        print("WARNING: no GPU (CUDA or MPS) available, using CPU")
    rssm_model = RecurrentStateSpaceModel(env.action_size).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)

    policy = RSSMPolicy(
        model=rssm_model,
        planning_horizon=20,
        num_candidates=1000,
        num_iterations=10,
        top_candidates=100,
        device=device,
    )

    rollout_gen = RolloutGenerator(
        env,
        device,
        policy=policy,
        episode_gen=lambda: Episode(partial(postprocess_img, depth=args.bit_depth)),
        max_episode_steps=args.max_episode_steps,
    )

    mem = Memory(100)
    mem.append(rollout_gen.rollout_n(1, random_policy=True))
    res_dir = args.outdir
    os.makedirs(res_dir, exist_ok=True)
    summary = TensorBoardMetrics(f"{res_dir}/")

    for i in trange(args.epochs, desc="Epoch", leave=False):
        metrics: dict[str, Any] = {}
        for _ in trange(args.iters, desc="Iter ", leave=False):
            train_metrics = train(mem, rssm_model.train(), optimizer, device)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f"{k}_mean"] = np.array(metrics[k]).mean()

        summary.update(metrics)
        mem.append([rollout_gen.rollout_once(explore=True)])
        # rollout_eval returns (episode, frames, metrics, latents); unpacking
        # three raised ValueError at the end of the first epoch, so this loop
        # had never run to completion.
        eval_episode, eval_frames, eval_metrics, _ = rollout_gen.rollout_eval()
        mem.append([eval_episode])
        # normalize frames to (T,H,W,3) float in [0,1] before saving
        safe_frames = normalize_frames_for_saving(eval_frames)
        save_video(safe_frames, res_dir, f"vid_{i + 1}")
        summary.update(eval_metrics)

        if args.checkpoint_interval > 0 and (i + 1) % args.checkpoint_interval == 0:
            path = os.path.join(res_dir, f"ckpt_{i + 1}.pth")
            torch.save(rssm_model.state_dict(), path)
            print(f"Wrote {path}")

    if os.getenv("TRAIN_RSSM_DEBUG", "0") == "1":
        pdb.set_trace()


if __name__ == "__main__":
    main()
