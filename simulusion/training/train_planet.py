from typing import Any, Optional
import argparse
import pdb
import torch
import numpy as np
import torch.nn.functional as F
import os

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from simulusion.utils.utils import (
    preprocess_img,
    bottle,
)
from simulusion.training.precision import GradientUpdate, MixedPrecision


def train(
    memory: Any,
    rssm: Any,
    optimizer: Any,
    device: torch.device,
    N: int = 32,
    H: int = 50,
    beta: float = 1.0,
    grads: bool = False,
    update: Optional[GradientUpdate] = None,
    precision: Optional[MixedPrecision] = None,
) -> dict:
    """
    Training implementation as indicated in:
    Learning Latent Dynamics for Planning from Pixels
    arXiv:1811.04551

    (a.) The Standard Variational Bound Method
        using only single step predictions.

    ``update`` steps ``optimizer`` (the shared Trainer passes one at the run's
    precision); without it the step is plain fp32 with the gradient norm
    clipped at 1000, as it always was. ``precision`` sets the autocast for the
    loss.
    """
    if precision is None:
        precision = (
            update.precision if update is not None else MixedPrecision("fp32", device)
        )
    if update is None:
        update = GradientUpdate(
            optimizer, rssm.parameters(), precision, max_grad_norm=1000.0
        )
    free_nats = torch.ones(1, device=device) * 3.0
    (x, u, r, t), lengths = memory.sample(N, H, time_first=True)
    x, u, r, t = [
        torch.from_numpy(np.ascontiguousarray(arr)).to(device).float()
        for arr in (x, u, r, t)
    ]
    preprocess_img(x, depth=5)

    with precision.autocast():
        loss, kld_loss, rec_loss, rew_loss = _variational_bound(
            rssm, x, u, r, beta, free_nats
        )
    grad_norm = update(loss)

    metrics: dict[str, Any] = {
        "losses": {
            "kl": kld_loss.item(),
            "reconstruction": rec_loss.item(),
            "reward_pred": rew_loss.item(),
        },
    }
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if grads:
        metrics["grad_norms"] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }
    return metrics


def _variational_bound(
    rssm: Any,
    x: torch.Tensor,
    u: torch.Tensor,
    r: torch.Tensor,
    beta: float,
    free_nats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PlaNet's one-step variational bound over a time-major batch."""
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

    loss = beta * kld_loss + rec_loss + rew_loss
    return loss, kld_loss, rec_loss, rew_loss


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
    """PlaNet/RSSM training from the command line.

    Builds a :class:`~simulusion.models.planet.Planet` on ``--env`` and runs
    :meth:`Planet.train` (so the shared Trainer): ``--iters`` gradient steps
    per epoch, one exploration and one evaluation episode per epoch, videos
    and ``ckpt_<epoch>.pth`` checkpoints in ``--outdir``. The optimizer is the
    one this script always used: Adam at 1e-3, eps 1e-4, no weight decay,
    no learning-rate schedule.
    """
    args = build_parser().parse_args(argv)

    from simulusion.models.planet import Planet
    from simulusion.utils.device import resolve_device

    device = resolve_device(args.device or None)
    if device.type == "cpu":
        print("WARNING: no GPU (CUDA or MPS) available, using CPU")
    planet = Planet(
        args.env,
        bit_depth=args.bit_depth,
        device=device,
        memory_size=100,
        max_episode_steps=args.max_episode_steps,
        results_dir=args.outdir,
    )
    planet.optimizer = torch.optim.Adam(planet.rssm.parameters(), lr=1e-3, eps=1e-4)
    planet.train(
        epochs=args.epochs,
        steps_per_epoch=args.iters,
        save_every=args.checkpoint_interval,
        results_dir=args.outdir,
        scheduler_type="none",
    )

    if os.getenv("TRAIN_RSSM_DEBUG", "0") == "1":
        pdb.set_trace()


if __name__ == "__main__":
    main()
