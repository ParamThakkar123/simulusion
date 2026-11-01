import argparse
from typing import Any, Optional, Sequence
from collections import defaultdict
from pprint import pprint
import os
import pickle

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl_div
from tqdm import trange

from simulusion.utils.utils import (
    load_memory,
    get_combined_params,
    save_frames,
    plot_metrics,
    get_mask,
)

from simulusion.models.rssm import RecurrentStateSpaceModel
from simulusion.controller.rssm_policy import RSSMPolicy
from simulusion.controller.rollout_generator import RolloutGenerator
from simulusion.utils.utils import TorchImageEnvWrapper
from simulusion.memory.planet_memory import Memory

BIT_DEPTH = 5
FREE_NATS = 2
STATE_SIZE = 200
LATENT_SIZE = 30
EMBEDDING_SIZE = 1024


def train_rssm(
    memory: Any,
    model: Any,
    optimizer: Any,
    record_grads: bool = True,
    *,
    iterations: int = 10,
    batch_size: int = 32,
) -> dict:
    """Train an RSSM on replayed trajectories for one optimization phase.

    Samples ``iterations`` batches of ``batch_size`` sequences from memory,
    computes reconstruction and KL objectives across rollout steps, and
    returns the per-iteration losses.
    """
    model.train()
    metrics: dict[str, Any] = defaultdict(list)
    if record_grads:
        metrics["grads"] = defaultdict(list)
    device = next(model.parameters()).device
    for _ in trange(iterations, desc="# Epoch: ", leave=False):
        # sample and convert to torch tensors on the model device
        (x, u, _, _), lens = memory.sample(batch_size)
        x = torch.as_tensor(x).float().to(device)
        u = torch.as_tensor(u).float().to(device)
        # forward through model (expects tensors)
        states, priors, posteriors = model(x, u)
        prior_dists = [Normal(*p) for p in priors]
        posterior_dists = [Normal(*p) for p in posteriors]
        posterior_samples = [d.rsample() for d in posterior_dists]
        # Reconstruction loss
        rx = model.decoder(states[0], posterior_samples[0])
        iloss = (((x[:, 0] - rx) ** 2).sum((1, 2, 3))).mean()
        # KL Divergence
        kl = kl_div(prior_dists[0], posterior_dists[0]).sum(-1)
        kloss = kl.clamp(min=FREE_NATS).mean()
        mask = get_mask(u[..., 0], lens).T
        for i in range(1, len(states)):
            rx = model.decoder(states[i], posterior_samples[i])
            iloss += (((x[:, i] - rx) ** 2).sum((1, 2, 3)) * mask[i]).mean()
            kl = kl_div(prior_dists[i], posterior_dists[i]).sum(-1)
            kloss += (kl.clamp(min=FREE_NATS) * mask[i]).mean()

        kloss /= len(states)
        iloss /= len(states)
        optimizer.zero_grad()
        loss = iloss + kloss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100, norm_type=2)
        if record_grads:
            pprint(
                {
                    k: 0 if x.grad is None else x.grad.mean().item()
                    for k, x in dict(model.named_parameters()).items()
                }
            )

        metrics["kl_losses"].append(kloss.item())
        metrics["rec_losses"].append(iloss.item())
        optimizer.step()

    return metrics


def evaluate(memory: Any, model: Any, path: str, eps: Any) -> None:
    """Run one RSSM reconstruction/prediction evaluation and save visual outputs.

    Decodes priors/posteriors for a sampled sequence and writes frame grids
    for qualitative inspection.
    """
    model.eval()
    device = next(model.parameters()).device
    # sample and convert to torch tensors on the model device
    (x, u, _, _), lens = memory.sample(1)
    x = torch.as_tensor(x).float().to(device)
    u = torch.as_tensor(u).float().to(device)

    if x.dim() == 4:
        x = x.unsqueeze(0)
    if u.dim() == 2:
        u = u.unsqueeze(0)
    elif u.dim() == 1:
        u = u.unsqueeze(0).unsqueeze(-1)

    states, priors, posteriors = model(x, u)

    prior_means = [p[0] for p in priors]
    post_means = [p[0] for p in posteriors]

    pred1_list = [model.decoder(states[t], prior_means[t]) for t in range(len(states))]
    pred2_list = [model.decoder(states[t], post_means[t]) for t in range(len(states))]

    # stack => [T, B, C, H, W]; save_frames expects target [T+1, C, H, W] and preds [T, C, H, W]
    pred1 = torch.stack(pred1_list)  # [T, B, C, H, W]
    pred2 = torch.stack(pred2_list)

    # move preds to CPU and reduce batch dim for saving (use batch 0)
    save_frames(x[0].cpu(), pred1[:, 0].cpu(), pred2[:, 0].cpu(), f"{path}_{eps}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Standalone training loop for RSSM with generated replay fallback support.

    Initializes environment/policy/memory, trains over episodes, logs metrics,
    and periodically evaluates and checkpoints the model.
    """
    from simulusion.utils.device import get_default_device, resolve_device

    parser = argparse.ArgumentParser(description="Train a PlaNet RSSM.")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out_dir", default="results/test_rssm")
    parser.add_argument(
        "--data_dir",
        default=".",
        help="Where the replay files are read from, or generated into.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--state_size", type=int, default=STATE_SIZE)
    parser.add_argument("--latent_size", type=int, default=LATENT_SIZE)
    parser.add_argument("--embedding_size", type=int, default=EMBEDDING_SIZE)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--planning_horizon", type=int, default=12)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--warmup_episodes", type=int, default=50)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    args = parser.parse_args(argv)

    device = (
        resolve_device(args.device) if args.device is not None else get_default_device()
    )
    if device.type == "cpu":
        print("WARNING: no GPU (CUDA or MPS) available, using CPU")
    rssm = RecurrentStateSpaceModel(
        1, args.state_size, args.latent_size, args.embedding_size
    ).to(device)
    optimizer = torch.optim.Adam(get_combined_params(rssm), lr=1e-3)

    # create a planning policy that uses the trained rssm
    policy = RSSMPolicy(
        model=rssm,
        planning_horizon=args.planning_horizon,
        num_candidates=args.candidates,
        num_iterations=5,
        top_candidates=min(50, args.candidates),
        device=device,
    )
    env = TorchImageEnvWrapper(args.env, BIT_DEPTH)
    rollout_gen = RolloutGenerator(
        env, device, policy=policy, max_episode_steps=args.max_episode_steps
    )

    # fallback: if replay files don't exist, generate and save them with rollouts
    def ensure_memory(
        path: str, n_warmup: int = 25, mem_size: int = 1000, random_policy: bool = True
    ) -> Memory:
        if os.path.exists(path):
            return load_memory(path, device)
        print(f"{path} not found → generating {n_warmup} episodes with rollout_gen")
        mem = Memory(mem_size)
        eps = rollout_gen.rollout_n(n=n_warmup, random_policy=random_policy)
        mem.append(eps)
        with open(path, "wb") as f:
            pickle.dump(mem, f)
        mem.device = device  # type: ignore[attr-defined]

        return mem

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    test_data = ensure_memory(
        os.path.join(args.data_dir, "test_exp_replay.pth"),
        n_warmup=args.test_episodes,
        random_policy=True,
    )
    train_data = ensure_memory(
        os.path.join(args.data_dir, "train_exp_replay.pth"),
        n_warmup=args.warmup_episodes,
        random_policy=False,
    )
    # warm up training memory with a few rollouts
    new_eps = rollout_gen.rollout_n(n=5, random_policy=False)
    train_data.append(new_eps)

    global_metrics = defaultdict(list)
    for i in trange(args.episodes, desc="# Episode: ", leave=False):
        metrics = train_rssm(
            train_data,
            rssm,
            optimizer,
            record_grads=False,
            iterations=args.iterations,
            batch_size=args.batch_size,
        )
        for k, v in metrics.items():
            global_metrics[k].extend(metrics[k])
        plot_metrics(global_metrics, path=args.out_dir, prefix="TRAIN_")
        if (i + 1) % args.eval_every == 0:
            evaluate(test_data, rssm, os.path.join(args.out_dir, "eps"), i + 1)
        if (i + 1) % args.checkpoint_every == 0:
            torch.save(
                rssm.state_dict(), os.path.join(args.out_dir, f"ckpt_{i + 1}.pth")
            )


if __name__ == "__main__":
    main()
