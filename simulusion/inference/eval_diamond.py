#!/usr/bin/env python3
"""Evaluate a trained DIAMOND world model using FID, FVD, and LPIPS.

Usage:
    python -m simulusion.inference.eval_diamond --checkpoint path/to/checkpoint.pt \\
                                  --game Breakout-v5 \\
                                  --num_videos 1024

Metrics follow Appendix M of Alonso et al., "DIAMOND: Diffusion As a
Model Of eNvironment Dreams", NeurIPS 2024.
"""

import argparse
import json
import time
import torch
from simulusion.utils.device import default_device_name
import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm

from simulusion.configs.diamond_config import DiamondConfig
from simulusion.models.diffusion.diamond_diffusion import (
    DiffusionUNet,
    EulerSampler,
    EDMPreconditioner,
)
from simulusion.envs.diamond_atari import make_diamond_atari_env

from simulusion.evals import FID, FVD, LPIPS, PSNR
from simulusion.evals.diamond_utils import (
    generate_trajectories,
    collect_real_trajectories_from_env,
)


def load_diamond_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """Load a DIAMOND diffusion model and sampler from a checkpoint.

    Returns:
        Tuple of (diffusion_model, sampler, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    cfg_dict = checkpoint.get("config", {})
    if isinstance(cfg_dict, dict):
        config = DiamondConfig(**cfg_dict)
    else:
        config = cfg_dict  # already a DiamondConfig object

    config.device = str(device)

    action_dim = checkpoint.get("action_dim", 18)

    diffusion_model = DiffusionUNet(
        obs_channels=3,
        num_conditioning_frames=config.num_conditioning_frames,
        base_channels=config.diffusion_channels[0],
        channel_multipliers=tuple(
            int(c // config.diffusion_channels[0]) for c in config.diffusion_channels
        ),
        num_res_blocks=config.diffusion_res_blocks,
        cond_dim=config.diffusion_cond_dim,
        action_dim=action_dim,
    ).to(device)

    state_dict = checkpoint["diffusion_model"]
    # Strip potential "module." prefix from DDP saves
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    diffusion_model.load_state_dict(state_dict)
    diffusion_model.eval()

    edm_precond = EDMPreconditioner(
        sigma_data=config.sigma_data,
        p_mean=config.p_mean,
        p_std=config.p_std,
    )

    sampler = EulerSampler(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        num_steps=config.num_sampling_steps,
        edm_precond=edm_precond,
    )

    return diffusion_model, sampler, config, action_dim


def resolve_checkpoint_path(path: str) -> str:
    """Resolve a checkpoint path, searching common locations."""
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    alt = Path("checkpoints/diamond") / p
    if alt.exists():
        return str(alt.resolve())
    raise FileNotFoundError(f"Checkpoint not found at {path} or {alt}")


def run_eval(
    checkpoint: str,
    game: str = "Breakout-v5",
    num_videos: int = 256,
    trajectory_length: int = 20,
    batch_size: int = 16,
    device: str | None = None,
    output: str | None = None,
    seed: int = 42,
    metrics: list[str] | None = None,
    record: str | None = None,
) -> dict:
    """Run DIAMOND world model evaluation.

    Args:
        checkpoint: Path to DIAMOND checkpoint (.pt file).
        game: Atari game name.
        num_videos: Number of trajectories to generate and compare.
        trajectory_length: Frames per trajectory (including conditioning).
        batch_size: Batch size for trajectory generation.
        device: Device to run on (defaults to cuda if available).
        output: Path to save results JSON.
        seed: Random seed.
        metrics: List of metrics to compute ("fid", "fvd", "lpips").

    Returns:
        Dict of evaluation results.
    """
    if device is None:
        device = default_device_name()
    if metrics is None:
        metrics = ["fid", "fvd", "lpips"]

    device_obj = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Device: {device_obj}")
    print(f"Checkpoint: {checkpoint}")

    checkpoint_path = resolve_checkpoint_path(checkpoint or "")
    print(f"Resolved checkpoint: {checkpoint_path}")

    print("\nLoading DIAMOND diffusion model...")
    t0 = time.time()
    diffusion_model, sampler, config, action_dim = load_diamond_from_checkpoint(
        checkpoint_path, device_obj
    )
    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Game (from config): {config.game}")
    print(f"Action dim: {action_dim}")

    print(f"\nCollecting {num_videos} real trajectories...")
    t0 = time.time()

    def make_env() -> Any:
        return make_diamond_atari_env(
            game=game,
            frameskip=config.frameskip,
            max_noop=config.max_noop,
            terminate_on_life_loss=False,
            reward_clip=False,
            resize=(config.obs_size, config.obs_size),
            seed=seed,
        )

    real_trajs = collect_real_trajectories_from_env(
        env_fn=make_env,
        action_dim=action_dim,
        num_trajectories=num_videos,
        trajectory_length=trajectory_length,
        num_conditioning_frames=config.num_conditioning_frames,
        device=device_obj,
        seed=seed,
    )
    print(
        f"Collected {real_trajs['observations'].shape[0]} trajectories "
        f"({real_trajs['observations'].shape[1]} frames each) in "
        f"{time.time() - t0:.1f}s"
    )

    print(f"\nGenerating {num_videos} imagined trajectories...")
    t0 = time.time()

    horizon = trajectory_length - config.num_conditioning_frames
    pbar = tqdm(total=num_videos, desc="Generating")

    gen_trajs = generate_trajectories(
        diffusion_model=diffusion_model,
        sampler=sampler,
        real_trajectories=real_trajs,
        num_conditioning_frames=config.num_conditioning_frames,
        horizon=horizon,
        device=device_obj,
        batch_size=batch_size,
        progress_callback=lambda cur, total: pbar.update(
            min(batch_size, total - (cur - batch_size))
        ),
    )
    pbar.close()
    gen_time = time.time() - t0
    fps_value = (num_videos * horizon) / gen_time
    print(
        f"Generated {gen_trajs['observations'].shape[0]} trajectories "
        f"({gen_trajs['observations'].shape[1]} frames each) in "
        f"{gen_time:.1f}s ({fps_value:.1f} frames/s)"
    )

    real_frames = gen_trajs["real_observations"]
    B_f, H_f, C_f, H, W = real_frames.shape
    real_frames_flat = real_frames.reshape(-1, C_f, H, W)

    gen_frames = gen_trajs["observations"]
    gen_frames_flat = gen_frames.reshape(-1, C_f, H, W)

    print("\nFrames for evaluation:")
    print(f"  Real frames: {real_frames_flat.shape}")
    print(f"  Generated frames: {gen_frames_flat.shape}")

    results = {"game": game, "checkpoint": checkpoint}

    if "fid" in metrics:
        print("\nComputing FID...")
        t0 = time.time()
        fid = FID(device=device_obj, batch_size=batch_size)
        fid_score = fid(real_frames_flat, gen_frames_flat)
        results["FID"] = fid_score
        print(f"  FID: {fid_score:.2f}  ({time.time() - t0:.1f}s)")

    if "lpips" in metrics:
        print("\nComputing LPIPS...")
        t0 = time.time()
        lpips = LPIPS(device=device_obj, batch_size=batch_size)
        lpips_score = lpips(real_frames_flat, gen_frames_flat)
        results["LPIPS"] = lpips_score
        print(f"  LPIPS: {lpips_score:.4f}  ({time.time() - t0:.1f}s)")

    if "psnr" in metrics:
        print("\nComputing PSNR...")
        t0 = time.time()
        psnr = PSNR(batch_size=batch_size)
        psnr_score = psnr(real_frames_flat, gen_frames_flat)
        results["PSNR"] = psnr_score
        print(f"  PSNR: {psnr_score:.2f} dB  ({time.time() - t0:.1f}s)")

    if "fvd" in metrics:
        print("\nComputing FVD...")
        t0 = time.time()
        fvd = FVD(device=device_obj, batch_size=min(16, batch_size))
        real_videos = real_frames.permute(0, 2, 1, 3, 4)
        gen_videos = gen_frames.permute(0, 2, 1, 3, 4)
        fvd_score = fvd(real_videos, gen_videos)
        results["FVD"] = fvd_score
        print(f"  FVD: {fvd_score:.2f}  ({time.time() - t0:.1f}s)")

    if record:
        from simulusion.utils.utils import save_video

        record_path = Path(record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        real_vid_path = (
            record_path.parent / f"{record_path.stem}_real{record_path.suffix}"
        )
        gen_vid_path = (
            record_path.parent / f"{record_path.stem}_gen{record_path.suffix}"
        )
        save_video(real_frames, str(real_vid_path.parent), real_vid_path.stem)
        save_video(gen_frames, str(gen_vid_path.parent), gen_vid_path.stem)
        print(f"Saved real video: {real_vid_path}")
        print(f"Saved generated video: {gen_vid_path}")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for key in ["FID", "FVD", "LPIPS", "PSNR"]:
        if key in results:
            print(f"  {key}: {results[key]:.4f}")
    print("=" * 50)

    if output:
        output_path = output
    else:
        game_clean = game.replace("-v5", "").replace("-v0", "").lower()
        ckpt_name = Path(checkpoint).stem
        output_path = f"results/diamond_eval_{game_clean}_{ckpt_name}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DIAMOND world model with FID/FVD/LPIPS"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--game", type=str, default="Breakout-v5")
    parser.add_argument("--num_videos", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default=default_device_name(),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fid", "fvd", "lpips"],
        choices=["fid", "fvd", "lpips", "psnr"],
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Save real + generated videos to this path (.mp4)",
    )
    args = parser.parse_args()
    run_eval(**vars(args))


if __name__ == "__main__":
    main()
