"""Complete World Model training pipeline for any Gym environment.

This script trains a complete World Model pipeline consisting of:
1. ConvVAE - Encodes observations into latent space
2. MDNRNN - Predicts future latent states given actions
3. Controller - Linear controller trained with CMA-ES

Usage:
    python train_world_model.py --env CarRacing-v3 --data_dir ./data --logdir ./results
    python train_world_model.py --env BipedalWalker-v3 --action_size 4  # if env loading fails

The script will:
1. Generate rollout data (if not already present)
2. Train VAE
3. Train MDNRNN
4. Train Controller
"""

import os
import argparse
from typing import Any, Optional, Sequence
import numpy as np
import multiprocessing as mp
from glob import glob
from tqdm import tqdm

import uuid

import torch
from simulusion.utils.device import default_device_name

from simulusion.configs.wm_config import (
    WMVAEConfig,
    WMMDNRNNConfig,
    WMControllerConfig,
)
from simulusion.training.train_convvae import train_convae
from simulusion.training.train_mdn_rnn import train_mdn_rnn
from simulusion.training.train_controller import train_controller
from simulusion.models.controller import Controller


def generate_rollouts(
    data_dir: str,
    env_name: str,
    num_rollouts: int = 1000,
    seq_len: int = 1000,
    num_workers: int = 8,
) -> None:
    """Generate random rollouts from the specified environment.

    Args:
        data_dir: Directory to save rollout files
        env_name: Name of the Gym environment
        num_rollouts: Total number of rollouts to generate
        seq_len: Maximum length per rollout
        num_workers: Number of parallel workers
    """
    os.makedirs(data_dir, exist_ok=True)

    existing = len(glob(os.path.join(data_dir, "*.npz")))
    if existing >= num_rollouts:
        print(f"Found {existing} existing rollouts, skipping generation.")
        return

    missing = num_rollouts - existing
    num_workers = max(1, min(num_workers, missing))
    print(f"Generating {missing} rollouts with {num_workers} workers...")

    try:
        # Spread the remainder too, so exactly `missing` rollouts are attempted.
        share, extra = divmod(missing, num_workers)
        args = [
            (data_dir, env_name, seq_len, share + (1 if i < extra else 0))
            for i in range(num_workers)
        ]

        if num_workers == 1:
            _generate_worker(args[0])
        else:
            with mp.Pool(num_workers) as pool:
                list(tqdm(pool.imap(_generate_worker, args), total=num_workers))

        print(f"Generated rollouts in {data_dir}")

    except KeyboardInterrupt:
        print("\nData generation interrupted by user (Ctrl+C)")
        print("Partial rollouts may have been saved.")
        raise


def _generate_worker(args: tuple) -> None:
    """Worker function for parallel rollout generation.

    Rollouts are recorded through the environment the controller later acts
    in (:func:`~simulusion.training.train_controller.make_controller_env`):
    64x64 RGB frames, rendered for environments whose observations are
    vectors, and actions as the controller emits them (for a discrete action
    space, +1 for the action taken and -1 for the others).
    """
    from simulusion.training.train_controller import make_controller_env

    data_dir, env_name, seq_len, num_rollouts = args
    env = make_controller_env(env_name)

    for i in range(num_rollouts):
        try:
            obs = env.reset()
            observations = []
            actions = []
            rewards = []
            terminals = []

            for _ in range(seq_len):
                observations.append(obs["image"].transpose(1, 2, 0))
                obs, reward, done, info = env.step(env.action_space.sample())
                actions.append(info["action"])
                rewards.append(reward)
                terminals.append(info.get("terminated", done))
                if done:
                    break

            if len(observations) < 10:
                continue

            filepath = os.path.join(data_dir, f"rollout_{uuid.uuid4().hex[:8]}.npz")
            np.savez(
                filepath,
                observations=np.array(observations, dtype=np.uint8),
                rewards=np.array(rewards, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                terminals=np.array(terminals, dtype=np.float32),
            )
        except Exception as e:
            print(f"Error in rollout {i}: {e}")
            continue

    env.close()


def env_action_size(env_name: str) -> int:
    """The action size the models see in ``env_name``.

    The size of a continuous action, or the number of choices of a discrete
    one (the controller emits a score per choice).
    """
    from simulusion.training.train_controller import make_controller_env

    env = make_controller_env(env_name)
    try:
        return int(env.action_space.shape[0])
    finally:
        env.close()


def run_training_pipeline(args: Any, action_size: int) -> None:
    """Execute the complete World Model training pipeline."""

    try:
        vae_config = WMVAEConfig(
            height=64,
            width=64,
            latent_size=args.latent_size,
            device=args.device,
            train_batch_size=args.vae_batch_size,
            num_epochs=args.vae_epochs,
            data_dir=args.data_dir,
            learning_rate=args.vae_lr,
            logdir=args.logdir,
            noreload=args.noreload,
            nosamples=args.nosamples,
        )

        mdrnn_config = WMMDNRNNConfig(
            latent_size=args.latent_size,
            action_size=action_size,
            hidden_size=args.rnn_hidden,
            gmm_components=args.gmm_components,
            device=args.device,
            batch_size=args.rnn_batch_size,
            seq_len=args.seq_len,
            num_epochs=args.rnn_epochs,
            data_dir=args.data_dir,
            learning_rate=args.rnn_lr,
            logdir=args.logdir,
            noreload=args.noreload,
            include_reward=True,
        )

        ctrl_config = WMControllerConfig(
            latent_size=args.latent_size,
            hidden_size=args.rnn_hidden,
            action_size=action_size,
            env_name=args.env,
            logdir=args.logdir,
            n_samples=args.ctrl_samples,
            pop_size=args.ctrl_pop_size,
            target_return=args.ctrl_target,
            max_workers=args.ctrl_workers,
            display=True,
            time_limit=args.ctrl_time_limit,
        )

        if args.stage in ["all", "vae"]:
            print("\n" + "=" * 50)
            print("STAGE 1: Training ConvVAE")
            print("=" * 50)
            train_convae(vae_config)

        if args.stage in ["all", "rnn"]:
            print("\n" + "=" * 50)
            print("STAGE 2: Training MDNRNN")
            print("=" * 50)
            use_precomputed = getattr(args, "precompute_latents", True)
            use_amp = getattr(args, "use_amp", True)
            train_mdn_rnn(
                vae_config,
                mdrnn_config,
                use_precomputed_latents=use_precomputed,
                use_amp=use_amp,
            )

        if args.stage in ["all", "ctrl"]:
            print("\n" + "=" * 50)
            print("STAGE 3: Training Controller")
            print("=" * 50)
            train_controller(ctrl_config)

        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Training interrupted by user (Ctrl+C)")
        print("Partial checkpoints may have been saved.")
        print("You can resume training later with --noreload=False")
        print("=" * 50)
        raise  # Re-raise to exit


def test_trained_model(
    logdir: str,
    env_name: str,
    action_size: Optional[int] = None,
    num_episodes: int = 5,
    *,
    time_limit: int = 1000,
    env: Any = None,
) -> list[float]:
    """Run the trained V + M + C in the environment; the episode returns.

    Reads ``<logdir>/{vae,mdrnn,ctrl}/best.tar``. The models' sizes come
    from the checkpoints, so ``action_size`` is only checked against them.
    Returns an empty list (after saying which) when a checkpoint is missing.

    Args:
        logdir: The pipeline's ``--logdir``.
        env_name: The environment to act in.
        action_size: The action size the models must have, if given.
        num_episodes: Episodes to run.
        time_limit: Steps per episode at most.
        env: An environment to use instead of making ``env_name`` (it is
            closed afterwards).
    """
    from simulusion.training.tasks.world_models import controller_episode
    from simulusion.training.train_controller import make_controller_env
    from simulusion.utils.device import get_default_device
    from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

    device = get_default_device()
    if device.type == "cpu":
        print("WARNING: no GPU (CUDA or MPS) available, using CPU")

    vae_file = os.path.join(logdir, "vae", "best.tar")
    rnn_file = os.path.join(logdir, "mdrnn", "best.tar")
    ctrl_file = os.path.join(logdir, "ctrl", "best.tar")

    for f in [vae_file, rnn_file, ctrl_file]:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            return []

    print("\nLoading trained models...")
    wm = MDRNNWorldModel.from_checkpoints(vae_file, rnn_file, device=device)
    cell: Any = getattr(wm, "cell")
    if action_size is not None and action_size != cell.actions:
        raise ValueError(
            f"The trained models act with {cell.actions} actions, not {action_size}."
        )
    ctrl = Controller(cell.latents, cell.hiddens, cell.actions).to(device)
    ctrl_state = torch.load(ctrl_file, map_location=device, weights_only=True)
    ctrl.load_state_dict(ctrl_state["state_dict"])
    ctrl.eval()

    if env is None:
        env = make_controller_env(env_name)

    print(f"\nRunning {num_episodes} episodes...")
    returns: list[float] = []
    try:
        for episode in range(num_episodes):
            total_reward = controller_episode(wm, ctrl, env, time_limit=time_limit)
            returns.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    except KeyboardInterrupt:
        print("\nTesting interrupted by user (Ctrl+C)")
    finally:
        env.close()
    return returns


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train World Model on any Gym environment"
    )

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gym environment name (e.g., CarRacing-v3, Pendulum-v1)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory to store rollout data",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device_name(),
        help="Device to use",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "vae", "rnn", "ctrl"],
        help="Training stage to run",
    )
    parser.add_argument(
        "--noreload", action="store_true", help="Don't reload existing checkpoints"
    )
    parser.add_argument(
        "--nosamples", action="store_true", help="Don't save VAE samples"
    )

    parser.add_argument(
        "--latent_size", type=int, default=32, help="VAE latent dimension"
    )
    parser.add_argument("--rnn_hidden", type=int, default=256, help="RNN hidden size")
    parser.add_argument(
        "--gmm_components", type=int, default=5, help="Number of GMM components"
    )
    parser.add_argument(
        "--seq_len", type=int, default=32, help="Sequence length for RNN"
    )

    parser.add_argument(
        "--vae_epochs", type=int, default=50, help="VAE training epochs"
    )
    parser.add_argument("--vae_batch_size", type=int, default=64, help="VAE batch size")
    parser.add_argument("--vae_lr", type=float, default=1e-3, help="VAE learning rate")

    parser.add_argument(
        "--rnn_epochs", type=int, default=30, help="RNN training epochs"
    )
    parser.add_argument("--rnn_batch_size", type=int, default=16, help="RNN batch size")
    parser.add_argument("--rnn_lr", type=float, default=1e-3, help="RNN learning rate")

    parser.add_argument(
        "--ctrl_pop_size", type=int, default=10, help="CMA-ES population size"
    )
    parser.add_argument(
        "--ctrl_samples", type=int, default=4, help="Samples per controller evaluation"
    )
    parser.add_argument(
        "--ctrl_workers", type=int, default=8, help="Parallel workers for controller"
    )
    parser.add_argument(
        "--ctrl_target",
        type=float,
        default=900.0,
        help="Target return to stop training",
    )
    parser.add_argument(
        "--ctrl_time_limit",
        type=int,
        default=1000,
        help="Max steps per episode for controller training",
    )
    parser.add_argument(
        "--action_size",
        type=int,
        default=None,
        help="Action space size (determined automatically if not provided)",
    )

    parser.add_argument(
        "--num_rollouts", type=int, default=1000, help="Number of rollouts to generate"
    )
    parser.add_argument(
        "--generate_only", action="store_true", help="Only generate data, don't train"
    )
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument(
        "--precompute_latents",
        action="store_true",
        default=True,
        help="Pre-compute VAE latents for memory-efficient RNN training",
    )
    parser.add_argument(
        "--no_precompute_latents",
        action="store_false",
        dest="precompute_latents",
        help="Disable pre-computed latents (slower, more memory)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision for faster training",
    )
    parser.add_argument(
        "--no_amp",
        action="store_false",
        dest="use_amp",
        help="Disable mixed precision training",
    )

    args = parser.parse_args(argv)

    # Set default directories based on environment
    if args.data_dir is None:
        args.data_dir = f"./data/{args.env.replace('-', '_').lower()}"
    if args.logdir is None:
        args.logdir = f"./results/{args.env.replace('-', '_').lower()}"

    # Determine action size from environment or argument
    if args.action_size is not None:
        action_size = args.action_size
    else:
        try:
            action_size = env_action_size(args.env)
        except Exception as e:
            print(
                f"Failed to load environment '{args.env}' to determine action size: {e}"
            )
            print("Please ensure the environment and its dependencies are installed.")
            print(
                "For example, for Box2D environments, run: pip install 'gymnasium[box2d]'"
            )
            print("Alternatively, specify --action_size manually.")
            import sys

            sys.exit(1)

    if args.test:
        vae_file = os.path.join(args.logdir, "vae", "best.tar")
        rnn_file = os.path.join(args.logdir, "mdrnn", "best.tar")
        ctrl_file = os.path.join(args.logdir, "ctrl", "best.tar")

        missing = []
        for name, f in [
            ("VAE", vae_file),
            ("MDNRNN", rnn_file),
            ("Controller", ctrl_file),
        ]:
            if not os.path.exists(f):
                missing.append((name, f))

        if missing:
            print("\nMissing trained models, running training pipeline first...")
            if (
                not os.path.exists(args.data_dir)
                or len(glob(os.path.join(args.data_dir, "*.npz"))) < 100
            ):
                generate_rollouts(
                    args.data_dir, args.env, num_rollouts=args.num_rollouts
                )

            os.makedirs(args.logdir, exist_ok=True)
            os.makedirs(os.path.join(args.logdir, "vae"), exist_ok=True)
            os.makedirs(os.path.join(args.logdir, "mdrnn"), exist_ok=True)
            os.makedirs(os.path.join(args.logdir, "ctrl"), exist_ok=True)

            run_training_pipeline(args, action_size)

        test_trained_model(args.logdir, args.env, action_size)
        return

    if (
        not os.path.exists(args.data_dir)
        or len(glob(os.path.join(args.data_dir, "*.npz"))) < 100
    ):
        generate_rollouts(args.data_dir, args.env, num_rollouts=args.num_rollouts)

    if args.generate_only:
        print("Data generation complete!")
        return

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "vae"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "mdrnn"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "ctrl"), exist_ok=True)

    run_training_pipeline(args, action_size)


if __name__ == "__main__":
    main()
