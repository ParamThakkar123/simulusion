"""Training a linear controller on latent + recurrent state with CMA-ES.

This module provides functions to train a linear controller using Covariance
Matrix Adaptation Evolution Strategy (CMA-ES). The controller maps latent
and hidden states to actions for the learned world model.

Reference:
    Ha & Schmidhuber (2018). Recurrent World Models Facilitate Policy Evolution.
    https://arxiv.org/abs/1805.11111
"""

from typing import Any
from os.path import join
import torch
import gymnasium as gym

from simulusion.configs.wm_config import WMControllerConfig
from simulusion.envs.gym_env import GymImageEnv


def make_controller_env(env_name: str) -> Any:
    """``env_name`` as the controller sees it: 64x64 image observations.

    Environments that take ``continuous`` (CarRacing) get the continuous
    action space the World Models controller outputs. Environments are made
    with ``render_mode="rgb_array"``, so those whose observations are vectors
    (Pendulum, BipedalWalker) are seen through their rendered frames.
    """
    try:
        env: Any = gym.make(env_name, continuous=True, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_name, render_mode="rgb_array")
    return GymImageEnv(env=env, size=(64, 64))


def train_controller(config: WMControllerConfig) -> None:
    """Train a linear controller using CMA-ES.

    Runs on the shared :class:`~simulusion.training.Trainer` through the
    ``controller`` task, with the VAE and MDN-RNN in ``<logdir>/vae/best.tar``
    and ``<logdir>/mdrnn/best.tar``. One step is one CMA-ES generation of
    ``pop_size`` candidates, each scored over ``n_samples`` episodes of up to
    ``time_limit`` steps in ``env_name``. The best controller so far is kept
    in ``<logdir>/ctrl/best.tar`` (and started from when it exists); training
    stops when CMA-ES converges or the best return passes ``target_return``.

    Candidates are scored one after another in this process, so
    ``max_workers`` and ``display`` no longer apply. The controller's sizes
    come from the trained models, whatever ``latent_size`` / ``hidden_size``
    / ``action_size`` say.

    Args:
        config: WMControllerConfig containing training hyperparameters,
                including env_name and action_size.
    """
    from simulusion.training.legacy import fit_legacy
    from simulusion.training.tasks.world_models import (
        ControllerTask,
        ControllerTaskConfig,
    )
    from simulusion.world_model.vae_mdrnn import MDRNNWorldModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDRNNWorldModel.from_checkpoints(
        join(config.logdir, "vae", "best.tar"),
        join(config.logdir, "mdrnn", "best.tar"),
        device=device,
    )

    ctrl_dir = join(config.logdir, "ctrl")
    fit_legacy(
        ControllerTask(
            ControllerTaskConfig(model=config),
            env_factory=lambda: make_controller_env(config.env_name),
            model=model,
            legacy_dir=ctrl_dir,
        ),
        ctrl_dir,
        # CMA-ES has no fixed number of generations; it runs until it
        # converges or reaches target_return (ControllerTask.should_stop).
        max_steps=10**9,
        device=device,
        eval_every=3,
    )
