"""`iris` (the agent, epoch by epoch) and `iris-wm` (its world model from data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.training.run import resolve_run_config, run
from simulusion.training.trainer import Trainer
from simulusion.world_model import load_world_model

from .conftest import write_episodes

TINY = {
    "vocab_size": 32,
    "tokens_per_frame": 16,
    "token_embedding_dim": 32,
    "encoder_channels": 16,
    "decoder_depth": 16,
    "transformer_layers": 1,
    "transformer_heads": 2,
    "transformer_embed_dim": 32,
    "transformer_timesteps": 4,
    "perceptual_weight": 0.0,
    "actor_hidden_size": 32,
    "use_amp": False,
    "gradient_checkpointing": False,
    "autoencoder_batch_size": 4,
    "transformer_batch_size": 2,
    "actor_critic_batch_size": 2,
    "imagination_horizon": 2,
    "burn_in_length": 2,
}


class FakeAtari:
    """Gymnasium-shaped image env with 6-step episodes and 3 actions."""

    class _Space:
        n = 3

    action_space = _Space()

    def __init__(self) -> None:
        self.t = 0

    def _frame(self) -> np.ndarray:
        return np.full((210, 160, 3), (self.t * 20) % 256, dtype=np.uint8)

    def reset(self, *args, **kwargs):
        self.t = 0
        return self._frame(), {}

    def step(self, action):
        self.t += 1
        return self._frame(), float(action == 1), self.t >= 6, False, {}


def _overrides(trainer_overrides: list[str], **extra) -> list[str]:
    model = {
        **TINY,
        "collection_epochs": 10,
        "env_steps_per_epoch": 30,
        "training_steps_per_epoch": 1,
        "transformer_steps_per_epoch": 1,
        "actor_critic_steps_per_epoch": 1,
        "start_autoencoder_after": 0,
        "start_transformer_after": 1,
        "start_actor_critic_after": 1,
        "eval_episodes": 1,
        **extra,
    }
    return [*trainer_overrides, *(f"model.{k}={v}" for k, v in model.items())]


def _fit(overrides: list[str]):
    cls, trainer_cfg, cfg, resolved = resolve_run_config(
        overrides=overrides, task="iris"
    )
    return Trainer(trainer_cfg, run_config=resolved).fit(cls(cfg, env=FakeAtari()))


def test_iris_trains_by_epoch_resumes_and_exports(
    tmp_path: Path, trainer_overrides
) -> None:
    pytest.importorskip("cv2")
    args = _overrides(trainer_overrides) + [
        "trainer.max_steps=3",
        "trainer.eval_every=3",
    ]
    first = _fit(args)
    assert first.progress == 3
    assert "eval_mean_return" in first.metrics
    assert (first.checkpoint / "task" / "replay.npz").exists()

    resumed = _fit(args[:-2] + ["trainer.max_steps=4", "trainer.resume=auto"])
    assert resumed.progress == 4 and resumed.iterations == 4

    model = load_world_model(resumed.world_model_path, device="cpu")
    assert model.spec.family == "iris" and model.spec.action.n == 3
    state = model.encode(torch.rand(1, 1, 3, 64, 64))
    out = model.rollout(state, torch.tensor([[0, 1]]))
    assert out.frames.shape == (1, 2, 3, 64, 64) and out.rewards.shape == (1, 2)


def test_iris_world_model_trains_from_discrete_episodes(
    tmp_path: Path, trainer_overrides
) -> None:
    data = write_episodes(tmp_path / "eps", count=4, length=8, size=64, actions=3)
    args = [
        *trainer_overrides,
        *(f"model.{k}={v}" for k, v in TINY.items()),
        f"data.path={data.as_posix()}",
        "data.batch_size=2",
        "data.val_fraction=0.25",
        "trainer.max_steps=2",
        "trainer.eval_every=2",
    ]
    result = run(overrides=args, task="iris-wm")
    assert {"val_recon_loss", "val_token_loss"} <= set(result.metrics)
    model = load_world_model(result.world_model_path, device="cpu")
    assert model.spec.action.n == 3
    state = model.encode(torch.rand(2, 2, 3, 64, 64), torch.tensor([[1], [2]]))
    assert model.rollout(state, torch.tensor([[0, 2], [1, 1]])).frames.shape == (
        2,
        2,
        3,
        64,
        64,
    )

    resumed = run(
        overrides=[*args, "trainer.max_steps=3", "trainer.resume=auto"], task="iris-wm"
    )
    assert resumed.iterations == 3


def test_iris_world_model_rejects_continuous_actions(
    tmp_path: Path, trainer_overrides
) -> None:
    data = write_episodes(
        tmp_path / "eps", count=2, length=8, size=64, actions=("continuous", 2)
    )
    args = [
        *trainer_overrides,
        *(f"model.{k}={v}" for k, v in TINY.items()),
        f"data.path={data.as_posix()}",
        "data.batch_size=2",
        "action_dim=2",
        "trainer.max_steps=1",
    ]
    with pytest.raises(ValueError, match="discrete"):
        run(overrides=args, task="iris-wm")
