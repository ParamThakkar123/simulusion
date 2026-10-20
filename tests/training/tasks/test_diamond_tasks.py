"""`diamond` (the agent, epoch by epoch) and `diamond-wm` (its world model from data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.training.run import resolve_run_config, run
from simulusion.training.trainer import Trainer
from simulusion.world_model import load_world_model

from .conftest import write_episodes

TINY = {
    "game": "Breakout-v5",
    "obs_size": 32,
    "preset": "small",
    "num_conditioning_frames": 4,
    "burn_in_length": 4,
    "imagination_horizon": 3,
    "num_sampling_steps": 1,
    "use_amp": False,
    "batch_size": 2,
    "training_steps_per_epoch": 1,
    "environment_steps_per_epoch": 20,
    "data_loader_num_workers": 0,
    "persistent_workers": False,
    "pin_memory": False,
    "max_episode_steps": 30,
}


class FakeDiamondEnv:
    """The DIAMOND wrapper's API: 4-tuple steps with terminated/truncated in info."""

    def __init__(self) -> None:
        from gymnasium.spaces import Discrete

        self.action_space = Discrete(3)
        self.t = 0

    def _frame(self) -> np.ndarray:
        return np.full((32, 32, 3), (self.t * 15) % 256, dtype=np.uint8)

    def reset(self, *args, **kwargs):
        self.t = 0
        return self._frame(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= 8
        return (
            self._frame(),
            float(action == 2),
            done,
            {"terminated": done, "truncated": False},
        )


def _overrides(trainer_overrides: list[str]) -> list[str]:
    return [*trainer_overrides, *(f"model.{k}={v}" for k, v in TINY.items())]


def _fit(overrides: list[str]):
    cls, trainer_cfg, cfg, resolved = resolve_run_config(
        overrides=overrides, task="diamond"
    )
    return Trainer(trainer_cfg, run_config=resolved).fit(cls(cfg, env=FakeDiamondEnv()))


def test_diamond_trains_by_epoch_resumes_and_exports(
    tmp_path: Path, trainer_overrides
) -> None:
    pytest.importorskip("gymnasium")
    args = _overrides(trainer_overrides) + [
        "trainer.max_steps=2",
        "trainer.eval_every=2",
    ]
    first = _fit(args)
    assert first.progress == 2 and "eval_reward" in first.metrics
    assert (first.checkpoint / "task" / "agent_replay.npz").exists()

    resumed = _fit(args[:-2] + ["trainer.max_steps=3", "trainer.resume=auto"])
    assert resumed.progress == 3
    # The replay came back with the checkpoint, from beside agent.pt.
    rows = [
        json.loads(r)
        for r in (resumed.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    train = [r for r in rows if r["step"] == 3 and "train/replay/size" in r]
    assert train and train[-1]["train/replay/size"] == 60.0

    model = load_world_model(resumed.world_model_path, device="cpu")
    assert model.spec.family == "diamond" and model.spec.action.n == 3
    state = model.encode(torch.rand(1, 4, 3, 32, 32))
    assert model.rollout(state, torch.tensor([[0, 2]])).frames.shape == (
        1,
        2,
        3,
        32,
        32,
    )


def test_diamond_world_model_trains_from_episodes(
    tmp_path: Path, trainer_overrides
) -> None:
    pytest.importorskip("gymnasium")
    data = write_episodes(tmp_path / "eps", count=4, length=10, size=32, actions=3)
    args = [
        *_overrides(trainer_overrides),
        f"data.path={data.as_posix()}",
        "data.batch_size=2",
        "data.val_fraction=0.25",
        "trainer.max_steps=2",
        "trainer.eval_every=2",
    ]
    result = run(overrides=args, task="diamond-wm")
    assert "val_reward_loss" in result.metrics
    model = load_world_model(result.world_model_path, device="cpu")
    assert model.spec.action.n == 3
    state = model.encode(torch.rand(2, 4, 3, 32, 32), torch.randint(0, 3, (2, 3)))
    assert model.rollout(state, torch.tensor([[0, 1], [2, 2]])).frames.shape == (
        2,
        2,
        3,
        32,
        32,
    )

    resumed = run(
        overrides=[*args, "trainer.max_steps=3", "trainer.resume=auto"],
        task="diamond-wm",
    )
    assert resumed.iterations == 3


def test_diamond_world_model_batch_matches_the_replay_windows() -> None:
    from simulusion.training.tasks.diamond import DiamondWorldModelTask

    frames = (
        torch.arange(5, dtype=torch.uint8).view(1, 5, 1, 1, 1).expand(1, 5, 3, 2, 2)
    )
    batch = DiamondWorldModelTask._batch(
        {
            "frames": frames.contiguous(),
            "actions": torch.tensor([[0, 1, 2, 1]]),
            "rewards": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            "terminated": torch.tensor([[False, False, False, True]]),
        }
    )
    assert batch["obs_seq"].shape == (1, 4, 3, 2, 2)
    # next_obs is the frame after the last conditioning frame.
    assert torch.allclose(
        batch["next_obs"], (torch.full((1, 3, 2, 2), 4.0) / 255.0) * 2 - 1
    )
    assert batch["action_seq"].tolist() == [[0, 1, 2, 1]]
    assert batch["dones"].tolist() == [[False, False, False, True]]
