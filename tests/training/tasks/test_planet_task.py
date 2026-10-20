"""The `planet` task: PlaNet's RSSM and CEM planner on the shared Trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simulusion.training.run import run
from simulusion.world_model import load_world_model


def test_planet_trains_resumes_and_exports(tmp_path: Path, trainer_overrides) -> None:
    pytest.importorskip("gymnasium")
    args = [
        *trainer_overrides,
        "model.env=Pendulum-v1",
        "model.max_episode_steps=12",
        "model.state_size=16",
        "model.latent_size=4",
        "model.planning_horizon=3",
        "model.num_candidates=8",
        "model.num_iterations=2",
        "model.top_candidates=2",
        "model.steps_per_epoch=2",
        "model.batch_size=2",
        "model.chunk_length=6",
        "trainer.max_steps=1",
    ]
    first = run(overrides=args, task="planet")
    assert first.progress == 1
    assert (first.checkpoint / "task" / "replay").is_dir()

    resumed = run(
        overrides=[*args, "trainer.max_steps=2", "trainer.resume=auto"], task="planet"
    )
    assert resumed.progress == 2 and resumed.iterations == 2

    model = load_world_model(resumed.world_model_path, device="cpu")
    assert model.spec.family == "planet"
    state = model.encode(torch.rand(1, 2, *model.spec.obs_shape))
    out = model.rollout(state, torch.zeros(1, 2, model.spec.action.n))
    assert out.frames.shape[:2] == (1, 2)
