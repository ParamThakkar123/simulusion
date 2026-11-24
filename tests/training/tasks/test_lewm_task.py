"""LeWorldModel's training task: episodes in, a ``lewm`` world model out."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simulusion.training.run import run
from simulusion.world_model import load_world_model

from .conftest import write_episodes

pytestmark = pytest.mark.timeout(180)

_TINY = [
    "model.img_size=28",
    "model.patch_size=14",
    "model.encoder_dim=32",
    "model.embed_dim=32",
    "model.encoder_depth=1",
    "model.encoder_heads=2",
    "model.projector_hidden_dim=16",
    "model.frameskip=2",
    "model.history_size=2",
    "model.predictor_depth=1",
    "model.predictor_heads=2",
    "model.predictor_dim_head=8",
    "model.predictor_mlp_dim=16",
    "model.sigreg_num_proj=16",
]


def test_lewm_trains_resumes_and_exports_a_world_model(
    tmp_path: Path, trainer_overrides
) -> None:
    data = write_episodes(
        tmp_path / "eps", count=6, length=12, size=16, actions=("continuous", 3)
    )
    args = [
        *trainer_overrides,
        *_TINY,
        f"data.path={data.as_posix()}",
        "data.batch_size=4",
        "data.val_fraction=0.34",
        "data.val_batches=1",
        "trainer.max_steps=2",
        "trainer.eval_every=2",
    ]
    result = run(overrides=args, task="lewm")
    assert result.progress == 2
    for key in ("val_pred_loss", "val_sigreg_loss", "val_straightness"):
        assert key in result.metrics

    model = load_world_model(result.world_model_path, device="cpu")
    assert model.spec.family == "lewm"
    # Actions are blocks of frameskip (2) raw 3-D actions.
    assert model.spec.action.n == 6
    # The z-score statistics come from the data: uniform(-1, 1) actions.
    assert model.action_std.min() > 0.4 and model.action_mean.abs().max() < 0.3
    state = model.encode(torch.rand(1, 2, 3, 16, 16), torch.zeros(1, 1, 6))
    out = model.rollout(state, torch.zeros(1, 3, 6), decode=False)
    assert out.state["emb"].shape == (1, 2, 32)

    resumed = run(
        overrides=[*args, "trainer.max_steps=3", "trainer.resume=auto"], task="lewm"
    )
    assert resumed.progress == 3
