"""The `genie` task: Genie trained by the shared Trainer from an episode folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.training.run import run
from simulusion.world_model import load_world_model

from .conftest import write_episodes

TINY = [
    "model.num_frames=4",
    "model.image_size=16",
    "model.tokenizer_vocab_size=16",
    "model.tokenizer_embedding_dim=8",
    "model.tokenizer_encoder_dim=16",
    "model.tokenizer_decoder_dim=16",
    "model.tokenizer_encoder_depth=1",
    "model.tokenizer_decoder_depth=1",
    "model.action_encoder_dim=16",
    "model.action_decoder_dim=16",
    "model.action_encoder_depth=1",
    "model.dynamics_dim=16",
    "model.dynamics_depth=1",
    "model.dynamics_num_heads=2",
    "model.warmup_steps=1",
    "model.max_steps=6",
    "data.batch_size=2",
]


def test_genie_trains_validates_resumes_and_exports(
    tmp_path: Path, trainer_overrides
) -> None:
    data = write_episodes(tmp_path / "clips", count=5, length=8)
    args = [
        *trainer_overrides,
        *TINY,
        f"data.path={data.as_posix()}",
        "data.val_fraction=0.4",
        "trainer.eval_every=2",
        "trainer.max_steps=3",
    ]
    first = run(overrides=args, task="genie")
    assert first.progress == 3
    rows = [
        json.loads(x)
        for x in (first.run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("eval/val_recon_loss" in r for r in rows)
    assert any(np.isfinite(r.get("train/total_loss", 0.0)) for r in rows)

    resumed = run(
        overrides=[*args, "trainer.max_steps=5", "trainer.resume=auto"], task="genie"
    )
    assert resumed.iterations == 5

    model = load_world_model(resumed.world_model_path, device="cpu")
    assert model.spec.family == "genie" and model.spec.obs_shape == (3, 16, 16)
    state = model.encode(torch.rand(2, 3, 3, 16, 16))
    out = model.rollout(state, torch.zeros(2, 2, dtype=torch.long))
    assert out.frames.shape == (2, 2, 3, 16, 16)


def test_genie_trains_in_place_on_a_lerobot_dataset(
    tmp_path: Path, trainer_overrides
) -> None:
    pytest.importorskip("av")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from tests.datasets.test_episodes import _make_v21

    _make_v21(tmp_path / "ds", [6, 6, 6], (pa, pq))
    result = run(
        overrides=[
            *trainer_overrides,
            *TINY,
            "data.source=lerobot",
            f"data.path={(tmp_path / 'ds').as_posix()}",
            "data.camera=top",
            "data.preload=true",
            "trainer.max_steps=2",
        ],
        task="genie",
    )
    assert result.progress == 2 and result.world_model_path is not None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_genie_trains_on_cuda(
    tmp_path: Path, trainer_overrides, precision: str
) -> None:
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU has no bf16")
    data = write_episodes(tmp_path / "clips", count=3, length=6)
    overrides = [o for o in trainer_overrides if o != "trainer.device=cpu"]
    result = run(
        overrides=[
            *overrides,
            *TINY,
            "trainer.device=cuda",
            f"trainer.precision={precision}",
            f"data.path={data.as_posix()}",
            "trainer.max_steps=2",
        ],
        task="genie",
    )
    assert np.isfinite(result.progress)
    model = load_world_model(result.world_model_path, device="cuda")
    state = model.encode(torch.rand(1, 2, 3, 16, 16, device="cuda"))
    assert model.rollout(state, torch.zeros(1, 1, dtype=torch.long)).frames.is_cuda
