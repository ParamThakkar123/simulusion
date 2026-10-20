"""The `genie` task: Genie trained by the shared Trainer from an episode folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
