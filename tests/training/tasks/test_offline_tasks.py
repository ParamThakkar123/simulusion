"""Dataset-driven tasks that are not RL agents: dit, jepa, and World Models' V, M, C."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.training.run import resolve_run_config, run
from simulusion.training.trainer import Trainer
from simulusion.world_model import load_world_model

from .conftest import write_episodes


def _image_folder(root: Path, n: int = 12, size: int = 32) -> Path:
    from PIL import Image

    rng = np.random.default_rng(0)
    for cls in ("a", "b"):
        (root / "train" / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n // 2):
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
                root / "train" / cls / f"{i}.png"
            )
    return root


def test_dit_trains_and_exports_a_loadable_model(
    tmp_path: Path, trainer_overrides
) -> None:
    root = _image_folder(tmp_path / "imgs")
    args = [
        *trainer_overrides,
        "model.DATASET=imagefolder",
        f"model.ROOT_PATH={root.as_posix()}",
        "image_folder=train",
        "model.VAL_SPLIT=0.25",
        "model.BATCH=4",
        "model.IMG_SIZE=32",
        "model.PATCH=8",
        "model.WIDTH=32",
        "model.DEPTH=1",
        "model.HEADS=2",
        "model.TIMESTEPS=10",
        "model.NUM_WORKERS=0",
        "val_batches=1",
        "trainer.max_steps=3",
        "trainer.eval_every=3",
    ]
    result = run(overrides=args, task="dit")
    assert "val_loss" in result.metrics and result.world_model_path is None
    from simulusion.models.diffusion.DiT import DiT

    model = DiT.from_pretrained(result.model_path / "dit_model.pth").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32), torch.zeros(1, dtype=torch.long), None)
    assert out.shape[-2:] == (32, 32)

    resumed = run(
        overrides=[*args, "trainer.max_steps=4", "trainer.resume=auto"], task="dit"
    )
    assert resumed.iterations == 4


def test_jepa_trains_with_ema_target_and_resumes(
    tmp_path: Path, trainer_overrides
) -> None:
    root = _image_folder(tmp_path / "imgs", n=16)
    args = [
        *trainer_overrides,
        "model.dataset=imagefolder",
        f"model.root_path={root.as_posix()}",
        "model.image_folder=train",
        "model.model_name=vit_tiny",
        "model.crop_size=64",
        "model.patch_size=8",
        "model.pred_depth=1",
        "model.pred_emb_dim=48",
        "model.batch_size=4",
        "model.num_workers=0",
        "model.pin_mem=false",
        "model.min_keep=2",
        "model.epochs=2",
        "model.warmup=1",
        "trainer.max_steps=3",
    ]
    first = run(overrides=args, task="jepa")
    assert first.progress == 3
    state = torch.load(first.model_path / "ijepa.pth.tar", weights_only=True)
    assert {"encoder", "predictor", "target_encoder"} <= set(state)
    # The target encoder trails the online one (EMA), so they differ.
    key = next(k for k in state["encoder"] if k.endswith("weight"))
    assert not torch.equal(state["encoder"][key], state["target_encoder"][key])

    resumed = run(
        overrides=[*args, "trainer.max_steps=5", "trainer.resume=auto"], task="jepa"
    )
    assert resumed.iterations == 5


class _FrameEnv:
    """Only reset() is used in dream mode; step() is for env mode."""

    def __init__(self) -> None:
        self.t = 0

    def reset(self, *args, **kwargs):
        self.t = 0
        return {"image": np.full((3, 64, 64), 100, np.uint8)}, {}

    def step(self, action):
        self.t += 1
        return (
            {"image": np.full((3, 64, 64), 100, np.uint8)},
            float(np.sum(action)),
            self.t >= 5,
            False,
            {},
        )


@pytest.mark.parametrize("mode", ["dream", "env"])
def test_world_models_vae_then_mdrnn_then_controller(
    tmp_path: Path, trainer_overrides, mode
) -> None:
    data = write_episodes(
        tmp_path / "eps", count=4, length=10, size=64, actions=("continuous", 3)
    )
    base = [o.replace("run_name=run", "run_name=") for o in trainer_overrides]

    vae = run(
        overrides=[
            *base[:2],
            *base[3:],
            "trainer.run_name=vae",
            "model.latent_size=8",
            f"data.path={data.as_posix()}",
            "data.batch_size=4",
            "data.val_fraction=0.25",
            "trainer.max_steps=2",
            "trainer.eval_every=2",
        ],
        task="vae",
    )
    assert "val_loss" in vae.metrics and (vae.model_path / "vae.tar").exists()

    mdrnn = run(
        overrides=[
            *base[:2],
            *base[3:],
            "trainer.run_name=mdrnn",
            f"vae={(vae.model_path / 'vae.tar').as_posix()}",
            "model.seq_len=4",
            "model.hidden_size=16",
            "model.gmm_components=2",
            f"data.path={data.as_posix()}",
            "data.batch_size=2",
            "trainer.max_steps=2",
        ],
        task="mdrnn",
    )
    model = load_world_model(mdrnn.world_model_path, device="cpu")
    assert model.spec.family == "world-models" and model.spec.action.n == 3
    state = model.encode(torch.rand(1, 2, 3, 64, 64), torch.zeros(1, 1, 3))
    out = model.rollout(state, torch.zeros(1, 2, 3))
    assert out.frames.shape == (1, 2, 3, 64, 64) and out.rewards.shape == (1, 2)

    cls, tcfg, cfg, resolved = resolve_run_config(
        overrides=[
            *base[:2],
            *base[3:],
            f"trainer.run_name=ctrl_{mode}",
            f"world_model={mdrnn.world_model_path.as_posix()}",
            f"mode={mode}",
            "model.pop_size=4",
            "model.n_samples=1",
            "model.time_limit=5",
            "trainer.max_steps=2",
            "trainer.eval_every=2",
        ],
        task="controller",
    )
    result = Trainer(tcfg, run_config=resolved).fit(cls(cfg, env_factory=_FrameEnv))
    assert result.progress == 2 and "eval_return" in result.metrics
    saved = torch.load(result.model_path / "controller.tar", weights_only=True)
    assert saved["state_dict"]["fc.weight"].shape == (3, 8 + 16)
