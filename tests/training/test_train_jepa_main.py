"""``train_jepa.main``'s own loop, which W&B sweeps still run, and the sweep wiring."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from simulusion.configs.jepa_config import JEPAConfig
from simulusion.training import train_jepa

pytestmark = pytest.mark.timeout(180)


def _tiny(tmp_path: Path) -> JEPAConfig:
    from tests.training.tasks.test_offline_tasks import _image_folder

    root = _image_folder(tmp_path / "imgs", n=8)
    config = JEPAConfig()
    for key, value in dict(
        dataset="imagefolder",
        root_path=str(root),
        image_folder="train",
        model_name="vit_tiny",
        crop_size=64,
        patch_size=8,
        pred_depth=1,
        pred_emb_dim=48,
        batch_size=4,
        num_workers=0,
        pin_mem=False,
        min_keep=2,
        epochs=1,
        warmup=1,
        write_tag="run",
        folder=str(tmp_path / "jepa"),
    ).items():
        setattr(config, key, value)
    return config


def test_main_trains_and_writes_its_checkpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _tiny(tmp_path)
    train_jepa.main(config)
    folder = tmp_path / "jepa"
    assert (folder / "params-ijepa.yaml").exists()
    state = torch.load(folder / "run-latest.pth.tar", weights_only=True)
    assert state["epoch"] == 1
    assert {"encoder", "predictor", "target_encoder", "opt"} <= set(state)


def test_print_config_from_the_cli(capsys) -> None:
    cfg = train_jepa.main_from_cli(["--print-config", "data.batch_size=8"])
    assert cfg["data"]["batch_size"] == 8
    assert "batch_size: 8" in capsys.readouterr().out


class _FakeWandb:
    """Records sweeps and runs the agent's function once, as one trial."""

    def __init__(self, trial: dict[str, Any]) -> None:
        self.trial = trial
        self.sweeps: list[tuple] = []
        self.config: dict[str, Any] = {}

    def sweep(self, config: Any, project: Any = None, entity: Any = None) -> str:
        self.sweeps.append((config, project, entity))
        return "sweep-1"

    def agent(self, sweep_id: str, function: Any) -> None:
        assert sweep_id == "sweep-1"
        function()

    @contextlib.contextmanager
    def init(self) -> Any:
        self.config = dict(self.trial)
        yield SimpleNamespace(config=self.config)


def test_a_sweep_trial_trains_with_the_sampled_settings(monkeypatch) -> None:
    fake = _FakeWandb({"lr": 0.123, "batch_size": 16, "not_a_field": 1})
    monkeypatch.setattr(train_jepa, "_require_wandb", lambda: fake)
    runs: list[dict] = []
    real_main = train_jepa.main

    def main(args: Any = None, resume_preempt: bool = False) -> Any:
        if isinstance(args, dict) and not args["logging"]["enable_sweep"]:
            runs.append(args)  # a trial: record it instead of training
            return None
        return real_main(args, resume_preempt)

    monkeypatch.setattr(train_jepa, "main", main)
    config = JEPAConfig()
    config.enable_sweep = True
    config.sweep_config = {"method": "random"}
    train_jepa.main(config)

    assert fake.sweeps == [({"method": "random"}, config.wandb_project, None)]
    (trial,) = runs
    assert trial["optimization"]["lr"] == 0.123
    assert trial["data"]["batch_size"] == 16
