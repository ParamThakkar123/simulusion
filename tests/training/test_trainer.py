"""The shared Trainer: budget, logging, evaluation, checkpoints, resume, configs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn

from simulusion.configs.structured import ConfigError
from simulusion.training.precision import GradientUpdate, MixedPrecision
from simulusion.training.run import resolve_run_config
from simulusion.training.task import TrainingTask, get_task, list_tasks, register_task
from simulusion.training.trainer import Trainer, TrainerConfig, latest_checkpoint


@dataclass
class _ModelConfig:
    width: int = 4
    lr: float = 0.1


@dataclass
class _ToyConfig:
    model: _ModelConfig = field(default_factory=_ModelConfig)
    steps_per_call: int = 1
    eval_values: Optional[list] = None


class _ToyTask(TrainingTask):
    """Fit y = 2x with a linear layer; progress can jump by several units."""

    config_cls = _ToyConfig
    distributed = True
    default_max_steps = 6

    def setup(self, trainer: Trainer) -> None:
        self.model = trainer.prepare_module(nn.Linear(self.config.model.width, 1))
        opt = torch.optim.SGD(self.model.parameters(), lr=self.config.model.lr)
        self.update = trainer.gradient_update(
            opt, self.model.parameters(), max_grad_norm=10.0
        )
        self.units = 0
        self.evals = 0

    def train_step(self) -> dict:
        x = torch.randn(8, self.config.model.width, device=self.trainer.device)
        with self.trainer.precision.autocast():
            loss = ((self.model(x) - 2 * x.sum(-1, keepdim=True)) ** 2).mean()
        self.update(loss)
        self.units += self.config.steps_per_call
        return {"loss": loss.item()}

    def progress(self) -> int:
        return self.units

    def evaluate(self) -> dict:
        values = self.config.eval_values
        self.evals += 1
        if values:
            return {"score": float(values[min(self.evals - 1, len(values) - 1)])}
        return {"score": float(self.units)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "update": self.update.state_dict(),
            "units": self.units,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.update.load_state_dict(state["update"])
        self.units = int(state["units"])


register_task("toy-test")(_ToyTask)


def _config(tmp_path: Path, **kw: Any) -> TrainerConfig:
    base = dict(
        device="cpu", out_dir=str(tmp_path), run_name="run", log_every=1, console=False
    )
    base.update(kw)
    return TrainerConfig(**base)


def _metrics(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]


def test_runs_to_budget_and_writes_run_files(tmp_path: Path) -> None:
    result = Trainer(_config(tmp_path, max_steps=5)).fit(_ToyTask())
    assert result.progress == 5 and result.iterations == 5
    run = tmp_path / "run"
    assert (run / "config.yaml").exists()
    info = json.loads((run / "run.json").read_text())
    assert info["task"] == "toy-test" and info["precision"] == "fp32"
    assert "torch" in info["packages"]
    assert result.checkpoint == run / "checkpoints" / "step_00000005"
    assert latest_checkpoint(run) == result.checkpoint
    logged = [row for row in _metrics(run) if "train/loss" in row]
    assert [row["step"] for row in logged] == [1, 2, 3, 4, 5]


def test_progress_jumps_trigger_logging_eval_and_checkpoints(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path, max_steps=20, log_every=10, eval_every=10, checkpoint_every=10
    )
    task = _ToyTask(_ToyConfig(steps_per_call=4))
    result = Trainer(cfg).fit(task)
    assert result.progress == 20  # 5 calls of 4 units
    steps = sorted({row["step"] for row in _metrics(tmp_path / "run")})
    assert steps == [12, 20]  # crossed 10 at 12, 20 at 20
    # Two in-loop evaluations plus the final one.
    assert task.evals == 3
    names = sorted(p.name for p in (tmp_path / "run" / "checkpoints").iterdir())
    assert names == ["step_00000012", "step_00000020"]


def test_resume_continues_from_checkpoint(tmp_path: Path) -> None:
    Trainer(_config(tmp_path, max_steps=3)).fit(_ToyTask())
    task = _ToyTask()
    result = Trainer(_config(tmp_path, max_steps=7, resume="auto")).fit(task)
    assert result.iterations == 7  # 3 restored + 4 new
    assert result.progress == 7
    steps = [row["step"] for row in _metrics(tmp_path / "run") if "train/loss" in row]
    assert steps == [1, 2, 3, 4, 5, 6, 7]


def test_resume_is_bit_exact(tmp_path: Path) -> None:
    straight = Trainer(_config(tmp_path / "a", max_steps=6, seed=3)).fit(_ToyTask())
    Trainer(_config(tmp_path / "b", max_steps=3, seed=3)).fit(_ToyTask())
    resumed_task = _ToyTask()
    Trainer(_config(tmp_path / "b", max_steps=6, seed=3, resume="auto")).fit(
        resumed_task
    )

    def weights(path: Path) -> dict:
        return torch.load(path / "task" / "task.pt", weights_only=True)["model"]

    a = weights(straight.checkpoint)
    b = weights(latest_checkpoint(tmp_path / "b" / "run"))
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_early_stopping_on_eval_metric(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        max_steps=100,
        eval_every=1,
        early_stopping_metric="score",
        early_stopping_mode="max",
        patience=2,
    )
    task = _ToyTask(_ToyConfig(eval_values=[1, 2, 2, 2, 2, 2]))
    result = Trainer(cfg).fit(task)
    assert result.stopped_early
    # Improves at 1 and 2; EarlyStopping stops once bad evaluations exceed
    # `patience`, i.e. on the third flat one.
    assert result.progress == 5


def test_unknown_early_stopping_metric_errors(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_steps=3, eval_every=1, early_stopping_metric="nope")
    with pytest.raises(KeyError, match="nope"):
        Trainer(cfg).fit(_ToyTask())


def test_old_checkpoints_are_pruned(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_steps=6, checkpoint_every=1, keep_checkpoints=2)
    Trainer(cfg).fit(_ToyTask())
    names = sorted(p.name for p in (tmp_path / "run" / "checkpoints").iterdir())
    assert names == ["step_00000005", "step_00000006"]


def test_single_process_task_refuses_data_parallel(tmp_path: Path, monkeypatch) -> None:
    class Local(_ToyTask):
        distributed = False

    trainer = Trainer(_config(tmp_path, max_steps=1))
    monkeypatch.setattr(Trainer, "num_processes", property(lambda self: 2))
    with pytest.raises(RuntimeError, match="single process"):
        trainer.fit(Local())


def test_resolve_run_config_layers_defaults_file_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("task: toy-test\ntrainer:\n  max_steps: 9\nmodel:\n  width: 3\n")
    cls, trainer, task_cfg, resolved = resolve_run_config(
        path, ["model.lr=0.5", "trainer.precision=bf16"]
    )
    assert cls is _ToyTask
    assert trainer.max_steps == 9 and trainer.precision == "bf16"
    assert task_cfg.model == _ModelConfig(width=3, lr=0.5)
    assert resolved["task"] == "toy-test"
    assert resolved["model"] == {"width": 3, "lr": 0.5}
    assert resolved["trainer"]["max_steps"] == 9


def test_resolved_config_reproduces_the_run(tmp_path: Path) -> None:
    from simulusion.training.run import run

    first = run({"task": "toy-test", "trainer": vars(_config(tmp_path, max_steps=2))})
    _, trainer, task_cfg, _ = resolve_run_config(first.run_dir / "config.yaml")
    assert trainer.max_steps == 2
    assert task_cfg == _ToyConfig()


def test_unknown_config_keys_name_their_path() -> None:
    with pytest.raises(ConfigError, match="model.depth"):
        resolve_run_config({"task": "toy-test", "model": {"depth": 2}})
    with pytest.raises(ConfigError, match="trainer.max_stepz"):
        resolve_run_config({"task": "toy-test", "trainer": {"max_stepz": 2}})


def test_unknown_task_lists_known_ones() -> None:
    with pytest.raises(KeyError, match="genie"):
        get_task("no-such-task")
    assert "genie" in list_tasks()


def test_gradient_update_accumulates() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    update = GradientUpdate(
        opt, model.parameters(), MixedPrecision("fp32", "cpu"), accumulation_steps=2
    )
    before = model.weight.detach().clone()
    x = torch.ones(1, 2)
    assert update(model(x).sum()) is None
    assert torch.equal(model.weight, before)  # no step on the first call
    norm = update(model(x).sum())
    assert norm is not None
    # Mean of two identical gradients is that gradient: d(sum)/dw = x.
    assert torch.allclose(model.weight, before - x)


def test_mixed_precision_from_use_amp() -> None:
    assert MixedPrecision.from_use_amp(False, "cpu").name == "fp32"
    cpu = MixedPrecision.from_use_amp(True, "cpu")
    assert cpu.name == "bf16" and not cpu.scales_loss
    with cpu.autocast():
        out = nn.Linear(4, 4)(torch.randn(2, 4))
    assert out.dtype == torch.bfloat16
    with pytest.raises(ValueError):
        MixedPrecision("fp16", "cpu")
