"""`simulusion train` on the shared Trainer and `simulusion datasets import/info`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from simulusion import cli


def test_train_runs_a_registered_task_with_overrides(tmp_path: Path) -> None:
    from tests.training.tasks.conftest import write_episodes
    from tests.training.tasks.test_genie_task import TINY

    data = write_episodes(tmp_path / "clips", count=3, length=6)
    result = CliRunner().invoke(
        cli.app,
        [
            "train",
            "genie",
            f"data.path={data.as_posix()}",
            "trainer.device=cpu",
            "trainer.max_steps=1",
            "trainer.console=false",
            f"trainer.out_dir={tmp_path.as_posix()}/runs",
            "trainer.run_name=cli",
            *TINY,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "World model:" in result.output
    assert (tmp_path / "runs" / "cli" / "world_model" / "world_model.json").exists()


def test_train_print_config_resolves_a_shipped_config() -> None:
    result = CliRunner().invoke(
        cli.app, ["train", "--config", "dreamer", "--print-config"]
    )
    assert result.exit_code == 0, result.output
    assert "task: dreamer" in result.output and "max_steps: 1000000" in result.output


def test_train_reports_config_mistakes_without_a_traceback() -> None:
    result = CliRunner().invoke(cli.app, ["train", "genie", "model.no_such_field=1"])
    assert result.exit_code == 1
    assert "model.no_such_field" in result.output


def test_train_legacy_flag_reaches_old_entrypoints(monkeypatch) -> None:
    import sys

    seen = []
    module = type(sys)("fake_legacy")
    module.main = lambda argv: seen.append(argv)
    monkeypatch.setitem(sys.modules, "fake_legacy", module)
    monkeypatch.setitem(cli.TRAINING_MODULES, "genie", "fake_legacy")
    result = CliRunner().invoke(
        cli.app, ["train", "genie", "--legacy", "--inproc", "x=1"]
    )
    assert result.exit_code == 0, result.output
    assert seen == [["x=1"]]


def test_datasets_import_lerobot_and_info(tmp_path: Path) -> None:
    pytest.importorskip("av")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from tests.datasets.test_episodes import _make_v30

    _make_v30(tmp_path / "ds", [5, 6, 4], (pa, pq))
    out = tmp_path / "eps"
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "import",
            "lerobot",
            str(tmp_path / "ds"),
            str(out),
            "--episodes",
            "0-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 episodes, 11 frames" in result.output
    assert "continuous (2)" in result.output

    info = CliRunner().invoke(cli.app, ["datasets", "info", str(out)])
    assert info.exit_code == 0 and "observations: state" in info.output


def test_parse_index_list() -> None:
    assert cli._parse_index_list("0-2,5") == [0, 1, 2, 5]


def test_old_style_commands_still_reach_the_legacy_entrypoint(
    monkeypatch, tmp_path
) -> None:
    import sys

    seen = []
    module = type(sys)("fake_iris_legacy")
    module.main = lambda argv: seen.append(argv)
    monkeypatch.setitem(sys.modules, "fake_iris_legacy", module)
    monkeypatch.setitem(cli.TRAINING_MODULES, "iris", "fake_iris_legacy")
    # A flat override, as `simulusion train iris total_epochs=100` always took.
    result = CliRunner().invoke(
        cli.app, ["train", "iris", "--inproc", "total_epochs=100"]
    )
    assert result.exit_code == 0, result.output
    assert "legacy 'iris' entrypoint" in result.output
    # A config file in the family's own flat format (no `task:`).
    flat = tmp_path / "iris.yaml"
    flat.write_text("total_epochs: 3\n")
    result = CliRunner().invoke(
        cli.app, ["train", "iris", "--inproc", "--config", str(flat)]
    )
    assert result.exit_code == 0, result.output
    assert seen == [["total_epochs=100"], ["--config", str(flat)]]


def test_run_config_style_commands_use_the_trainer(monkeypatch) -> None:
    from simulusion.training import run as run_module

    calls = []
    monkeypatch.setattr(run_module, "main", lambda argv: calls.append(argv))
    result = CliRunner().invoke(
        cli.app, ["train", "iris", "model.total_epochs=3", "game=ALE/Pong-v5"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [["iris", "model.total_epochs=3", "game=ALE/Pong-v5"]]


def test_train_help_tasks_lists_every_task() -> None:
    result = CliRunner().invoke(cli.app, ["train", "--help-tasks"])
    assert result.exit_code == 0
    for name in ("genie", "dreamer-wm", "iris", "diamond-wm", "controller", "jepa"):
        assert name in result.output


def test_datasets_import_rejects_a_missing_folder(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "import",
            "lerobot",
            str(tmp_path / "nope"),
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
