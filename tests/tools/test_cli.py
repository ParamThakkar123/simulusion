import re

import pytest
import simulusion

try:
    import simulusion.cli as cli
except ImportError:
    pytest.skip("click not installed", allow_module_level=True)

try:
    from click.testing import CliRunner
except ImportError:
    CliRunner = None  # type: ignore


def _runner() -> "CliRunner":
    if CliRunner is None:
        pytest.importorskip("click")
    return CliRunner()  # type: ignore


def test_version_shows_package_version():
    runner = _runner()
    res = runner.invoke(cli.app, ["version"])
    assert res.exit_code == 0
    assert re.search(r"\d+\.\d+\.\d+", res.output)
    assert simulusion.__version__ in res.output


def test_envs_list_outputs_backends():
    runner = _runner()
    res = runner.invoke(cli.app, ["envs", "list"])
    assert res.exit_code == 0
    # ui.server declares these backends; ensure at least 'gym' appears
    assert "gym:" in res.output


def test_datasets_list_empty_dir(tmp_path):
    runner = _runner()
    # point to an empty temporary folder
    res = runner.invoke(cli.app, ["datasets", "list", str(tmp_path)])
    assert res.exit_code == 0
    assert "No datasets found" in res.output


def test_datasets_list_with_subdir(tmp_path):
    d = tmp_path / "sample"
    d.mkdir()
    runner = _runner()
    res = runner.invoke(cli.app, ["datasets", "list", str(tmp_path)])
    assert res.exit_code == 0
    assert "- sample" in res.output


def test_train_unknown_model_errors():
    runner = _runner()
    res = runner.invoke(cli.app, ["train", "notamodel"])
    assert res.exit_code == 1
    assert "Unknown model" in res.output


def test_benchmark_requires_agent_or_all_agents():
    runner = _runner()
    res = runner.invoke(cli.app, ["benchmark", "--game", "ALE/Pong-v5"])
    assert res.exit_code == 1
    assert "Either --agent or --all-agents" in res.output


def test_benchmark_single_agent_requires_checkpoint():
    runner = _runner()
    res = runner.invoke(
        cli.app, ["benchmark", "--agent", "iris", "--game", "ALE/Pong-v5"]
    )
    assert res.exit_code == 1
    assert "--checkpoint is required" in res.output


def test_benchmark_single_agent_runs_with_checkpoint(monkeypatch, tmp_path):
    calls = {}

    class FakeBenchmarkRunner:
        def __init__(self, adapter_cls, out_dir):
            calls["adapter_cls"] = adapter_cls
            calls["out_dir"] = out_dir

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return {"aggregate": {}}

    class FakeMultiAgentBenchmarkRunner:
        pass

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(
        cli,
        "_load_benchmark_runtime",
        lambda: (
            {"iris": object},
            FakeBenchmarkRunner,
            FakeMultiAgentBenchmarkRunner,
            FakeTorch,
        ),
    )

    runner = _runner()
    res = runner.invoke(
        cli.app,
        [
            "benchmark",
            "--agent",
            "iris",
            "--game",
            "ALE/Pong-v5",
            "--env-backend",
            "bsuite",
            "--checkpoint",
            str(tmp_path / "iris.pt"),
            "--seeds",
            "0,2",
            "--episodes",
            "3",
            "--out-dir",
            str(tmp_path / "bench"),
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0
    assert "Benchmark finished" in res.output
    assert calls["out_dir"] == str(tmp_path / "bench")
    assert calls["run_kwargs"]["env_spec"] == {
        "game": "ALE/Pong-v5",
        "env_backend": "bsuite",
    }
    assert calls["run_kwargs"]["seeds"] == [0, 2]
    assert calls["run_kwargs"]["num_episodes"] == 3
    assert calls["run_kwargs"]["checkpoint"] == str(tmp_path / "iris.pt")
    assert calls["run_kwargs"]["extra_kwargs"]["device"] == "cpu"


def test_console_entrypoint_run_is_exported():
    assert callable(cli.run)


def test_train_lists_diamond_entrypoint():
    assert cli.TRAINING_MODULES["diamond"] == "simulusion.training.train_diamond"


def test_dmlab_registered_in_backend_specs():
    from simulusion.api import ENV_BACKEND_SPECS, EnvBackendSpec
    from simulusion.catalog import ENV_BACKENDS

    dmlab_spec = ENV_BACKEND_SPECS["dmlab"]

    assert isinstance(dmlab_spec, EnvBackendSpec)
    assert dmlab_spec.name == "dmlab"
    assert "deepmind_lab" in dmlab_spec.aliases
    assert "dmlab" in ENV_BACKENDS


def test_dmlab_backend_specs_are_public_api():
    import simulusion

    assert simulusion.EnvBackendSpec is not None
    assert "dmlab" in simulusion.ENV_BACKEND_SPECS


def test_play_supports_dreamer_as_well_as_diamond():
    assert set(cli.PLAY_MODULES) == {"diamond", "dreamer"}
    assert cli.PLAY_MODULES["dreamer"] == "simulusion.inference.play_dreamer"
    assert set(cli.PLAY_DEFAULT_GAMES) == set(cli.PLAY_MODULES)


def test_play_defaults_game_per_model(monkeypatch):
    # ``--game`` means an Atari ROM for DIAMOND and a control task for Dreamer,
    # so omitting it must not hand Breakout-v5 to Dreamer.
    calls = {}

    class FakeModule:
        @staticmethod
        def run_play(**kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(cli.importlib, "import_module", lambda path: FakeModule)

    runner = _runner()
    res = runner.invoke(cli.app, ["play", "-m", "dreamer", "-c", "ckpt.pt"])

    assert res.exit_code == 0, res.output
    assert calls["game"] == "walker-walk"

    calls.clear()
    res = runner.invoke(cli.app, ["play", "-m", "diamond", "-c", "ckpt.pt"])

    assert res.exit_code == 0, res.output
    assert calls["game"] == "Breakout-v5"
