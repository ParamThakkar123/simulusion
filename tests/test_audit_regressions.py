"""Regression tests for smaller issues found in the September 2026 code audit."""

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


# -- Dreamer config / replay -------------------------------------------------


def test_dreamer_config_with_env_instance_serialises():
    from simulusion.configs.dreamer_config import DreamerConfig

    config = DreamerConfig(env_instance=object())
    text = config.to_yaml()
    assert "env_instance" not in text
    assert "env_instance" not in config.to_dict()
    assert DreamerConfig.from_yaml(text).env_instance is None


def test_replay_buffer_separates_truncation_from_termination():
    from simulusion.memory.dreamer_memory import ReplayBuffer

    buffer = ReplayBuffer(8, (1, 2, 2), 1, seq_len=2, batch_size=1)
    obs = {"image": np.zeros((1, 2, 2), dtype=np.uint8)}
    buffer.add(obs, np.zeros(1), 0.0, done=True, terminated=False)  # time limit
    buffer.add(obs, np.zeros(1), 0.0, done=True, terminated=True)
    buffer.add(obs, np.zeros(1), 0.0, done=False)
    assert buffer.terminals[:3].tolist() == [1.0, 1.0, 0.0]
    assert buffer.terminated[:3].tolist() == [0.0, 1.0, 0.0]
    assert len(buffer.sample()) == 4
    assert len(buffer.sample(include_terminated=True)) == 5


@pytest.mark.parametrize(
    "done, info, expected",
    [
        (False, {}, False),
        (True, {}, True),
        (True, {"terminated": False, "truncated": True}, False),
        (True, {"terminated": True, "truncated": False}, True),
        (True, {"TimeLimit.truncated": True}, False),
        (True, None, True),
    ],
)
def test_true_termination(done, info, expected):
    from simulusion.models.dreamer import _true_termination

    assert _true_termination(done, info) is expected


def test_time_limit_wrapper_truncation_is_not_stored_as_terminal():
    from simulusion.envs.wrappers import TimeLimit
    from simulusion.models.dreamer import _true_termination

    class Endless:
        def reset(self):
            return {"image": np.zeros((3, 4, 4), dtype=np.uint8)}

        def step(self, action):
            return self.reset(), 0.0, False, {}

    env = TimeLimit(Endless(), duration=2)
    env.reset()
    env.step(None)
    _, _, done, info = env.step(None)
    assert done
    assert _true_termination(done, info) is False


# -- CLI -----------------------------------------------------------------------


def test_inproc_training_errors_propagate_without_rerun(monkeypatch):
    from click.testing import CliRunner

    from simulusion import cli

    calls = []

    def failing_main(argv):
        calls.append(argv)
        raise TypeError("bug inside training")

    module = type(sys)("fake_train_module")
    module.main = failing_main
    monkeypatch.setitem(sys.modules, "fake_train_module", module)
    monkeypatch.setitem(cli.TRAINING_MODULES, "fake", "fake_train_module")
    ran_subprocess = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: ran_subprocess.append(a))

    result = CliRunner().invoke(cli.app, ["train", "fake", "--inproc", "lr=1"])

    assert result.exit_code != 0
    assert calls == [["lr=1"]]
    assert ran_subprocess == []


def test_inproc_argv_less_main_reads_sys_argv():
    from simulusion import cli

    seen = []
    cli._call_training_main(lambda: seen.append(list(sys.argv)), "mod", ["--env", "x"])
    assert seen == [["mod", "--env", "x"]]


def test_eval_and_play_entry_points_live_in_the_package():
    from simulusion import cli

    for module in (*cli.EVAL_MODULES.values(), *cli.PLAY_MODULES.values()):
        assert module.startswith("simulusion."), module


def test_benchmark_cli_does_not_import_hydra_at_module_level():
    tree = ast.parse(
        (REPO / "simulusion/benchmarks/cli.py").read_text(encoding="utf-8")
    )
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "hydra" not in top_level
    assert "omegaconf" not in top_level


# -- Environment backends ------------------------------------------------------


def test_registered_env_backend_is_used_by_make_env(monkeypatch):
    import simulusion
    from simulusion.registry import deregister_env_backend, register_env_backend

    module = type(sys)("fake_env_backend")
    module.make = lambda env_id, **kwargs: ("made", env_id, kwargs)
    monkeypatch.setitem(sys.modules, "fake_env_backend", module)

    register_env_backend("fake-backend", factory_path="fake_env_backend:make")
    try:
        assert "fake-backend" in simulusion.list_env_backends()
        assert simulusion.get_env_backend_spec("fake_backend").name == "fake-backend"
        assert simulusion.make_env("x", backend="fake-backend", k=1) == (
            "made",
            "x",
            {"k": 1},
        )
    finally:
        deregister_env_backend("fake-backend")


def test_dmc_backend_is_available():
    import simulusion

    assert "dmc" in simulusion.list_env_backends()
    assert simulusion.get_env_backend_spec("dm_control").factory_path.endswith(
        "make_dmc_env"
    )


def test_dmc_rejects_underscore_task_names_clearly():
    from simulusion.envs.dmc import DeepMindControlEnv

    with pytest.raises(ValueError, match="cartpole-balance"):
        DeepMindControlEnv("cartpole_balance", seed=0)


# -- Devices -------------------------------------------------------------------


def test_resolve_device_falls_back_when_unavailable(monkeypatch):
    from simulusion.utils import device as device_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "mps_is_available", lambda: True)
    assert device_utils.get_default_device().type == "mps"
    assert device_utils.resolve_device("cuda").type == "mps"
    assert device_utils.get_default_device(allow_gpu=False).type == "cpu"

    monkeypatch.setattr(device_utils, "mps_is_available", lambda: False)
    assert device_utils.resolve_device("mps").type == "cpu"
    assert device_utils.resolve_device(None).type == "cpu"


def test_cuda_fallback_is_logged_not_silent(monkeypatch, caplog):
    from simulusion.utils import device as device_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "mps_is_available", lambda: False)

    with caplog.at_level("WARNING", logger=device_utils.__name__):
        assert device_utils.resolve_device("cuda").type == "cpu"
    # A run that asked for cuda and trained on CPU must say so.
    assert any("CUDA is not available" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING", logger=device_utils.__name__):
        assert device_utils.resolve_device("mps").type == "cpu"
    assert any("MPS is not available" in record.message for record in caplog.records)

    # An honoured request logs nothing.
    caplog.clear()
    with caplog.at_level("WARNING", logger=device_utils.__name__):
        assert device_utils.resolve_device("cpu").type == "cpu"
    assert caplog.records == []


def test_mujoco_gl_egl_default_is_linux_only():
    source = (REPO / "simulusion/models/dreamer.py").read_text(encoding="utf-8")
    assert 'sys.platform.startswith("linux") and os.environ.get("MUJOCO_GL")' in source


# -- Security ------------------------------------------------------------------


def test_no_unsafe_torch_load():
    offenders = []
    for folder in ("simulusion", "scripts", "demos", "tools", "examples"):
        for path in (REPO / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if re.search(r"weights_only\s*=\s*False", text):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


# -- Genie / packaging ---------------------------------------------------------


def test_no_top_level_evals_package_outside_simulusion():
    assert not (REPO / "evals").exists()
    from simulusion.evals import PSNR  # noqa: F401


# -- Video logging -------------------------------------------------------------


def test_video_frames_are_not_wrapped_and_gif_needs_no_moviepy(tmp_path, monkeypatch):
    from PIL import Image

    from simulusion.utils.dreamer_utils import _video_to_uint8, _write_gif

    uint8_video = np.full((3, 4, 4, 3), 200, dtype=np.uint8)
    assert _video_to_uint8(uint8_video).max() == 200
    assert _video_to_uint8(uint8_video.astype(np.float32) / 255).max() == 200

    monkeypatch.setitem(sys.modules, "moviepy", None)
    # Distinct frames: Pillow merges identical consecutive GIF frames.
    moving = np.stack([np.full((4, 4, 3), v, dtype=np.uint8) for v in (0, 120, 240)])
    path = tmp_path / "clip.gif"
    _write_gif(moving, str(path), fps=10)
    with Image.open(path) as gif:
        assert gif.n_frames == 3


# -- DeepMind Control installer ------------------------------------------------


def test_installer_matches_dm_control_to_the_installed_mujoco():
    from simulusion.install_dmc import (
        NEWEST_DM_CONTROL,
        dm_control_requirement,
        install_steps,
    )

    # mujoco 3.13 removed an enum mujoco-mjx still uses, so the installed mujoco
    # is kept and dm-control is matched to it rather than the other way round.
    assert dm_control_requirement((3, 5, 0)) == "dm-control==1.0.37"
    assert dm_control_requirement((3, 11, 2)) == "dm-control==1.0.44"
    assert dm_control_requirement((3, 13, 0)) == "dm-control==1.0.46"
    assert dm_control_requirement(None) == NEWEST_DM_CONTROL
    assert dm_control_requirement((3, 5, 0), upgrade_mujoco=True) == NEWEST_DM_CONTROL

    # mujoco is only installed when it is missing, or on --upgrade-mujoco.
    steps = install_steps()
    if __import__("simulusion.install_dmc", fromlist=["x"]).installed_mujoco_version():
        assert not any(arg.startswith("mujoco") for step in steps for arg in step)


def test_installer_rejects_a_mujoco_older_than_any_known_dm_control():
    from simulusion.install_dmc import dm_control_requirement

    with pytest.raises(SystemExit, match="older than"):
        dm_control_requirement((3, 1, 0))


def test_uv_constraints_keep_brax_and_dmc_compatible():
    import tomllib

    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    constraints = config["tool"]["uv"]["constraint-dependencies"]

    assert "mujoco<3.13" in constraints


# -- System metrics ------------------------------------------------------------


def test_missing_nvml_does_not_kill_training(monkeypatch, caplog):
    """A logging metric must never raise: torch.cuda.utilization needs NVML."""
    from simulusion.utils import logging_utils

    def no_nvml(index):
        raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 2048)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 4096)
    monkeypatch.setattr(torch.cuda, "utilization", no_nvml)
    monkeypatch.setattr(logging_utils, "_NVML_WARNED", False)

    with caplog.at_level("WARNING", logger=logging_utils.__name__):
        stats = logging_utils.collect_system_stats("cuda")

    assert stats["system/gpu_memory_allocated_mb"] == 1024 / (1024**2)
    assert "system/gpu_utilization_percent" not in stats
    assert any("GPU utilization is unavailable" in r.message for r in caplog.records)


def test_gpu_utilization_is_reported_when_nvml_works(monkeypatch):
    from simulusion.utils import logging_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 0)
    monkeypatch.setattr(torch.cuda, "utilization", lambda index: 42)

    stats = logging_utils.collect_system_stats("cuda")

    assert stats["system/gpu_utilization_percent"] == 42.0
