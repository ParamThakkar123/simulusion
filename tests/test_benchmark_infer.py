"""CLI surface for scripts/benchmark_infer.py — no checkpoints or GPUs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_infer.py"


@pytest.fixture(scope="module")
def infer():
    spec = importlib.util.spec_from_file_location("benchmark_infer", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_and_play_models_are_documented(infer):
    assert "diamond" in infer.RECORD_MODELS
    assert "dreamer" in infer.PLAY_MODELS
    assert set(infer.PLAY_MODELS) <= set(infer.RECORD_MODELS)


def test_resolve_model_aliases(infer):
    args = infer.build_parser().parse_args(["--model", "dreamer-v2"])
    assert infer.resolve_model(args) == "dreamer"
    args = infer.build_parser().parse_args(["--models", "ijepa"])
    assert infer.resolve_model(args) == "jepa"


def test_versus_sets_up_from_flag(infer):
    args = infer.build_parser().parse_args(
        ["--mode", "play", "--versus", "--model", "iris"]
    )
    assert args.versus
    assert args.mode == "play"
    assert infer.resolve_model(args) == "iris"
