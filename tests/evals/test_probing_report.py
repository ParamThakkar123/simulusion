"""Tests for the one-call probing report and the `simulusion probe` command."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from click.testing import CliRunner  # noqa: E402

from simulusion.data.episode import Episode  # noqa: E402
from simulusion.envs.dmc import PhysicsStateSpec  # noqa: E402
from simulusion.evals.probing import (  # noqa: E402
    ProbeDataset,
    ProbeReport,
    probe_world_model,
)
from tests.evals.test_probing_latents import _action_fn  # noqa: E402
from tests.test_world_model_protocol import _build  # noqa: E402

_SPEC = PhysicsStateSpec(
    qpos_names=("slider", "hinge_1"),
    qpos_kinds=("slide", "hinge"),
    qvel_names=("slider", "hinge_1"),
    qvel_kinds=("slide", "hinge"),
    act_names=(),
)
_FAST = {"mlp": False, "horizons": (1, 2), "starts_every": 3}


def _data(episodes=5, T=12, size=32, seed=0):
    rng = np.random.default_rng(seed)
    return ProbeDataset(
        task="cartpole-swingup",
        spec=_SPEC,
        episodes=[
            Episode(
                frames=rng.integers(0, 256, (T, size, size, 3), dtype=np.uint8),
                actions=rng.uniform(-1, 1, (T - 1, 2)).astype(np.float32),
                observations={
                    "qpos": rng.normal(size=(T, 2)).astype(np.float32),
                    "qvel": rng.normal(size=(T, 2)).astype(np.float32),
                    "act": np.zeros((T, 0), dtype=np.float32),
                },
            )
            for _ in range(episodes)
        ],
    )


def test_report_scores_the_model_its_baselines_and_imagination():
    report = probe_world_model(_build("planet"), _data(), **_FAST)

    sources = {row.source for row in report.rows}
    assert sources == {"planet", "planet (1 frame)", "planet (untrained)", "pixels"}
    assert {row.feature for row in report.rows if row.source == "planet"} == {
        "deter",
        "stoch",
        "features",
    }
    assert report.episodes == {"train": 3, "val": 1, "test": 1}
    assert report.target_names == [
        "qpos/slider",
        "qpos/hinge_1/sin",
        "qpos/hinge_1/cos",
        "qvel/slider",
        "qvel/hinge_1",
    ]
    row = report.row("planet", "deter")
    assert row.dims == 16 and row.mlp_r2 is None
    assert set(row.targets) == set(report.target_names)
    assert [r.feature for r in report.rollouts] == ["features"]
    assert report.rollouts[0].horizons == [1, 2]
    assert report.rollouts[0].rollouts > 0
    assert report.notes == []


def test_report_round_trips_through_json_and_renders_markdown(tmp_path):
    report = probe_world_model(
        _build("planet"), _data(), baselines=False, single_frame=False, **_FAST
    )

    loaded = ProbeReport.load(report.save(tmp_path / "report.json"))
    assert loaded == report
    text = report.to_markdown()
    assert text.startswith("# Probing planet on cartpole-swingup")
    assert "| planet | deter | 16 |" in text
    assert "## Imagined states" in text and "## Linear R² per target" in text
    with pytest.raises(ValueError, match="Not a probe report"):
        ProbeReport.from_dict({"format": "other"})
    with pytest.raises(KeyError):
        report.row("pixels", "pixels")


def test_report_notes_what_a_family_cannot_do():
    iris = probe_world_model(_build("iris"), _data(), baselines=False, **_FAST)
    assert iris.rollouts == []
    assert "discrete actions" in iris.notes[0]

    diamond = probe_world_model(
        _build("diamond"),
        _data(size=16),
        baselines=False,
        action_fn=_action_fn(_build("diamond")),
        **_FAST,
    )
    assert any("one-frame variant was skipped" in note for note in diamond.notes)
    assert {row.source for row in diamond.rows} == {"diamond"}

    short = probe_world_model(
        _build("planet"),
        _data(T=4),
        baselines=False,
        single_frame=False,
        mlp=False,
        horizons=(5,),
    )
    assert short.rollouts == [] and "not probed" in short.notes[0]


def test_report_notes_targets_too_flat_to_score(tmp_path):
    data = _data()
    for episode in data:
        episode.observations["qvel"][:, 1] = 0.0  # the pole never moves
    report = probe_world_model(
        _build("planet"), data, baselines=False, single_frame=False, **_FAST
    )

    assert "qvel/hinge_1" in report.notes[0]
    assert report.row("planet", "deter").targets["qvel/hinge_1"]["linear"] is None
    # NaN never reaches the JSON.
    assert "NaN" not in report.save(tmp_path / "r.json").read_text(encoding="utf-8")


def test_report_needs_three_episodes_and_valid_splits():
    with pytest.raises(ValueError, match="three episodes"):
        probe_world_model(_build("planet"), _data(episodes=2))
    with pytest.raises(ValueError, match="splits"):
        probe_world_model(_build("planet"), _data(), splits=(0.5, 0.5, 0.5))


def test_probe_command_writes_json_and_markdown(tmp_path):
    from simulusion.cli import app

    model_dir = _build("planet").save(tmp_path / "planet")
    data_dir = _data(size=64).save(tmp_path / "data")
    out = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "probe",
            "--checkpoint",
            str(model_dir),
            "--data",
            str(data_dir),
            "--no-mlp",
            "--no-baselines",
            "--horizons",
            "1,2",
            "--keys",
            "deter",
            "--device",
            "cpu",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["family"] == "planet"
    assert {row["feature"] for row in report["rows"]} == {"deter"}
    assert out.with_suffix(".md").exists()
    assert "Probing planet on cartpole-swingup" in result.output


def test_probe_command_needs_a_task_or_data(tmp_path):
    from simulusion.cli import app

    model_dir = _build("planet").save(tmp_path / "planet")
    result = CliRunner().invoke(app, ["probe", "--checkpoint", str(model_dir)])
    assert result.exit_code != 0
    assert "--task" in result.output
