"""The IRIS Atari 100k benchmark runner: scores, aggregation, failures, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulusion.training import run_atari_100k as bench

pytestmark = pytest.mark.timeout(180)


def test_every_benchmark_game_has_reference_scores() -> None:
    assert len(bench.ATARI_100K_GAMES) == 26
    for game in bench.ATARI_100K_GAMES:
        name = bench.game_name(game)
        assert name in bench.HUMAN_SCORES and name in bench.RANDOM_SCORES, name


def test_human_normalized_score() -> None:
    human, random = bench.HUMAN_SCORES["Pong"], bench.RANDOM_SCORES["Pong"]
    assert bench.compute_human_normalized_score(random, "Pong") == 0.0
    assert bench.compute_human_normalized_score(human, "Pong") == pytest.approx(1.0)
    with pytest.raises(KeyError, match="NotAGame"):
        bench.compute_human_normalized_score(1.0, "NotAGame")


def test_interquartile_mean_trims_a_quarter_from_each_end() -> None:
    values = [100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -100.0]
    assert bench.interquartile_mean(values) == pytest.approx(np.mean([1, 2, 3, 4]))
    # Not the mean of the two quartiles, which the runner used to report.
    skewed = [0.0, 0.0, 1.0, 1.0, 1.0, 8.0, 9.0, 9.0]
    assert bench.interquartile_mean(skewed) == pytest.approx(2.75)
    assert np.mean(np.percentile(skewed, [25, 75])) == pytest.approx(4.5)
    assert bench.interquartile_mean([]) == 0.0


def test_run_single_game_trains_and_scores(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from simulusion.configs.iris_config import IRISConfig

    from tests.training.tasks.test_iris_tasks import TINY, FakeAtari

    config = IRISConfig()
    for key, value in {**TINY, "collection_epochs": 1, "eval_episodes": 1}.items():
        setattr(config, key, value)
    config.total_epochs = 1
    result = bench.run_single_game(
        "ALE/Pong-v5",
        config,
        "cpu",
        save_dir=str(tmp_path / "Pong"),
        eval_episodes=2,
        env=FakeAtari(),
    )
    assert "error" not in result, result
    assert result["game_name"] == "Pong"
    assert result["human_normalized_score"] == pytest.approx(
        bench.compute_human_normalized_score(result["mean_return"], "Pong")
    )
    assert np.isfinite(result["std_return"])
    assert (tmp_path / "Pong" / "run.json").exists()


def test_run_atari_100k_aggregates_and_records_failures(
    tmp_path: Path, monkeypatch
) -> None:
    scores = {"Pong": 2.0, "Breakout": 0.5, "Boxing": 1.5}
    seen_dirs: list[str] = []

    def fake_run(game, config, device, *, save_dir, eval_episodes):
        seen_dirs.append(save_dir)
        name = bench.game_name(game)
        if name not in scores:
            return {"game": game, "error": "RuntimeError: boom"}
        return {
            "game": game,
            "game_name": name,
            "mean_return": 0.0,
            "std_return": 0.0,
            "human_normalized_score": scores[name],
        }

    monkeypatch.setattr(bench, "run_single_game", fake_run)
    out = tmp_path / "out" / "results.json"
    games = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/Alien-v5", "ALE/Boxing-v5"]
    results, aggregate = bench.run_atari_100k(
        games, output_file=str(out), save_dir=str(tmp_path / "ckpt")
    )
    assert [r["game_name"] for r in results] == ["Pong", "Breakout", "Boxing"]
    assert aggregate["num_games"] == 3 and aggregate["num_failed"] == 1
    assert aggregate["num_superhuman"] == 2
    assert aggregate["mean_hns"] == pytest.approx(4.0 / 3)
    assert aggregate["median_hns"] == pytest.approx(1.5)
    # Each game trains in its own folder.
    assert len(set(seen_dirs)) == 4
    written = json.loads(out.read_text())
    assert written["errors"] == [
        {"game": "ALE/Alien-v5", "error": "RuntimeError: boom"}
    ]
    assert written["aggregate"] == aggregate


def test_main_reads_overrides(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("omegaconf")
    calls = []
    monkeypatch.setattr(
        bench, "run_atari_100k", lambda **kwargs: calls.append(kwargs) or ([], {})
    )
    bench.main(
        [
            "device=cpu",
            f"output={tmp_path / 'r.json'}",
            "games=[Pong,ALE/Breakout-v5]",
            "eval_episodes=3",
        ]
    )
    (call,) = calls
    assert call["device"] == "cpu"
    assert call["games"] == ["ALE/Pong-v5", "ALE/Breakout-v5"]
    assert call["eval_episodes"] == 3
    assert call["config"].max_env_steps == 100000
