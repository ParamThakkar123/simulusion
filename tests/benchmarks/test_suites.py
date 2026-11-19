"""The environment protocols every model's benchmarks share."""

import pytest

from simulusion.benchmarks import (
    ATARI100K_GAMES,
    ATARI_HUMAN_SCORES,
    ATARI_RANDOM_SCORES,
    BENCHMARKS,
    atari_game,
    get_benchmark,
    human_normalized_score,
)


@pytest.mark.parametrize(
    "spelling", ["ALE/MsPacman-v5", "MsPacman-v5", "MsPacman", "ms_pacman"]
)
def test_atari_game_accepts_every_spelling(spelling):
    assert atari_game(spelling) == "ms_pacman"


def test_every_atari100k_game_has_reference_scores():
    for game in ATARI100K_GAMES:
        assert ATARI_HUMAN_SCORES[game] > ATARI_RANDOM_SCORES[game], game


def test_human_normalized_score_anchors():
    assert human_normalized_score(14.6, "ALE/Pong-v5") == pytest.approx(1.0)
    assert human_normalized_score(-20.7, "Pong") == pytest.approx(0.0)
    with pytest.raises(KeyError, match="No human/random"):
        human_normalized_score(0.0, "not_a_game")


def test_env_fields_are_copies_and_carry_the_protocol():
    atari = get_benchmark("atari-100k")
    fields = atari.env_fields("pong")
    assert fields["env"] == "ALE/Pong-v5"
    assert fields["action_repeat"] == 4 and fields["total_steps"] == 400_000
    fields["gym_make_kwargs"]["frameskip"] = 4
    assert atari.overrides["gym_make_kwargs"]["frameskip"] == 1


def test_bsuite_has_no_step_budget():
    assert "total_steps" not in BENCHMARKS["bsuite"].env_fields()


def test_dreamer_v3_and_other_models_share_the_tables():
    from simulusion.configs import diamond_config
    from simulusion.configs.dreamer_v3_benchmarks import BENCHMARKS as V3
    from simulusion.training import run_atari_100k as iris

    for name, spec in BENCHMARKS.items():
        assert V3[name].tasks == spec.tasks
        assert V3[name].overrides == spec.overrides
    assert diamond_config.HUMAN_SCORES["MsPacman-v5"] == ATARI_HUMAN_SCORES["ms_pacman"]
    assert iris.HUMAN_SCORES["UpNDown"] == ATARI_HUMAN_SCORES["up_n_down"]
