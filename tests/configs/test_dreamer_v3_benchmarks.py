"""DreamerV3 benchmark presets follow Table 2 of the paper."""

import dataclasses

import pytest

from simulusion.configs.dreamer_v3_benchmarks import (
    BENCHMARKS,
    atari_env_id,
    benchmark_config,
)
from simulusion.configs.dreamer_v3_config import MODEL_SIZES, DreamerV3Config

# Table 2: tasks, env steps, action repeat, env instances, replay ratio, model size.
TABLE_2 = {
    "dmlab": (30, 100_000_000, 4, 16, 32, "200m"),
    "procgen": (16, 50_000_000, 1, 16, 64, "200m"),
    "atari": (57, 200_000_000, 4, 16, 32, "200m"),
    "atari100k": (26, 400_000, 4, 1, 128, "200m"),
    "bsuite": (23, None, 1, 1, 1024, "200m"),
    "minecraft": (1, 100_000_000, 1, 64, 32, "200m"),
    "dmc-proprio": (18, 500_000, 2, 16, 512, "12m"),
    "dmc-visual": (20, 1_000_000, 2, 16, 512, "12m"),
}


@pytest.mark.parametrize("name", sorted(TABLE_2))
def test_benchmarks_match_table_2(name):
    spec = BENCHMARKS[name]
    tasks, steps, repeat, instances, ratio, size = TABLE_2[name]
    assert len(spec.tasks) == len(set(spec.tasks)) == tasks
    assert spec.env_steps == steps
    assert spec.action_repeat == repeat
    assert spec.env_instances == instances
    assert spec.replay_ratio == ratio
    assert spec.model_size == size


@pytest.mark.parametrize("name", sorted(TABLE_2))
def test_configs_carry_the_protocol(name):
    spec = BENCHMARKS[name]
    cfg = benchmark_config(name)
    # DMLab repeats inside Lab; the frames per agent step still match Table 2.
    inner = cfg.dmlab_action_repeat if cfg.env_backend == "dmlab" else 1
    assert cfg.action_repeat * inner == spec.action_repeat
    assert cfg.replay_ratio == spec.replay_ratio
    assert cfg.model_size == spec.model_size
    assert cfg.hidden_size == MODEL_SIZES[spec.model_size]["hidden_size"]
    if spec.env_steps is not None:
        assert cfg.total_steps == spec.env_steps
    for key, value in spec.overrides.items():
        assert getattr(cfg, key) == value


def test_replay_ratio_sets_the_update_steps():
    cfg = benchmark_config("dmc-proprio", "walker-walk")
    # 1000 frames / action repeat 2 = 500 policy steps, x512 / (16 x 64).
    assert cfg.update_steps == 250


def test_proprio_and_visual_control_differ_in_inputs_and_tasks():
    assert benchmark_config("dmc-proprio").dmc_obs_type == "state"
    assert benchmark_config("dmc-visual").dmc_obs_type == "pixels"
    proprio, visual = BENCHMARKS["dmc-proprio"].tasks, BENCHMARKS["dmc-visual"].tasks
    assert set(visual) - set(proprio) == {"quadruped-run", "quadruped-walk"}


def test_atari_steps_single_frames_so_the_action_repeat_is_not_doubled():
    for name in ("atari", "atari100k"):
        assert benchmark_config(name).gym_make_kwargs["frameskip"] == 1
    assert (
        benchmark_config("atari").gym_make_kwargs["repeat_action_probability"] == 0.25
    )
    assert benchmark_config("atari").gym_make_kwargs["full_action_space"] is True
    no_sticky = benchmark_config("atari100k").gym_make_kwargs
    assert no_sticky["repeat_action_probability"] == 0.0


def test_procgen_is_hard_with_unlimited_levels():
    cfg = benchmark_config("procgen", "coinrun")
    assert cfg.procgen_distribution_mode == "hard"
    assert cfg.procgen_num_levels == 0


def test_dmlab_repeats_actions_inside_lab():
    cfg = benchmark_config("dmlab")
    assert cfg.dmlab_action_repeat == 4 and cfg.action_repeat == 1
    assert cfg.env == "contributed/dmlab30/rooms_collect_good_objects_train"


def test_dmlab_uses_the_fixed_popart_action_set_on_train_levels():
    cfg = benchmark_config("dmlab", "language_select_described_object")
    assert cfg.dmlab_action_set == "popart"
    assert cfg.dmlab_mode == "train"


def test_bsuite_learns_from_the_observation_vector_of_a_setting():
    cfg = benchmark_config("bsuite", "catch")
    assert cfg.env == "catch/0"
    assert cfg.bsuite_obs_type == "state"
    assert benchmark_config("bsuite", "deep_sea/7").env == "deep_sea/7"


def test_minecraft_encodes_the_image_and_status_vectors():
    cfg = benchmark_config("minecraft")
    assert cfg.env == "diamond"
    assert cfg.time_limit == 36_000
    assert cfg.minecraft_break_speed == 100.0
    assert cfg.obs_keys == [
        "image",
        "inventory",
        "inventory_max",
        "equipped",
        "health",
        "hunger",
        "breath",
    ]
    cfg.obs_keys.append("x")
    assert "x" not in BENCHMARKS["minecraft"].overrides["obs_keys"]


def test_task_names():
    assert atari_env_id("ms_pacman") == "ALE/MsPacman-v5"
    assert atari_env_id("up_n_down") == "ALE/UpNDown-v5"
    assert benchmark_config("atari100k", "pong").env == "ALE/Pong-v5"
    assert benchmark_config("atari100k", "ALE/Pong-v5").env == "ALE/Pong-v5"
    assert benchmark_config("dmc-visual", "cheetah-run").env == "cheetah-run"


def test_overrides_win_and_resize_the_model():
    cfg = benchmark_config("atari100k", "pong", model_size="12m", total_steps=1000)
    assert cfg.total_steps == 1000
    assert cfg.hidden_size == MODEL_SIZES["12m"]["hidden_size"]
    assert cfg.recurrent_units == MODEL_SIZES["12m"]["recurrent_units"]
    # The protocol fields are still there.
    assert cfg.replay_ratio == 128 and cfg.gym_make_kwargs["frameskip"] == 1


def test_presets_do_not_share_mutable_options():
    benchmark_config("atari").gym_make_kwargs["frameskip"] = 4
    assert BENCHMARKS["atari"].overrides["gym_make_kwargs"]["frameskip"] == 1


def test_aliases_and_errors():
    assert benchmark_config("Proprio").dmc_obs_type == "state"
    assert benchmark_config("dmlab30").dmlab_action_repeat == 4
    assert benchmark_config("minecraft-diamond").env_backend == "minecraft"
    with pytest.raises(ValueError, match="Unknown benchmark"):
        benchmark_config("crafter")
    with pytest.raises(ValueError, match="not_a_field"):
        benchmark_config("dmc-proprio", not_a_field=1)


class TestModelSizeChanges:
    def test_replace_refills_the_widths(self):
        cfg = dataclasses.replace(DreamerV3Config(), model_size="200m")
        for key, value in MODEL_SIZES["200m"].items():
            assert getattr(cfg, key) == value

    def test_explicit_widths_are_kept(self):
        cfg = dataclasses.replace(DreamerV3Config(), model_size="50m", hidden_size=100)
        assert cfg.hidden_size == 100
        assert cfg.recurrent_units == MODEL_SIZES["50m"]["recurrent_units"]

    def test_create_config_resizes(self):
        import simulusion

        cfg = simulusion.create_config("dreamer-v3", model_size="25m")
        assert cfg.hidden_size == MODEL_SIZES["25m"]["hidden_size"]
