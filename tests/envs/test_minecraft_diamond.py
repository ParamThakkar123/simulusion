"""Minecraft Diamond follows the DreamerV3 reference environment.

MineRL 0.4.4 needs Linux, Python 3.11 and Java, so these tests drive the
adapter with a fake backend that has MineRL's nested observation and action
layout (``inventory/<item>``, ``equipped_items/mainhand/type``, ``life_stats``).
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from simulusion.envs.minecraft_diamond import (  # noqa: E402
    BASIC_ACTIONS,
    DIAMOND_ACTIONS,
    DIAMOND_MILESTONES,
    MinecraftDiamondEnv,
)

ITEMS = ("air", "dirt", "log", "log2", *DIAMOND_MILESTONES[1:])
EQUIP = ("air", "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "other")


class _EnumSpace(gym.spaces.Discrete):
    """Stands in for MineRL's Enum space, which lists its values."""

    def __init__(self, values):
        super().__init__(len(values))
        self.values = np.array(values)


class FakeMineRL:
    """MineRL 0.4.4-shaped backend whose state the test sets directly."""

    def __init__(self, pov_shape=(64, 64, 3)):
        self.pov_shape = pov_shape
        inventory = {
            item: gym.spaces.Box(0, 2304, (), dtype=np.int64) for item in ITEMS
        }
        self.observation_space = gym.spaces.Dict(
            {
                "pov": gym.spaces.Box(0, 255, pov_shape, dtype=np.uint8),
                "inventory": gym.spaces.Dict(inventory),
                "equipped_items": gym.spaces.Dict(
                    {"mainhand": gym.spaces.Dict({"type": _EnumSpace(EQUIP)})}
                ),
                "life_stats": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(0, 300, (), dtype=np.float32)
                        for name in ("life", "food", "air")
                    }
                ),
            }
        )
        enum_keys = ("place", "equip", "craft", "nearbyCraft", "nearbySmelt")
        keys = ("forward", "back", "left", "right", "jump", "sneak", "attack")
        self.action_space = gym.spaces.Dict(
            {
                **{key: gym.spaces.Discrete(2) for key in keys},
                "camera": gym.spaces.Box(-180, 180, (2,), dtype=np.float32),
                **{key: _EnumSpace(["none", "planks"]) for key in enum_keys},
            }
        )
        self.inventory = {item: 0 for item in ITEMS}
        self.life = 20.0
        self.equipped = "air"
        self.done = False
        self.info: dict = {}
        self.actions: list[dict] = []
        self.resets = 0

    def _obs(self):
        return {
            "pov": np.full(self.pov_shape, 7, dtype=np.uint8),
            "inventory": {k: np.array(v) for k, v in self.inventory.items()},
            "equipped_items": {"mainhand": {"type": self.equipped}},
            "life_stats": {
                "life": np.array(self.life),
                "food": np.array(20.0),
                "air": np.array(300.0),
            },
        }

    def reset(self):
        self.resets += 1
        return self._obs()

    def step(self, action):
        self.actions.append(action)
        return self._obs(), 0.0, self.done, dict(self.info)


def _act(env, name):
    return env.action_names.index(name)


def test_action_space_is_the_25_reference_actions():
    env = MinecraftDiamondEnv(env=FakeMineRL())
    assert env.action_names == tuple(DIAMOND_ACTIONS)
    assert len(env.action_names) == 25
    assert env.action_space.shape == (25,)
    assert env.action_space.one_hot
    assert MinecraftDiamondEnv("wood", env=FakeMineRL()).action_names == tuple(
        BASIC_ACTIONS
    )


def test_observations_match_the_paper():
    env = MinecraftDiamondEnv(env=FakeMineRL(pov_shape=(32, 32, 3)))
    obs = env.reset()
    n_inventory = len(ITEMS) - 1  # log2 is merged into log
    assert obs["image"].shape == (3, 64, 64) and obs["image"].dtype == np.uint8
    assert obs["inventory"].shape == obs["inventory_max"].shape == (n_inventory,)
    assert obs["equipped"].shape == (len(EQUIP),)
    assert obs["equipped"][0] == 1.0  # holding air
    assert obs["health"][0] == pytest.approx(1.0)
    assert obs["breath"][0] == pytest.approx(1.0)
    for key, space in env.observation_space.spaces.items():
        assert obs[key].shape == space.shape, key


def test_each_milestone_is_rewarded_once_per_episode():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    noop = _act(env, "noop")
    backend.inventory["log"] = 1
    assert env.step(noop)[1] == pytest.approx(1.0)
    backend.inventory["log"] = 3  # more logs: no second reward
    assert env.step(noop)[1] == pytest.approx(0.0)
    backend.inventory["log"] = 0
    backend.inventory["planks"] = 4
    backend.inventory["stick"] = 2
    obs, reward, _, info = env.step(noop)
    assert reward == pytest.approx(2.0)
    backend.inventory["log"] = 1  # regained after losing it: still no reward
    assert env.step(noop)[1] == pytest.approx(0.0)
    assert info["achievements"]["planks"] and not info["achievements"]["diamond"]
    log = env._inv_keys.index("inventory/log")
    assert obs["inventory_max"][log] == 3


def test_log2_counts_as_log():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    backend.inventory["log2"] = 2
    obs, reward, _, _ = env.step(_act(env, "noop"))
    assert reward == pytest.approx(1.0)
    assert obs["inventory"][env._inv_keys.index("inventory/log")] == 2


def test_health_changes_are_rewarded_on_the_unit_scale():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    backend.life = 18.0  # one heart lost: 0.01 * (18 - 20) / 20
    assert env.step(_act(env, "noop"))[1] == pytest.approx(-0.001)


def test_wood_rewards_every_log():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv("wood", env=backend)
    env.reset()
    backend.inventory["log"] = 3
    assert env.step(_act(env, "noop"))[1] == pytest.approx(3.0)


def test_jump_is_held_and_moves_forward():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend, sticky_jump=3)
    env.reset()
    env.step(_act(env, "jump"))
    for _ in range(3):
        env.step(_act(env, "noop"))
    jumps = [(a["jump"], a["forward"]) for a in backend.actions]
    assert jumps == [(1, 1), (1, 1), (1, 1), (0, 0)]


def test_attack_is_not_held_with_fast_block_breaking():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend, break_speed=100.0)
    env.reset()
    env.step(_act(env, "attack"))
    env.step(_act(env, "noop"))
    assert [a["attack"] for a in backend.actions] == [1, 0]

    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend, break_speed=1.0, sticky_attack=2)
    env.reset()
    env.step(_act(env, "attack"))
    env.step(_act(env, "noop"))
    env.step(_act(env, "noop"))
    assert [a["attack"] for a in backend.actions] == [1, 1, 0]


def test_pitch_stays_within_limits():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    for _ in range(6):
        env.step(_act(env, "turn_down"))
    pitches = [float(a["camera"][0]) for a in backend.actions]
    assert pitches == [15.0, 15.0, 15.0, 15.0, 0.0, 0.0]
    env.step(_act(env, "turn_left"))
    assert backend.actions[-1]["camera"].tolist() == [0.0, -15.0]


def test_actions_only_carry_fields_the_backend_declares():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    env.step(np.eye(25, dtype=np.float32)[_act(env, "craft_planks")])
    action = backend.actions[-1]
    assert "sprint" not in action
    assert action["craft"] == "planks"
    assert action["nearbyCraft"] == "none"


def test_episodes_end_on_death_or_after_the_step_limit():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend, length=3)
    env.reset()
    results = [env.step(_act(env, "noop")) for _ in range(3)]
    assert [done for _, _, done, _ in results] == [False, False, True]
    assert results[-1][3]["truncated"] and not results[-1][3]["terminated"]

    env.reset()
    backend.done = True
    _, _, done, info = env.step(_act(env, "noop"))
    assert done and info["terminated"]


def test_a_backend_error_cuts_the_episode_short():
    # The reference drops the failed step and resets: no reward, no terminal.
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    last = env.reset()
    backend.inventory["log"] = 1
    backend.info = {"error": "malmo crashed"}
    obs, reward, done, info = env.step(_act(env, "noop"))
    assert done and info["truncated"] and not info["terminated"]
    assert reward == 0.0 and "error" not in info
    assert info["backend_error"] == "malmo crashed"
    for key, value in last.items():
        np.testing.assert_array_equal(obs[key], value)


def test_reset_starts_a_new_episode():
    backend = FakeMineRL()
    env = MinecraftDiamondEnv(env=backend)
    env.reset()
    backend.inventory["log"] = 1
    env.step(_act(env, "noop"))
    obs = env.reset()  # the world keeps the log, but it is the episode start
    assert env.step(_act(env, "noop"))[1] == pytest.approx(0.0)
    assert not env.achievements()["planks"]
    assert obs["inventory_max"][env._inv_keys.index("inventory/log")] == 1


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown Minecraft task"):
        MinecraftDiamondEnv("climb", env=FakeMineRL())


def test_missing_minerl_explains_the_install():
    from simulusion.envs import minecraft_diamond

    if minecraft_diamond.importlib.util.find_spec("minerl") is not None:
        pytest.skip("minerl is installed")
    with pytest.raises(ImportError, match="MineRL 0.4.4"):
        MinecraftDiamondEnv()


def test_make_env_builds_minecraft_with_image_and_vectors(monkeypatch):
    from simulusion.configs.dreamer_config import DreamerConfig
    from simulusion.envs import minecraft_diamond
    from simulusion.models import dreamer as dreamer_module

    backend = FakeMineRL()
    speeds = []

    def fake_backend(size, speed):
        speeds.append(speed)
        return backend

    monkeypatch.setattr(minecraft_diamond, "make_minerl_backend", fake_backend)
    cfg = DreamerConfig(
        env_backend="minecraft",
        env="diamond",
        action_repeat=1,
        minecraft_break_speed=100.0,
    )
    env = dreamer_module.make_env(cfg)
    assert speeds == [100.0]
    assert env.action_space.shape == (25,)
    keys = {"image", "inventory", "inventory_max", "equipped", "health"}
    assert keys <= set(env.observation_space.spaces)
    obs = env.reset()
    obs, _, done, _ = env.step(env.action_space.sample())
    assert keys <= set(obs) and not done


def test_make_env_ends_minecraft_episodes_after_minecraft_length(monkeypatch):
    # The episode length is the benchmark's (36000); time_limit does not apply.
    from simulusion.configs.dreamer_config import DreamerConfig
    from simulusion.envs import minecraft_diamond
    from simulusion.models import dreamer as dreamer_module

    monkeypatch.setattr(
        minecraft_diamond, "make_minerl_backend", lambda size, speed: FakeMineRL()
    )
    assert DreamerConfig().minecraft_length == 36000
    cfg = DreamerConfig(
        env_backend="minecraft",
        env="diamond",
        action_repeat=1,
        time_limit=2,
        minecraft_length=4,
    )
    env = dreamer_module.make_env(cfg)
    env.reset()
    dones = [env.step(env.action_space.sample())[2] for _ in range(4)]
    assert dones == [False, False, False, True]
