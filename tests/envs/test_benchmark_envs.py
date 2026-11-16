"""Crafter, Memory Maze, Atari 100k and Craftax adapters.

Crafter, Memory Maze and Craftax run against small fakes of their packages, so
the adapter logic is tested everywhere; each also has a test against the real
package that runs only when it is installed. Atari 100k drives the real ALE,
which ships with ale-py.
"""

from __future__ import annotations

import enum
import importlib.machinery
import importlib.util
import math
import sys
import types

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from simulusion.configs.dreamer_config import DreamerConfig  # noqa: E402
from simulusion.envs.crafter_env import (  # noqa: E402
    CRAFTER_ACHIEVEMENTS,
    CrafterEnv,
    crafter_score,
    crafter_score_from_rates,
    crafter_success_rates,
    normalize_crafter_task,
)
from simulusion.envs.memory_maze_env import (  # noqa: E402
    MemoryMazeEnv,
    normalize_memory_maze_task,
)


def _one_hot(index: int, n: int) -> np.ndarray:
    action = -np.ones(n, np.float32)
    action[index] = 1.0
    return action


def _assert_contract(info: dict, *, done: bool) -> None:
    assert {"terminated", "truncated", "discount", "action", "executed_action"} <= set(
        info
    )
    assert info["terminated"] or info["truncated"] or not done
    assert not (info["terminated"] and info["truncated"])
    if info["terminated"]:
        assert float(info["discount"]) == 0.0


# --------------------------------------------------------------------------- #
# Crafter
# --------------------------------------------------------------------------- #


class _FakeCrafterEnv:
    """Ends by death after ``death_at`` steps or by time limit after ``length``."""

    death_at = None

    def __init__(self, area, view, size, reward, length, seed):
        self.kwargs = dict(
            area=area, view=view, size=size, reward=reward, length=length
        )
        self._seed = seed
        self._episode = 0
        self._length = length
        self.action_space = types.SimpleNamespace(n=17)
        self.actions: list[int] = []

    def _frame(self):
        h, w = self.kwargs["size"]
        return np.full((h, w, 3), self._episode % 255, np.uint8)

    def reset(self):
        self._episode += 1
        self.t = 0
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        dead = self.death_at is not None and self.t >= self.death_at
        over = self._length is not None and self.t >= self._length
        info = {
            "discount": 0.0 if dead else 1.0,
            "achievements": {"collect_wood": int(self.t >= 2)},
            "inventory": {},
        }
        return self._frame(), 1.0 if self.kwargs["reward"] else 0.0, dead or over, info


def _fake_module(name: str) -> types.ModuleType:
    """A stand-in package that importlib.util.find_spec accepts."""
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


@pytest.fixture
def fake_crafter(monkeypatch):
    module = _fake_module("crafter")
    module.Env = _FakeCrafterEnv
    module.constants = types.SimpleNamespace(actions=[f"a{i}" for i in range(17)])
    monkeypatch.setitem(sys.modules, "crafter", module)
    _FakeCrafterEnv.death_at = None
    return module


def test_crafter_task_names():
    assert normalize_crafter_task("crafter_reward") == "reward"
    assert normalize_crafter_task("CrafterNoReward-v1") == "noreward"
    with pytest.raises(ValueError):
        normalize_crafter_task("crafter_hard")


def test_crafter_follows_the_image_contract(fake_crafter):
    env = CrafterEnv(seed=3, size=(32, 48), length=5)
    obs = env.reset()
    assert obs["image"].shape == (3, 32, 48) and obs["image"].dtype == np.uint8
    assert env.observation_space["image"].shape == (3, 32, 48)
    assert env.action_space.n == 17
    obs, reward, done, info = env.step(_one_hot(5, 17))
    assert env._env.actions == [5]
    assert info["executed_action"] == 5
    np.testing.assert_array_equal(info["action"], _one_hot(5, 17))
    assert reward == 1.0 and not done
    _assert_contract(info, done=done)
    assert env.render().shape == (32, 48, 3)


def test_crafter_time_limit_is_a_truncation(fake_crafter):
    env = CrafterEnv(length=3)
    env.reset()
    for _ in range(3):
        _, _, done, info = env.step(0)
    assert done and info["truncated"] and not info["terminated"]
    assert float(info["discount"]) == 1.0


def test_crafter_death_is_a_termination(fake_crafter):
    _FakeCrafterEnv.death_at = 2
    env = CrafterEnv(length=100)
    env.reset()
    env.step(0)
    _, _, done, info = env.step(0)
    assert done and info["terminated"] and not info["truncated"]
    assert float(info["discount"]) == 0.0
    assert info["achievements"] == {"collect_wood": 1}


def test_crafter_noreward_and_reseeding(fake_crafter):
    env = CrafterEnv("noreward", seed=1)
    assert env._env.kwargs["reward"] is False
    env.reset()
    env.reset(seed=9)
    assert env._env._seed == 9 and env._env._episode == 1


def test_crafter_score_matches_its_definition():
    zeros = {name: 0.0 for name in CRAFTER_ACHIEVEMENTS}
    assert crafter_score_from_rates(zeros) == pytest.approx(0.0)
    assert crafter_score_from_rates(
        {name: 100.0 for name in CRAFTER_ACHIEVEMENTS}
    ) == pytest.approx(100.0)
    one = dict(zeros, collect_wood=100.0)
    assert crafter_score_from_rates(one) == pytest.approx(
        math.exp(math.log(101) / 22) - 1
    )
    with pytest.raises(ValueError, match="missing"):
        crafter_score_from_rates({"collect_wood": 50.0})


def test_crafter_success_rates_count_episodes_with_an_unlock():
    episodes = [
        {"collect_wood": 3, "wake_up": 1},
        {"collect_wood": 0},
        {"collect_wood": 1},
    ]
    rates = crafter_success_rates(episodes)
    assert rates["collect_wood"] == pytest.approx(200.0 / 3)
    assert rates["wake_up"] == pytest.approx(100.0 / 3)
    assert rates["collect_diamond"] == 0.0
    assert crafter_score(episodes) == pytest.approx(crafter_score_from_rates(rates))


def test_real_crafter_episode_ends_cleanly():
    pytest.importorskip("crafter")
    env = CrafterEnv(seed=0, length=50)
    obs = env.reset()
    assert obs["image"].shape == (3, 64, 64)
    for _ in range(50):
        _, _, done, info = env.step(env.action_space.sample())
        if done:
            break
    assert done
    _assert_contract(info, done=True)
    assert set(info["achievements"]) == set(CRAFTER_ACHIEVEMENTS)


# --------------------------------------------------------------------------- #
# Memory Maze
# --------------------------------------------------------------------------- #


class _FakeTimeStep:
    def __init__(self, image, last=False, reward=0.0):
        self.observation = {"image": image, "target_color": np.array([1.0, 0.0, 0.0])}
        self.reward = reward
        self.discount = 1.0
        self._last = last

    def last(self):
        return self._last


class _FakeMemoryMaze:
    def __init__(self, length, **kwargs):
        self.kwargs = kwargs
        self.length = length
        self.actions: list[int] = []
        self.closed = False

    def action_spec(self):
        return types.SimpleNamespace(num_values=6)

    def _image(self):
        res = self.kwargs["camera_resolution"]
        return np.full((res, res, 3), 7, np.uint8)

    def reset(self):
        self.t = 0
        return _FakeTimeStep(self._image())

    def step(self, action):
        self.actions.append(action)
        self.t += 1
        return _FakeTimeStep(
            self._image(), last=self.t >= self.length, reward=float(action == 2)
        )

    def close(self):
        self.closed = True


@pytest.fixture
def fake_memory_maze(monkeypatch):
    tasks = types.ModuleType("memory_maze.tasks")
    tasks.memory_maze_9x9 = lambda **kw: _FakeMemoryMaze(4, **kw)
    tasks.memory_maze_15x15 = lambda **kw: _FakeMemoryMaze(8, **kw)
    package = types.ModuleType("memory_maze")
    package.tasks = tasks
    monkeypatch.setitem(sys.modules, "memory_maze", package)
    monkeypatch.setitem(sys.modules, "memory_maze.tasks", tasks)
    return tasks


@pytest.mark.parametrize("name", ["9x9", "memory_maze_9x9", "MemoryMaze-9x9-v0"])
def test_memory_maze_task_names(name):
    assert normalize_memory_maze_task(name) == "9x9"


def test_memory_maze_rejects_unknown_sizes_and_variants():
    with pytest.raises(ValueError, match="Valid sizes"):
        normalize_memory_maze_task("7x7")
    with pytest.raises(ValueError, match="different observations"):
        normalize_memory_maze_task("MemoryMaze-9x9-ExtObs-v0")


def test_memory_maze_follows_the_contract(fake_memory_maze):
    env = MemoryMazeEnv("MemoryMaze-9x9-v0", seed=5, size=(32, 32))
    assert env._env.kwargs["seed"] == 5
    assert env._env.kwargs["discrete_actions"] is True
    assert env.max_episode_steps == 1000
    obs = env.reset()
    assert obs["image"].shape == (3, 32, 32)
    np.testing.assert_array_equal(obs["target_color"], [1.0, 0.0, 0.0])
    obs, reward, done, info = env.step(_one_hot(2, 6))
    assert env._env.actions == [2] and reward == 1.0 and not done
    _assert_contract(info, done=done)
    np.testing.assert_array_equal(info["vector_observation"], [1.0, 0.0, 0.0])


def test_memory_maze_time_limit_is_a_truncation(fake_memory_maze):
    env = MemoryMazeEnv("15x15")
    env.reset()
    for _ in range(8):
        _, _, done, info = env.step(0)
    assert done and info["truncated"] and not info["terminated"]
    assert float(info["discount"]) == 1.0


def test_memory_maze_reseeding_rebuilds_the_maze(fake_memory_maze):
    env = MemoryMazeEnv(seed=1)
    old = env._env
    env.reset(seed=2)
    assert old.closed and env._env.kwargs["seed"] == 2


def test_memory_maze_reserves_its_own_options(fake_memory_maze):
    with pytest.raises(ValueError, match="discrete_actions"):
        MemoryMazeEnv(discrete_actions=False)


def test_memory_maze_explains_a_missing_extension(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("memory_maze"):
            raise TypeError("NoneType takes no arguments")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "memory_maze", raising=False)
    monkeypatch.delitem(sys.modules, "memory_maze.tasks", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ImportError, match="compiled RandomMaze"):
        MemoryMazeEnv()


def test_real_memory_maze_runs():
    labmaze = pytest.importorskip("labmaze")
    if getattr(labmaze, "RandomMaze", None) is None:
        pytest.skip("labmaze has no compiled RandomMaze")
    pytest.importorskip("memory_maze")
    env = MemoryMazeEnv("9x9", seed=0)
    obs = env.reset()
    assert obs["image"].shape == (3, 64, 64)
    _, _, done, info = env.step(env.action_space.sample())
    assert not done
    _assert_contract(info, done=False)
    env.close()


# --------------------------------------------------------------------------- #
# Atari 100k
# --------------------------------------------------------------------------- #


def test_atari_protocols_match_the_reference_implementations():
    from simulusion.envs.atari100k_env import ATARI_100K_PROTOCOLS as P

    # IRIS: v4 NoFrameskip, NoopReset [1, 30], EpisodicLifeEnv, 20000 frames,
    # PIL resize before MaxAndSkip.
    assert (P["iris"].sticky, P["iris"].noop_min, P["iris"].noop_max) == (0.0, 1, 30)
    assert P["iris"].life_loss == "terminate" and P["iris"].continue_noop
    assert P["iris"].max_frames == 20000 and P["iris"].resize_before_pool
    assert (P["iris-eval"].noop_max, P["iris-eval"].max_frames) == (1, 108000)
    assert P["iris-eval"].life_loss == "ignore"
    # DIAMOND: noops [1, 30], INTER_AREA after pooling, done on life loss.
    assert (P["diamond"].noop_min, P["diamond"].resize) == (1, "area")
    assert P["diamond"].life_loss == "terminate" and not P["diamond"].continue_noop
    # STORM keeps ALE v5's sticky actions and takes no no-ops.
    assert (P["storm"].sticky, P["storm"].noop_max) == (0.25, 0)
    # DreamerV3: noops [0, 30], lives unused.
    assert (P["dreamerv3"].noop_min, P["dreamerv3"].life_loss) == (0, "ignore")
    assert P["atari100k"] is P["dreamerv3"]
    for protocol in P.values():
        assert (protocol.frame_skip, protocol.pool_last, protocol.size) == (
            4,
            2,
            (64, 64),
        )


def test_atari_protocol_validation():
    from simulusion.envs.atari100k_env import AtariProtocol, resolve_atari_protocol

    with pytest.raises(ValueError):
        AtariProtocol(life_loss="reset")
    with pytest.raises(ValueError):
        AtariProtocol(noop_min=5, noop_max=1)
    with pytest.raises(ValueError, match="Unknown Atari protocol"):
        resolve_atari_protocol("muzero")
    assert resolve_atari_protocol("IRIS", max_frames=40).max_frames == 40


needs_ale = pytest.mark.skipif(
    importlib.util.find_spec("ale_py") is None, reason="Atari 100k needs ale-py"
)


@needs_ale
def test_atari_game_names_resolve_to_rom_ids():
    from simulusion.configs.diamond_config import ATARI_100K_GAMES
    from simulusion.envs.atari100k_env import ATARI_100K_ROMS, normalize_atari_game

    assert [normalize_atari_game(g) for g in ATARI_100K_GAMES] == list(ATARI_100K_ROMS)
    for name in (
        "MsPacman",
        "ALE/MsPacman-v5",
        "MsPacmanNoFrameskip-v4",
        "atari100k_ms_pacman",
    ):
        assert normalize_atari_game(name) == "ms_pacman"
    assert normalize_atari_game("UpNDown") == "up_n_down"
    assert normalize_atari_game("james_bond") == "jamesbond"
    with pytest.raises(ValueError):
        normalize_atari_game("NotAGame")


def _rollout(env, steps, action=1):
    frames = []
    env.reset()
    for _ in range(steps):
        obs, _, done, _ = env.step(action)
        frames.append(obs["image"])
        if done:
            env.reset()
    return np.stack(frames)


@needs_ale
def test_atari_observations_and_actions():
    from simulusion.envs.atari100k_env import Atari100kEnv

    env = Atari100kEnv("Breakout", seed=0)
    obs = env.reset()
    assert obs["image"].shape == (3, 64, 64) and obs["image"].dtype == np.uint8
    assert env.action_space.n == 4 and env.action_meanings[0] == "NOOP"
    _, _, _, info = env.step(_one_hot(1, 4))
    assert info["executed_action"] == 1
    _assert_contract(info, done=False)
    assert env.render().shape == (210, 160, 3)


@needs_ale
def test_atari_rollouts_are_reproducible_per_seed():
    from simulusion.envs.atari100k_env import Atari100kEnv

    a = _rollout(Atari100kEnv("Pong", protocol="iris", seed=1), 60, action=2)
    b = _rollout(Atari100kEnv("Pong", protocol="iris", seed=1), 60, action=2)
    np.testing.assert_array_equal(a, b)
    env = Atari100kEnv("Pong", protocol="iris", seed=1)
    env.reset(seed=4)
    first = [env.step(2)[0]["image"] for _ in range(30)]
    env.reset(seed=4)
    second = [env.step(2)[0]["image"] for _ in range(30)]
    np.testing.assert_array_equal(np.stack(first), np.stack(second))


@needs_ale
def test_atari_frame_limit_is_a_truncation():
    from simulusion.envs.atari100k_env import Atari100kEnv

    env = Atari100kEnv("Pong", protocol="dreamerv3", seed=0, max_frames=40, noop_max=0)
    env.reset()
    for step in range(1, 11):
        _, _, done, info = env.step(0)
    assert done and step == 10
    assert info["truncated"] and not info["terminated"]
    assert info["episode_frame_number"] == 40
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def _play_until_life_lost(env, limit=3000):
    for _ in range(limit):
        _, _, done, info = env.step(1)  # FIRE launches the ball; then it is missed.
        if done:
            return info
    raise AssertionError("no episode end within the step limit")


@needs_ale
def test_life_loss_ends_the_episode_but_not_the_game():
    from simulusion.envs.atari100k_env import Atari100kEnv

    env = Atari100kEnv("Breakout", protocol="diamond", seed=0)
    env.reset()
    info = _play_until_life_lost(env)
    assert info["life_lost"] and not info["game_over"]
    assert info["terminated"] and float(info["discount"]) == 0.0
    frames = info["episode_frame_number"]
    env.reset()
    _, _, _, info = env.step(0)
    # The game continued: the frame counter kept running from the lost life.
    assert info["episode_frame_number"] == frames + 4
    assert info["lives"] == 4


@needs_ale
def test_iris_continues_with_a_noop_step():
    from simulusion.envs.atari100k_env import Atari100kEnv

    env = Atari100kEnv("Breakout", protocol="iris", seed=0)
    env.reset()
    frames = _play_until_life_lost(env)["episode_frame_number"]
    env.reset()
    _, _, _, info = env.step(0)
    assert info["episode_frame_number"] == frames + 8


@needs_ale
def test_dreamerv3_ignores_life_loss():
    from simulusion.envs.atari100k_env import Atari100kEnv

    env = Atari100kEnv("Breakout", protocol="dreamerv3", seed=0)
    env.reset()
    info = _play_until_life_lost(env, limit=10000)
    assert info["game_over"] and info["lives"] == 0


@needs_ale
def test_area_resize_protocols_use_opencv():
    pytest.importorskip("cv2")
    from simulusion.envs.atari100k_env import Atari100kEnv

    obs = Atari100kEnv("Pong", protocol="storm", seed=0).reset()
    assert obs["image"].shape == (3, 64, 64)


@needs_ale
def test_dreamer_builds_atari100k_with_the_eval_protocol():
    from simulusion.models.dreamer import make_env

    cfg = DreamerConfig(
        env_backend="atari100k", env="Pong", action_repeat=1, atari100k_protocol="iris"
    )
    train = make_env(cfg)
    evaluation = make_env(cfg, evaluation=True)

    def base(env):
        while hasattr(env, "_env") and not hasattr(env, "protocol"):
            env = env._env
        while hasattr(env, "env") and not hasattr(env, "protocol"):
            env = env.env
        return env

    assert base(train).protocol.life_loss == "terminate"
    assert base(evaluation).protocol.life_loss == "ignore"
    assert base(evaluation).protocol.max_frames == 108000
    with pytest.raises(ValueError, match="action_repeat must be 1"):
        make_env(DreamerConfig(env_backend="atari100k", env="Pong", action_repeat=4))


# --------------------------------------------------------------------------- #
# Craftax
# --------------------------------------------------------------------------- #


class _FakeCraftaxState:
    def __init__(self, timestep, health, achievements):
        self.timestep = timestep
        self.player_health = health
        self.achievements = achievements


class _FakeCraftaxEnv:
    die_at = None

    def __init__(self, pixels):
        self.pixels = pixels
        self.default_params = types.SimpleNamespace(max_timesteps=5)
        self.calls: list[int] = []

    def action_space(self, params):
        return types.SimpleNamespace(n=17)

    def observation_space(self, params):
        return types.SimpleNamespace(shape=(1345,))

    def _obs(self, t):
        if self.pixels:
            return np.full((63, 63, 3), 0.5, np.float32)
        return np.full((1345,), t, np.float32)

    def reset(self, key, params):
        return self._obs(0), _FakeCraftaxState(0, 9.0, np.zeros(22, bool))

    def step(self, key, state, action, params):
        self.calls.append(int(action))
        t = state.timestep + 1
        achievements = state.achievements.copy()
        achievements[0] = True
        health = 0.0 if self.die_at is not None and t >= self.die_at else 9.0
        done = t >= params.max_timesteps or health <= 0
        new = _FakeCraftaxState(t, health, achievements)
        return self._obs(t), new, 1.0, done, {"score": 3.0}


@pytest.fixture
def fake_craftax(monkeypatch):
    jax = _fake_module("jax")
    jax.jit = lambda fn: fn
    jax.random = types.SimpleNamespace(
        PRNGKey=lambda seed: seed, split=lambda k: (k + 1, k + 2)
    )
    craftax_env = _fake_module("craftax.craftax_env")
    craftax_env.make_craftax_env_from_name = lambda name, auto_reset: _FakeCraftaxEnv(
        "Pixels" in name
    )
    names = {name.upper(): i for i, name in enumerate(CRAFTER_ACHIEVEMENTS)}
    constants = _fake_module("craftax.craftax_classic.constants")
    constants.Achievement = enum.Enum("Achievement", names)
    for name, module in {
        "jax": jax,
        "craftax": _fake_module("craftax"),
        "craftax.craftax_env": craftax_env,
        "craftax.craftax_classic": _fake_module("craftax.craftax_classic"),
        "craftax.craftax_classic.constants": constants,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    _FakeCraftaxEnv.die_at = None


def test_craftax_names():
    from simulusion.envs.craftax_env import normalize_craftax_env

    assert normalize_craftax_env("classic") == "Craftax-Classic-Pixels-v1"
    assert (
        normalize_craftax_env("craftax-classic-symbolic")
        == "Craftax-Classic-Symbolic-v1"
    )
    assert normalize_craftax_env("full") == "Craftax-Pixels-v1"
    with pytest.raises(ValueError):
        normalize_craftax_env("craftax-hard")


def test_craftax_pixels_follow_the_contract(fake_craftax):
    from simulusion.envs.craftax_env import CraftaxEnv

    env = CraftaxEnv("classic", seed=0)
    obs = env.reset()
    assert obs["image"].shape == (3, 64, 64) and obs["image"].dtype == np.uint8
    assert int(obs["image"].mean()) == 128  # 0.5 scaled to [0, 255]
    obs, reward, done, info = env.step(_one_hot(4, 17))
    assert env._env.calls == [4] and reward == 1.0 and not done
    _assert_contract(info, done=done)
    assert info["achievements"]["collect_coal"] == 1
    assert sum(info["achievements"].values()) == 1


def test_craftax_step_limit_is_a_truncation(fake_craftax):
    from simulusion.envs.craftax_env import CraftaxEnv

    env = CraftaxEnv("classic")
    env.reset()
    for _ in range(5):
        _, _, done, info = env.step(0)
    assert done and info["truncated"] and not info["terminated"]
    assert info["craftax_score"] == 3.0


def test_craftax_death_is_a_termination(fake_craftax):
    from simulusion.envs.craftax_env import CraftaxEnv

    _FakeCraftaxEnv.die_at = 2
    env = CraftaxEnv("classic")
    env.reset()
    env.step(0)
    _, _, done, info = env.step(0)
    assert done and info["terminated"] and float(info["discount"]) == 0.0


def test_craftax_symbolic_observations(fake_craftax):
    from simulusion.envs.craftax_env import CraftaxEnv

    env = CraftaxEnv("Craftax-Classic-Symbolic-v1")
    assert env.reset()["state"].shape == (1345,)
    _, _, _, info = env.step(0)
    np.testing.assert_array_equal(info["vector_observation"], np.ones(1345))
    with pytest.raises(RuntimeError, match="symbolic"):
        env.render()


def test_real_craftax_classic_steps():
    pytest.importorskip("craftax")
    from simulusion.envs.craftax_env import CraftaxEnv

    env = CraftaxEnv("classic", seed=0)
    assert env.reset()["image"].shape == (3, 64, 64)
    _, _, done, info = env.step(0)
    assert not done
    _assert_contract(info, done=False)
    assert set(info["achievements"]) == set(CRAFTER_ACHIEVEMENTS)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("backend", "factory"),
    [
        ("crafter", "make_crafter_env"),
        ("memory_maze", "make_memory_maze_env"),
        ("atari100k", "make_atari100k_env"),
        ("craftax", "make_craftax_env"),
    ],
)
def test_backends_are_registered(backend, factory):
    from simulusion.api import get_env_backend_spec
    from simulusion.models.dreamer import BENCHMARK_PROTOCOL_BACKENDS

    spec = get_env_backend_spec(backend)
    assert spec.factory_path == f"simulusion.envs:{factory}"
    assert backend in BENCHMARK_PROTOCOL_BACKENDS


def test_make_env_dispatches_to_the_new_backends(fake_crafter):
    from simulusion.envs import make_env

    assert isinstance(make_env("reward", backend="crafter"), CrafterEnv)
