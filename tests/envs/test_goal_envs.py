"""Goal-conditioned environments: the shared contract, then each environment's physics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulusion.envs.goal import GOAL_ENVS, goal_env_name, make_goal_env

SIZE = 32


def _make(name: str, **kwargs):
    if name == "pusht":
        pytest.importorskip("pymunk")
    if name == "cube":
        pytest.importorskip("ogbench")
    if name == "reacher_goal":
        pytest.importorskip("dm_control")
    return make_goal_env(name, size=SIZE, seed=0, **kwargs)


@pytest.fixture(params=sorted(GOAL_ENVS))
def env(request):
    environment = _make(request.param)
    yield environment
    environment.close()


# ------------------------------------------------------------------ contract


def test_reset_returns_frames_and_a_goal(env):
    obs = env.reset(seed=1)
    assert obs["image"].shape == (3, SIZE, SIZE) and obs["image"].dtype == np.uint8
    goal = env.goal_observation()["image"]
    assert goal.shape == (3, SIZE, SIZE)
    assert env.goal_state.shape == (env.state_dim,)
    assert env.observation_space["image"].shape == (3, SIZE, SIZE)


def test_step_reports_success_distance_and_state(env):
    env.reset(seed=1)
    obs, reward, done, info = env.step(env.action_space.sample())
    assert obs["image"].shape == (3, SIZE, SIZE)
    assert set(info) >= {"success", "goal_distance", "state", "terminated", "truncated"}
    assert info["state"].shape == (env.state_dim,)
    assert reward == float(info["success"])
    assert np.all(info["action"] >= env.action_space.low - 1e-6)


def test_state_round_trips_and_render_state_leaves_the_env_alone(env):
    env.reset(seed=2)
    for _ in range(3):
        env.step(env.action_space.sample())
    state = env.get_state()
    frame = env.render()
    other = env.goal_state
    env.render_state(other)
    np.testing.assert_allclose(env.get_state(), state)
    np.testing.assert_array_equal(env.render(), frame)
    env.set_state(other)
    env.set_state(state)
    np.testing.assert_allclose(env.get_state(), state, atol=1e-9)


def test_starting_at_the_goal_is_a_success(env):
    env.reset(seed=3)
    goal = env.goal_state
    env.reset(options={"state": goal, "goal_state": goal})
    assert env.is_success(env.get_state(), goal)
    assert env.goal_distance(env.get_state(), goal) == pytest.approx(0.0, abs=1e-6)


def test_options_set_the_start_and_goal(env):
    env.reset(seed=4)
    a = env.goal_state
    env.reset(seed=5)
    b = env.get_state()
    env.reset(options={"state": b, "goal_state": a})
    np.testing.assert_allclose(env.get_state(), b, atol=1e-9)
    np.testing.assert_allclose(env.goal_state, a)
    expected = env.render_state(a).transpose(2, 0, 1)
    np.testing.assert_array_equal(env.goal_observation()["image"], expected)


def test_seeded_resets_repeat(env):
    first = env.reset(seed=7)["image"]
    goal = env.goal_state
    second = env.reset(seed=7)["image"]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(env.goal_state, goal)


def test_wrong_state_size_is_rejected(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="states have"):
        env.set_state(np.zeros(env.state_dim + 1))


def test_time_limit_and_terminate_on_success():
    env = make_goal_env("two_room", size=SIZE, max_episode_steps=3, seed=0)
    env.reset(options={"state": [10.0, 10.0], "goal_state": [20.0, 50.0]})
    dones = [env.step([0.0, 0.0])[2] for _ in range(3)]
    assert dones == [False, False, True]
    env = make_goal_env("two_room", size=SIZE, terminate_on_success=True, seed=0)
    env.reset(options={"state": [10.0, 10.0], "goal_state": [11.0, 10.0]})
    _, reward, done, info = env.step([1.0, 0.0])
    assert done and info["terminated"] and reward == 1.0


def test_names_and_make_env_backends():
    assert goal_env_name("Two-Room") == "two_room"
    assert goal_env_name("push-t") == "pusht"
    assert goal_env_name("ogbench_cube") == "cube"
    with pytest.raises(ValueError, match="Unknown goal environment"):
        goal_env_name("maze")
    from simulusion.envs import TwoRoomEnv, make_env

    assert isinstance(make_env("two_room", backend="two_room", size=SIZE), TwoRoomEnv)


# ------------------------------------------------------------------ Two-Room


def test_two_room_walls_stop_the_agent_and_the_door_lets_it_through():
    env = make_goal_env("two_room", size=SIZE)
    env.reset(options={"state": [10.0, 10.0], "goal_state": [50.0, 10.0]})
    for _ in range(30):
        env.step([2.45, 0.0])
    x, y = env.get_state()
    assert x == pytest.approx(31.0, abs=1e-3) and y == pytest.approx(10.0)
    env.reset(options={"state": [10.0, 32.0], "goal_state": [50.0, 32.0]})
    for _ in range(30):
        env.step([2.45, 0.0])
    assert env.get_state()[0] > 50.0
    assert env.room_of(env.get_state()) == 1


def test_two_room_step_norm_is_capped():
    env = make_goal_env("two_room", size=SIZE)
    env.reset(options={"state": [16.0, 32.0], "goal_state": [50.0, 32.0]})
    env.step([2.45, 2.45])
    moved = np.linalg.norm(env.get_state() - np.array([16.0, 32.0]))
    assert moved == pytest.approx(2.45)


def test_two_room_default_goal_is_in_the_other_room():
    env = make_goal_env("two_room", size=SIZE, seed=0)
    for seed in range(5):
        env.reset(seed=seed)
        assert env.room_of(env.get_state()) != env.room_of(env.goal_state)
        assert env.is_free(env.get_state()) and env.is_free(env.goal_state)


# -------------------------------------------------------------------- Push-T


def test_pusht_pusher_moves_the_block_and_walls_hold_it():
    env = _make("pusht")
    start = np.array([256, 450, 256, 250, 0.0, 0, 0, 0, 0, 0])
    env.reset(options={"state": start, "goal_state": start})
    for _ in range(20):
        env.step([256.0, 60.0])
    block = env.get_state()[2:4]
    assert block[1] < 200.0  # pushed up the table
    assert block[1] > 0.0  # but not through the wall


def test_pusht_success_needs_block_pose_and_agent():
    env = _make("pusht")
    goal = np.array([100, 100, 256, 256, math.pi / 4, 0, 0, 0, 0, 0])
    env.reset(options={"state": goal, "goal_state": goal})
    near = goal.copy()
    near[2] += 10.0
    assert env.is_success(near, goal)
    turned = goal.copy()
    turned[4] += 0.5
    assert not env.is_success(turned, goal)
    away = goal.copy()
    away[0] += 100.0
    assert not env.is_success(away, goal)
    env.reset(seed=0)  # the classic goal: only the block is checked
    assert env.check_agent is False
    assert env.is_success(away, goal)


def test_pusht_relative_actions():
    env = _make("pusht", action_mode="relative")
    start = np.array([100, 100, 300, 300, 0.0, 0, 0, 0, 0, 0])
    env.reset(options={"state": start, "goal_state": start})
    xs = []
    for _ in range(5):
        env.step([20.0, 0.0])
        xs.append(env.get_state()[0])
    # The PD controller covers part of each offset per step: steady progress
    # to the right, less than the 100 pixels the offsets add up to.
    assert np.all(np.diff(xs) > 0) and 120.0 < xs[-1] < 200.0


# ---------------------------------------------------------------- Cube, Reacher


def test_cube_success_is_ogbenchs_four_centimetres():
    env = _make("cube")
    env.reset(seed=0)
    goal = env.goal_state
    moved = goal.copy()
    addr = env.cube_qpos[0]
    moved[addr] += 0.03
    assert env.is_success(moved, goal)
    moved[addr] += 0.02
    assert not env.is_success(moved, goal)


def test_reacher_success_wraps_angles():
    env = _make("reacher_goal")
    goal = np.array([math.pi - 0.05, 0.0, 0.0, 0.0])
    state = np.array([-math.pi + 0.05, 0.1, 0.0, 0.0])
    assert env.goal_distance(state, goal) == pytest.approx(0.1)
    assert env.is_success(state, goal)
    assert not env.is_success(np.array([0.0, 0.0, 0.0, 0.0]), goal)
