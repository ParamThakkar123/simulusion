"""Collection policies, episode collection and the goal-reaching protocol."""

from __future__ import annotations

import numpy as np
import pytest

from simulusion.benchmarks.goal_reaching import (
    GOAL_REACHING,
    GoalReachingSetup,
    RandomGoalAgent,
    evaluate_goal_reaching,
    run_goal_reaching,
    sample_goal_trials,
)
from simulusion.data.collect import collect_goal_episodes
from simulusion.data.episode import Episode
from simulusion.data.folder import EpisodeFolder
from simulusion.envs import goal_policies
from simulusion.envs.goal import make_goal_env

SIZE = 32


# ------------------------------------------------------------------ policies


def test_von_mises_steps_stay_within_the_norm_and_keep_a_heading():
    env = make_goal_env("two_room", size=SIZE, seed=0)
    env.reset(seed=0)
    policy = goal_policies.VonMisesWalk(seed=0)
    policy.reset(env)
    steps = np.stack([policy(env, None) for _ in range(200)])
    assert np.linalg.norm(steps, axis=1).max() <= env.max_step + 1e-6
    heading = np.arctan2(steps[:, 1], steps[:, 0])
    # Concentration 5 keeps directions near one heading.
    assert np.abs(np.mean(np.exp(1j * heading))) > 0.7


def test_door_seeker_crosses_the_door_and_ends_its_trip():
    env = make_goal_env("two_room", size=SIZE, seed=0, max_episode_steps=None)
    policy = goal_policies.DoorSeeker(seed=0)
    crossed = 0
    for seed in range(5):
        env.reset(seed=seed)
        policy.reset(env)
        start = env.room_of(env.get_state())
        for _ in range(400):
            env.step(policy(env, None))
            if policy.done:
                break
        assert policy.done
        crossed += env.room_of(env.get_state()) != start
    assert crossed == 5


def test_pose_seeker_actions_are_in_range():
    pytest.importorskip("dm_control")
    env = make_goal_env("reacher_goal", size=SIZE, seed=0)
    env.reset(seed=0)
    policy = goal_policies.PoseSeeker(seed=0)
    policy.reset(env)
    for _ in range(5):
        action = policy(env, None)
        assert np.all(np.abs(action) <= 1.0)
        env.step(action)


def test_block_pusher_moves_the_block():
    pytest.importorskip("pymunk")
    env = make_goal_env("pusht", size=SIZE, seed=0)
    policy = goal_policies.BlockPusher(seed=0)
    moved = []
    for seed in range(3):
        env.reset(seed=seed)
        policy.reset(env)
        start = env.get_state()[2:4]
        for _ in range(60):
            env.step(policy(env, None))
        moved.append(np.linalg.norm(env.get_state()[2:4] - start))
    assert max(moved) > 20.0


def test_cube_oracle_lifts_the_cube():
    pytest.importorskip("ogbench")
    env = make_goal_env("cube", size=SIZE, seed=0, max_episode_steps=None)
    env.reset(seed=0)
    policy = goal_policies.CubeOracle(seed=0)
    policy.reset(env)
    height = 0.0
    for _ in range(150):
        env.step(policy(env, None))
        height = max(height, env.cube_positions(env.get_state())[0, 2])
    assert height > 0.05


# ---------------------------------------------------------------- collection


def test_collect_writes_frames_actions_and_states(tmp_path):
    out = collect_goal_episodes(
        "two_room",
        tmp_path / "eps",
        episodes=3,
        length=12,
        policy="von_mises",
        env_kwargs={"size": SIZE},
    )
    folder = EpisodeFolder(out)
    assert len(folder) == 3
    ep = folder[0]
    assert ep.frames.shape == (12, SIZE, SIZE, 3)
    assert ep.actions.shape == (11, 2)
    assert ep.observations["state"].shape == (12, 2)
    assert folder.meta.action_kind == "continuous"
    assert folder.meta.observation_keys == ["state"]
    # The recorded states replay: stepping from one with its action gives the next.
    env = make_goal_env("two_room", size=SIZE)
    states = ep.observations["state"].astype(np.float64)
    env.reset(options={"state": states[0], "goal_state": states[-1]})
    env.step(ep.actions[0])
    np.testing.assert_allclose(env.get_state(), states[1], atol=1e-4)


def test_policies_that_finish_early_make_shorter_episodes(tmp_path):
    out = collect_goal_episodes(
        "two_room", tmp_path / "eps", episodes=3, length=400, env_kwargs={"size": SIZE}
    )
    lengths = EpisodeFolder(out).lengths
    assert max(lengths) < 400 and min(lengths) > 10


def test_unknown_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown policy"):
        collect_goal_episodes("two_room", tmp_path, episodes=1, length=3, policy="nope")


# --------------------------------------------------------------- the protocol


def _line_episodes(n=4, length=40):
    """Two-Room episodes moving right along y=10 (left room only)."""
    eps = []
    for i in range(n):
        xs = 4.0 + 0.5 * np.arange(length) + i
        states = np.stack([xs, np.full(length, 10.0)], 1)
        eps.append(
            Episode(
                frames=np.zeros((length, 4, 4, 3), np.uint8),
                actions=np.zeros((length - 1, 2), np.float32),
                observations={"state": states.astype(np.float32)},
            )
        )
    return eps


def test_trials_pair_states_the_offset_apart():
    eps = _line_episodes()
    trials = sample_goal_trials(eps, goal_offset=10, trials=20, seed=0)
    assert len(trials) == 20
    for t in trials:
        states = eps[t.episode].observations["state"]
        np.testing.assert_allclose(t.start_state, states[t.start])
        np.testing.assert_allclose(t.goal_state, states[t.start + 10])
    with pytest.raises(ValueError, match="longer than goal_offset"):
        sample_goal_trials(eps, goal_offset=40, trials=1)
    no_state = [Episode(frames=np.zeros((5, 4, 4, 3), np.uint8))]
    with pytest.raises(ValueError, match="observations"):
        sample_goal_trials(no_state, goal_offset=2, trials=1)


class _Greedy:
    """Reads the true goal: steps straight at it (an upper bound for the protocol)."""

    def __init__(self, env):
        self.env = env

    def reset(self, goal):
        assert goal["image"].shape == (3, SIZE, SIZE)
        self.goal = self.env.goal_state

    def act(self, observation):
        gap = self.goal - self.env.get_state()
        return gap / max(np.linalg.norm(gap), 1e-9) * min(2.45, np.linalg.norm(gap))


def test_evaluation_scores_an_agent_that_reaches_the_goal():
    env = make_goal_env("two_room", size=SIZE, max_episode_steps=None)
    trials = sample_goal_trials(_line_episodes(), goal_offset=10, trials=8, seed=0)
    result = evaluate_goal_reaching(env, _Greedy(env), trials, budget=20)
    assert result.success_rate == 1.0
    assert max(result.steps) <= 20
    still = RandomGoalAgent(env.action_space, seed=0)
    still.act = lambda observation: np.zeros(2, np.float32)
    assert evaluate_goal_reaching(env, still, trials, budget=5).success_rate == 0.0


def test_run_goal_reaching_on_a_collected_folder(tmp_path):
    out = collect_goal_episodes(
        "two_room",
        tmp_path / "eps",
        episodes=4,
        length=60,
        policy="von_mises",
        env_kwargs={"size": SIZE},
    )
    env = make_goal_env("two_room", size=SIZE, max_episode_steps=None)
    setup = GoalReachingSetup("two_room", goal_offset=20, budget=40, trials=6)
    result = run_goal_reaching(setup, _Greedy(env), out, env=env)
    assert len(result.successes) == 6
    assert set(result.summary()) == {
        "success_rate",
        "trials",
        "mean_steps",
        "mean_final_distance",
    }


def test_protocols_follow_lewm():
    assert (
        GOAL_REACHING["two_room"].goal_offset,
        GOAL_REACHING["two_room"].budget,
    ) == (100, 150)
    for name in ("pusht", "cube", "reacher_goal"):
        assert (GOAL_REACHING[name].goal_offset, GOAL_REACHING[name].budget) == (25, 50)


# ----------------------------------------------------------------- CLI, install


def test_cli_collect(tmp_path):
    from click.testing import CliRunner

    from simulusion.cli import app

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "collect",
            "two_room",
            str(tmp_path / "d"),
            "--episodes",
            "2",
            "--length",
            "5",
            "--size",
            "16",
            "--policy",
            "random",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 episodes" in result.output


def test_install_dmc_adds_ogbench_without_deps():
    from simulusion.install_dmc import OGBENCH, install_steps

    steps = install_steps(ogbench=True)
    assert steps[-1][-2:] == ["--no-deps", OGBENCH]
    assert all(OGBENCH not in step for step in install_steps())
