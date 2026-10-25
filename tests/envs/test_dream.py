"""DreamEnv / DreamVectorEnv: any WorldModel as a Gymnasium environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from simulusion.data import Episode, EpisodeWriter  # noqa: E402
from simulusion.envs import DreamEnv, DreamVectorEnv  # noqa: E402
from simulusion.world_model.protocol import (  # noqa: E402
    ActionSpec,
    Transition,
    WorldModelBase,
    WorldModelSpec,
)

LEVEL = 20  # pixel value per unit of position


class _Line(WorldModelBase):
    """A dot on a line: action 0 moves left, 1 stays, 2 moves right.

    Its frames are flat grey at ``position * LEVEL``. The reward is the new
    position and the episode ends at position 5. ``noise`` adds a random
    drift, drawn from PyTorch's RNG like a stochastic world model's samples.
    """

    family = "toy"

    def __init__(self, noise: float = 0.0, *, reward: bool = True) -> None:
        self.device = torch.device("cpu")
        self.noise = noise
        self.spec = WorldModelSpec(
            family="toy",
            obs_shape=(3, 8, 8),
            action=ActionSpec("discrete", 3),
            has_reward=reward,
            has_termination=True,
            stochastic=noise > 0,
        )

    def _encode(self, frames, actions):
        return {"pos": frames[:, -1].mean(dim=(1, 2, 3)) * 255.0 / LEVEL}

    def _step(self, state, action):
        pos = state["pos"] + (action.float() - 1.0)
        if self.noise:
            pos = pos + self.noise * torch.rand_like(pos)
        return Transition(
            state={"pos": pos},
            reward=pos.clone() if self.spec.has_reward else None,
            terminated=pos >= 5.0,
        )

    def _decode(self, state):
        level = state["pos"] * LEVEL / 255.0
        return level.view(-1, 1, 1, 1).expand(-1, 3, 8, 8)

    def _save_weights(self, directory: Path):
        return {}

    @classmethod
    def _load_weights(cls, path, manifest, device, options):
        return cls()


class _Window(_Line):
    """Like :class:`_Line`, but its state keeps the last frames it saw, up to
    three, the way IRIS and Genie keep a growing window of tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.spec = WorldModelSpec(
            family="toy",
            obs_shape=(3, 8, 8),
            action=ActionSpec("discrete", 3),
            context_frames=1,
            max_context=3,
            has_termination=True,
        )

    def _encode(self, frames, actions):
        frames = frames if actions is not None else frames[:, -1:]
        return {"pos": frames[:, -3:].mean(dim=(2, 3, 4)) * 255.0 / LEVEL}

    def _step(self, state, action):
        pos = state["pos"][:, -1] + (action.float() - 1.0)
        window = torch.cat([state["pos"], pos[:, None]], dim=1)[:, -3:]
        return Transition(state={"pos": window}, reward=pos, terminated=pos >= 5.0)

    def _decode(self, state):
        return super()._decode({"pos": state["pos"][:, -1]})


def _episodes(count: int = 3, length: int = 6, *, actions: bool = True) -> list:
    """Episode ``i`` sits at position ``i`` throughout."""
    return [
        Episode(
            frames=np.full((length, 8, 8, 3), i * LEVEL, np.uint8),
            actions=np.ones(length - 1, np.int64) if actions else None,
        )
        for i in range(count)
    ]


def _position(frame: np.ndarray) -> float:
    return float(frame.mean()) / LEVEL


# ------------------------------------------------------------------ DreamEnv


def test_dream_env_passes_gymnasium_checks() -> None:
    env = DreamEnv(_Line(), _episodes(), max_episode_steps=10)
    assert env.observation_space == gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)
    assert env.action_space == gym.spaces.Discrete(3)
    check_env(env, skip_render_check=True)


def test_dreams_start_from_real_clips_and_follow_the_model() -> None:
    env = DreamEnv(_Line(), _episodes(), max_episode_steps=10)
    obs, info = env.reset(seed=0)
    start = info["start_episode"]
    assert _position(obs) == pytest.approx(start)
    assert 0 <= info["start_frame"] <= 5

    obs, reward, terminated, truncated, info = env.step(2)
    assert _position(obs) == pytest.approx(start + 1, abs=0.05)
    assert reward == pytest.approx(start + 1)
    assert not truncated and info["elapsed_steps"] == 1
    for _ in range(5):
        obs, reward, terminated, truncated, _ = env.step(2)
        if terminated:
            break
    assert terminated and reward >= 5.0 and not truncated


def test_episodes_are_cut_off_after_max_episode_steps() -> None:
    env = DreamEnv(_Line(), _episodes(), max_episode_steps=3)
    env.reset(seed=0)
    results = [env.step(1) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]
    assert not any(r[2] for r in results)


def test_starts_can_be_a_folder_of_episodes(tmp_path: Path) -> None:
    with EpisodeWriter(tmp_path / "eps") as writer:
        for episode in _episodes():
            writer.write(episode)
    env = DreamEnv(_Line(), tmp_path / "eps")
    obs, info = env.reset(seed=1)
    assert _position(obs) == pytest.approx(info["start_episode"])


def test_a_model_without_a_reward_head_needs_reward_fn() -> None:
    with pytest.raises(ValueError, match="no reward head"):
        DreamEnv(_Line(reward=False), _episodes())

    def right(frames, actions):
        assert frames.shape == (1, 8, 8, 3) and frames.dtype == np.uint8
        return (actions == 2).astype(np.float32)

    env = DreamEnv(_Line(reward=False), _episodes(), reward_fn=right)
    env.reset(seed=0)
    assert env.step(2)[1] == 1.0 and env.step(0)[1] == 0.0


def test_reward_fn_replaces_the_reward_head() -> None:
    env = DreamEnv(_Line(), _episodes(), reward_fn=lambda f, a: -np.ones(len(a)))
    env.reset(seed=0)
    assert env.step(2)[1] == -1.0


def test_context_and_starts_are_checked() -> None:
    with pytest.raises(ValueError, match="needs at least 2 context frames"):

        class _Two(_Line):
            def __init__(self) -> None:
                super().__init__()
                self.spec = WorldModelSpec(
                    family="toy",
                    obs_shape=(3, 8, 8),
                    action=ActionSpec("discrete", 3),
                    context_frames=2,
                )

        DreamEnv(_Two(), _episodes(), context=1)
    with pytest.raises(ValueError, match="No episode in starts has the 3 frames"):
        DreamEnv(_Line(), _episodes(length=2), context=3)


def test_a_seed_makes_a_stochastic_dream_repeatable() -> None:
    def rollout(seed: int) -> list[float]:
        env = DreamEnv(_Line(noise=0.5), _episodes())
        env.reset(seed=seed)
        return [env.step(1)[1] for _ in range(5)]

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_dreams_leave_the_global_rng_alone() -> None:
    env = DreamEnv(_Line(noise=0.5), _episodes())
    torch.manual_seed(123)
    before = torch.get_rng_state()
    env.reset(seed=0)
    for _ in range(3):
        env.step(1)
    assert torch.equal(torch.get_rng_state(), before)


def test_restoring_a_snapshot_replays_a_stochastic_dream_exactly() -> None:
    env = DreamEnv(_Line(noise=0.5), _episodes(), max_episode_steps=50)
    env.reset(seed=0)
    env.step(1)
    snap = env.snapshot()
    first = [env.step(a)[:2] for a in (0, 1, 2)]
    env.restore(snap)
    again = [env.step(a)[:2] for a in (0, 1, 2)]
    for (obs_a, r_a), (obs_b, r_b) in zip(first, again):
        assert np.array_equal(obs_a, obs_b) and r_a == r_b


def test_render_returns_the_current_frame() -> None:
    env = DreamEnv(_Line(), _episodes(), render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    assert np.array_equal(env.render(), obs)
    obs = env.step(2)[0]
    assert np.array_equal(env.render(), obs)


# ------------------------------------------------------------ DreamVectorEnv


def test_vector_dreams_step_as_a_batch() -> None:
    envs = DreamVectorEnv(_Line(), _episodes(), num_envs=4, max_episode_steps=20)
    obs, _ = envs.reset(seed=0)
    assert obs.shape == (4, 8, 8, 3) and obs.dtype == np.uint8
    assert envs.single_action_space == gym.spaces.Discrete(3)
    start = np.array([_position(o) for o in obs])

    obs, rewards, terminated, truncated, _ = envs.step(np.array([0, 1, 2, 1]))
    assert np.allclose(rewards, start + np.array([-1, 0, 1, 0]))
    assert np.allclose([_position(o) for o in obs], np.maximum(rewards, 0), atol=0.05)
    assert not terminated.any() and not truncated.any()


def test_vector_dreams_reset_on_the_next_step() -> None:
    envs = DreamVectorEnv(_Line(), _episodes(), num_envs=2, max_episode_steps=2)
    envs.reset(seed=0)
    envs.step(np.array([1, 1]))
    _, _, terminated, truncated, _ = envs.step(np.array([1, 1]))
    assert truncated.all() and not terminated.any()

    obs, rewards, terminated, truncated, _ = envs.step(np.array([2, 2]))
    assert np.array_equal(rewards, [0.0, 0.0])
    assert not terminated.any() and not truncated.any()
    assert all(_position(o) == pytest.approx(round(_position(o))) for o in obs)

    _, rewards, _, _, _ = envs.step(np.array([2, 1]))
    assert rewards[0] == pytest.approx(_position(obs[0]) + 1)


def test_one_dream_resets_while_the_others_run_on() -> None:
    # Episode 4 starts at position 4; one step right ends it.
    starts = [_episodes(count=5)[4], _episodes(count=1)[0]]
    envs = DreamVectorEnv(_Line(), starts, num_envs=8)
    obs, _ = envs.reset(seed=0)
    positions = np.array([_position(o) for o in obs])
    _, _, terminated, _, _ = envs.step(np.full(8, 2))
    assert np.array_equal(terminated, positions == 4)
    assert terminated.any() and not terminated.all()

    _, rewards, _, _, _ = envs.step(np.full(8, 2))
    assert np.allclose(rewards[terminated], 0.0)
    assert np.allclose(rewards[~terminated], positions[~terminated] + 2)


def test_growing_states_reset_row_by_row_from_a_full_window() -> None:
    envs = DreamVectorEnv(
        _Window(), _episodes(count=5), num_envs=6, max_episode_steps=3
    )
    envs.reset(seed=0)
    for _ in range(10):  # several staggered resets
        envs.step(envs.action_space.sample())

    short = DreamVectorEnv(_Window(), _episodes(count=5), num_envs=6, context=1)
    short.reset(seed=0)
    with pytest.raises(ValueError, match="start from a full window"):
        for _ in range(10):
            short.step(np.full(6, 2))


def test_a_model_without_a_termination_head_never_terminates() -> None:
    class _Endless(_Line):
        def _step(self, state, action):
            out = super()._step(state, action)
            return Transition(state=out.state, reward=out.reward)

    envs = DreamVectorEnv(_Endless(), _episodes(), num_envs=2)
    envs.reset(seed=0)
    for _ in range(8):
        _, _, terminated, truncated, _ = envs.step(np.full(2, 2))
        assert not terminated.any() and not truncated.any()


# ---------------------------------------------------- real world models


@pytest.mark.timeout(180)
def test_a_dreamer_world_model_from_disk_is_an_environment(tmp_path: Path) -> None:
    from simulusion.training.run import run

    from tests.training.tasks.conftest import write_episodes

    data = write_episodes(
        tmp_path / "eps", count=3, length=8, size=64, actions=("continuous", 2)
    )
    result = run(
        overrides=[
            "trainer.device=cpu",
            f"trainer.out_dir={tmp_path.as_posix()}/runs",
            "trainer.console=false",
            "trainer.max_steps=2",
            f"data.path={data.as_posix()}",
            "data.batch_size=2",
            "model.train_seq_len=4",
            "model.obs_embed_size=32",
            "model.num_units=16",
            "model.deter_size=16",
            "model.stoch_size=4",
            "model.use_amp=false",
        ],
        task="dreamer-wm",
    )
    env = DreamEnv(result.world_model_path, data, device="cpu", max_episode_steps=4)
    assert env.action_space == gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    check_env(env, skip_render_check=True)

    envs = DreamVectorEnv(
        result.world_model_path, data, num_envs=3, device="cpu", max_episode_steps=2
    )
    obs, _ = envs.reset(seed=0)
    for _ in range(5):
        obs, rewards, _, _, _ = envs.step(envs.action_space.sample())
        assert obs.shape == (3, 64, 64, 3) and np.isfinite(rewards).all()


@pytest.mark.timeout(180)
def test_iris_dreams_keep_one_state_shape_so_any_can_reset(tmp_path: Path) -> None:
    """IRIS's token window would grow from a one-frame start; starting from a
    full window keeps every dream's state the same shape, which is what lets
    one dream reset while the others run on."""
    from simulusion.training.run import run

    from tests.training.tasks.conftest import write_episodes
    from tests.training.tasks.test_iris_tasks import TINY

    data = write_episodes(tmp_path / "eps", count=3, length=8, size=64, actions=3)
    result = run(
        overrides=[
            "trainer.device=cpu",
            f"trainer.out_dir={tmp_path.as_posix()}/runs",
            "trainer.console=false",
            "trainer.max_steps=2",
            f"data.path={data.as_posix()}",
            "data.batch_size=2",
            *(f"model.{k}={v}" for k, v in TINY.items()),
        ],
        task="iris-wm",
    )
    envs = DreamVectorEnv(
        result.world_model_path, data, num_envs=3, device="cpu", max_episode_steps=2
    )
    assert envs.single_action_space == gym.spaces.Discrete(3)
    envs.reset(seed=0)
    shapes = {k: v.shape for k, v in envs._state.items()}
    assert shapes["tokens"][1] == envs.world_model.spec.max_context
    for _ in range(5):  # past the first round of resets
        obs, rewards, _, _, _ = envs.step(envs.action_space.sample())
        assert obs.shape == (3, 64, 64, 3) and np.isfinite(rewards).all()
        assert {k: v.shape for k, v in envs._state.items()} == shapes
