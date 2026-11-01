"""Regression tests for IRIS evaluation.

Both failure modes here are silent: training keeps running and reporting
numbers. One holds every evaluated frame in memory until the process is killed
by the OS; the other quietly corrupts what collection writes into the replay
buffer.
"""

import gc
import weakref

import numpy as np
import pytest
import torch

from simulusion.configs.iris_config import IRISConfig
from simulusion.training.train_iris import IRISTrainer


class FakeEnv:
    """Minimal Gymnasium-shaped env returning 210x160x3 frames, like Atari."""

    def __init__(self, episode_length: int = 4, tracker: list | None = None) -> None:
        self.episode_length = episode_length
        self.resets = 0
        self.steps = 0
        self._t = 0
        # Weak references to every frame handed out, so a test can tell whether
        # anything still holds one once evaluation returns.
        self._tracker = tracker

    def _frame(self) -> np.ndarray:
        frame = np.full((210, 160, 3), self._t % 256, dtype=np.uint8)
        if self._tracker is not None:
            self._tracker.append(weakref.ref(frame))
            # How many frames handed out so far are still referenced. Measured
            # here, mid-evaluation, because the peak is what exhausts memory;
            # by the time evaluate() returns its discarded list is collectable.
            alive = sum(ref() is not None for ref in self._tracker)
            self.max_alive = max(getattr(self, "max_alive", 0), alive)
        return frame

    def reset(self, *args, **kwargs):
        self.resets += 1
        self._t = 0
        return self._frame(), {}

    def step(self, action):
        self.steps += 1
        self._t += 1
        done = self._t >= self.episode_length
        return self._frame(), 1.0, done, False, {}


class FakeAgent:
    """Stands in for IRISAgent: the parts evaluate() touches, nothing more."""

    def __init__(self) -> None:
        self.encoder = torch.nn.Identity()

    def reconstruct(self, frame_tensor):
        return frame_tensor

    def act(self, *args, **kwargs):
        return torch.tensor(0), None


def make_trainer(env_factory=None, env=None, tracker=None) -> IRISTrainer:
    """Build a trainer without its models, environment or replay buffer.

    ``IRISTrainer.__init__`` builds an Atari env and the full agent, neither of
    which these tests need.
    """
    trainer = IRISTrainer.__new__(IRISTrainer)
    trainer.config = IRISConfig()
    trainer.device = torch.device("cpu")
    trainer.agent = FakeAgent()
    trainer._env_factory = env_factory
    trainer._eval_env = None
    trainer.env = env if env is not None else FakeEnv(tracker=tracker)
    trainer._collect_obs = None
    trainer._collect_hidden = None
    trainer._collect_return = 0.0
    trainer._last_episode_return = 0.0
    trainer.env_steps = 0
    return trainer


pytest.importorskip("cv2", reason="preprocess_frame needs opencv")


class TestEvaluationMemory:
    def test_frames_are_not_retained_without_render(self):
        """render=False must hold no frames.

        A raw Atari frame is ~100 KB and eval_episodes defaults to 100, so
        collecting them regardless of render could reach tens of GB before
        evaluate() returned -- the process is killed by the OS with no
        traceback, which in a notebook looks like the kernel restarting.
        """
        tracker: list = []
        trainer = make_trainer(tracker=tracker)
        env = trainer.env

        trainer.evaluate(num_episodes=3, render=False)

        gc.collect()
        assert len(tracker) >= 12, "the fake env handed out too few frames"
        # Only the frame in flight and the one before it stay reachable; the
        # count must not grow with the length of the evaluation.
        assert env.max_alive <= 3, (
            f"{env.max_alive} frames held at once out of {len(tracker)}; "
            "evaluate() is accumulating frames without render"
        )
        assert not [ref for ref in tracker if ref() is not None]

    def test_render_still_returns_videos(self):
        tracker: list = []
        trainer = make_trainer(tracker=tracker)

        returns, videos, latents = trainer.evaluate(num_episodes=2, render=True)

        assert len(returns) == 2
        assert len(videos) == 2
        assert all(len(frames) > 0 for frames in videos)
        assert isinstance(latents, np.ndarray)


class TestEvaluationEnvIsolation:
    def test_evaluation_uses_its_own_env(self):
        """Evaluation must not reset the env collection is midway through.

        collect_experience keeps a partial episode alive across calls: its last
        observation, the policy's recurrent state and the running return. When
        evaluation shared the env, it reset that env underneath the state, and
        the next collection step paired the stale pre-eval observation with the
        post-eval env -- writing a transition into the replay buffer that never
        happened.
        """
        collection_env = FakeEnv()
        trainer = make_trainer(env_factory=lambda: FakeEnv(), env=collection_env)
        sentinel = np.zeros((3, 64, 64), dtype=np.uint8)
        trainer._collect_obs = sentinel
        trainer._collect_return = 7.0

        trainer.evaluate(num_episodes=2)

        assert collection_env.resets == 0, "evaluation reset the collection env"
        assert collection_env.steps == 0, "evaluation stepped the collection env"
        assert trainer._collect_obs is sentinel
        assert trainer._collect_return == 7.0

    def test_eval_env_is_reused_across_calls(self):
        built = []

        def factory():
            env = FakeEnv()
            built.append(env)
            return env

        trainer = make_trainer(env_factory=factory)

        trainer.evaluate(num_episodes=1)
        trainer.evaluate(num_episodes=1)

        assert len(built) == 1, "a new env was built for every evaluation"

    def test_shared_env_drops_the_collection_episode(self):
        """With a caller-supplied env there is no second env to build.

        Evaluation borrows it, so the in-flight collection episode no longer
        matches the env and must be abandoned rather than resumed.
        """
        trainer = make_trainer(env_factory=None)
        trainer._collect_obs = np.zeros((3, 64, 64), dtype=np.uint8)
        trainer._collect_hidden = (torch.zeros(1), torch.zeros(1))
        trainer._collect_return = 3.0

        trainer.evaluate(num_episodes=1)

        assert trainer._collect_obs is None
        assert trainer._collect_hidden is None
        assert trainer._collect_return == 0.0
