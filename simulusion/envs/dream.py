"""Train agents inside any world model: the model as a Gymnasium environment.

:class:`DreamEnv` turns a :class:`~simulusion.world_model.WorldModel` into a
``gymnasium.Env``, and :class:`DreamVectorEnv` into a ``gymnasium.vector``
environment that steps many dreams in one batched model call. Any library that
trains on Gymnasium environments (Stable-Baselines3, CleanRL, TorchRL) then
trains inside the model:

.. code-block:: python

    from simulusion.envs import DreamEnv, DreamVectorEnv

    env = DreamEnv("runs/iris-wm_.../world_model", starts="data/pong",
                   max_episode_steps=200)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    envs = DreamVectorEnv(model, starts="data/pong", num_envs=64,
                          max_episode_steps=200)

**Starts.** A dream begins from real frames: each reset picks a random clip
from ``starts`` (a folder of episodes in Simulusion's format, or episodes in
memory) and encodes it into a model state. The clip holds ``context`` frames,
by default the model's whole context window
(:attr:`~simulusion.world_model.WorldModelSpec.max_context`, or
``context_frames`` for recurrent models), and the actions between them.

**Observations** are the model's decoded frames, ``uint8`` ``(H, W, C)``, at
the model's resolution: what the agent sees is what the model imagines, from
the first frame on.

**Rewards and episode ends** come from the model's heads. A model without a
reward head (Genie) needs ``reward_fn``, which also replaces the head of a
model that has one, to train a new task in the same dream. Episodes end when
the model predicts it, or are cut off after ``max_episode_steps``.

**Randomness.** A stochastic model draws from PyTorch's RNG. Each environment
keeps its own RNG state and uses it only around its own model calls, so
``reset(seed=...)`` makes a dream repeatable without touching the global RNG
the training code uses.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.vector import AutoresetMode, VectorEnv

from simulusion.data.episode import Episode
from simulusion.data.folder import EpisodeFolder
from simulusion.world_model.protocol import State, WorldModel

#: Where dreams start: a folder of episodes, or episodes in memory.
Starts = Union[str, Path, EpisodeFolder, Episode, Sequence[Episode]]

#: ``reward_fn(frames, actions) -> rewards``: frames ``(B, H, W, C)`` uint8
#: after the step, actions ``(B,)`` or ``(B, A)``, rewards ``(B,)``.
RewardFn = Callable[[np.ndarray, np.ndarray], Any]


class _Clips:
    """Random clips of ``context`` frames from a set of episodes.

    Clips are drawn uniformly over every position where one fits, so long
    episodes contribute more starts than short ones. Loaded episodes are kept
    in a small cache, since a folder may not fit in memory.
    """

    def __init__(self, starts: Starts, context: int, cache: int = 32) -> None:
        if isinstance(starts, (str, Path)):
            starts = EpisodeFolder(starts)
        if isinstance(starts, Episode):
            starts = [starts]
        self.source = starts
        self.context = context
        lengths = (
            starts.lengths if isinstance(starts, EpisodeFolder) else map(len, starts)
        )
        self.positions = np.array(
            [max(n - context + 1, 0) for n in lengths], dtype=np.int64
        )
        if self.positions.sum() == 0:
            raise ValueError(
                f"No episode in starts has the {context} frames a dream starts "
                "from; pass longer episodes or a smaller context."
            )
        self.fps = getattr(getattr(starts, "meta", None), "fps", None)
        self._cache: OrderedDict[int, Episode] = OrderedDict()
        self._cache_size = cache

    def _episode(self, index: int) -> Episode:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        episode = self.source[index]
        self._cache[index] = episode
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return episode

    def sample(
        self, rng: np.random.Generator, count: int
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[tuple[int, int]]]:
        """``count`` clips: frames ``(B, T, H, W, C)``, actions, and ``(episode, start)``."""
        flat = rng.integers(int(self.positions.sum()), size=count)
        bounds = np.cumsum(self.positions)
        frames, actions, picks = [], [], []
        for position in flat:
            index = int(np.searchsorted(bounds, position, side="right"))
            start = int(position - (bounds[index] - self.positions[index]))
            clip = self._episode(index).slice(start, start + self.context)
            frames.append(clip.frames)
            actions.append(clip.actions)
            picks.append((index, start))
        has_actions = all(a is not None for a in actions) and self.context > 1
        return (
            np.stack(frames),
            np.stack(actions) if has_actions else None,
            picks,
        )


class _Dreams:
    """What both environments share: the model, its spaces, starts and RNG."""

    def __init__(
        self,
        world_model: Any,
        starts: Starts,
        *,
        context: Optional[int],
        reward_fn: Optional[RewardFn],
        device: Optional[Union[str, torch.device]],
    ) -> None:
        if isinstance(world_model, (str, Path)):
            from simulusion.world_model import load_world_model

            world_model = load_world_model(world_model, device=device)
        else:
            from simulusion.world_model import as_world_model

            world_model = as_world_model(world_model)
        self.model: WorldModel = world_model
        spec = world_model.spec
        if not spec.has_reward and reward_fn is None:
            raise ValueError(
                f"The {spec.family} world model has no reward head, so a dream "
                "has no rewards to train on. Pass reward_fn(frames, actions); "
                "to play a dream without rewards, return zeros."
            )
        self.reward_fn = reward_fn
        frames = (
            context
            if context is not None
            else (spec.max_context or spec.context_frames)
        )
        if frames < spec.context_frames:
            raise ValueError(
                f"The {spec.family} world model needs at least "
                f"{spec.context_frames} context frames; got context={frames}."
            )
        self.clips = _Clips(starts, frames)

        channels, height, width = spec.obs_shape
        self.observation_space = spaces.Box(
            0, 255, shape=(height, width, channels), dtype=np.uint8
        )
        action = spec.action
        if action.kind == "discrete":
            self.action_space: gym.Space = spaces.Discrete(action.n)
        else:
            self.action_space = spaces.Box(
                -np.inf if action.low is None else action.low,
                np.inf if action.high is None else action.high,
                shape=(action.n,),
                dtype=np.float32,
            )

        device_ = torch.device(world_model.device)
        self._cuda = device_.type == "cuda"
        self._cuda_index: int = 0
        if self._cuda:
            self._cuda_index = (
                device_.index
                if device_.index is not None
                else torch.cuda.current_device()
            )
        self._rng_state: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None

    # ------------------------------------------------------------------ RNG

    def seed(self, seed: int) -> None:
        """Start this environment's own PyTorch RNG stream from ``seed``."""
        cpu = torch.Generator().manual_seed(seed).get_state()
        cuda = None
        if self._cuda:
            device = torch.device("cuda", self._cuda_index)
            cuda = torch.Generator(device=device).manual_seed(seed).get_state()
        self._rng_state = (cpu, cuda)

    @contextmanager
    def _own_rng(self) -> Iterator[None]:
        """Run model calls on this environment's RNG, leaving the global one as it was."""
        devices = [self._cuda_index] if self._cuda else []
        with torch.random.fork_rng(devices=devices):
            if self._rng_state is not None:
                cpu, cuda = self._rng_state
                torch.set_rng_state(cpu)
                if cuda is not None:
                    torch.cuda.set_rng_state(cuda, self._cuda_index)
            yield
            self._rng_state = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(self._cuda_index) if self._cuda else None,
            )

    @property
    def rng_state(self) -> Any:
        return self._rng_state

    @rng_state.setter
    def rng_state(self, value: Any) -> None:
        self._rng_state = value

    # ---------------------------------------------------------------- model

    def begin(
        self, rng: np.random.Generator, count: int
    ) -> tuple[State, np.ndarray, list[tuple[int, int]]]:
        """``count`` fresh dreams from random real clips: state, first frames, picks."""
        clips, actions, picks = self.clips.sample(rng, count)
        with self._own_rng():
            state = self.model.encode(clips, actions)
            frames = self.frames(state)
        return state, frames, picks

    def advance(
        self, state: State, actions: np.ndarray
    ) -> tuple[State, np.ndarray, np.ndarray, np.ndarray]:
        """One imagined step for every row: ``(state, frames, rewards, terminated)``."""
        with self._own_rng():
            out = self.model.step(state, actions)
            frames = self.frames(out.state)
        batch = frames.shape[0]
        if self.reward_fn is not None:
            rewards = np.asarray(self.reward_fn(frames, actions), dtype=np.float32)
            rewards = rewards.reshape(batch)
        else:
            rewards = _numpy(out.reward).astype(np.float32).reshape(batch)
        if out.terminated is not None:
            terminated = _numpy(out.terminated).astype(bool).reshape(batch)
        else:
            terminated = np.zeros(batch, dtype=bool)
        return out.state, frames, rewards, terminated

    def frames(self, state: State) -> np.ndarray:
        """The current frames of ``state``: ``(B, H, W, C)`` uint8."""
        images = self.model.decode(state)  # float (B, C, H, W) in [0, 1]
        images = (images * 255.0).round().to(torch.uint8)
        return images.permute(0, 2, 3, 1).cpu().numpy()


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _rows(state: State, rows: np.ndarray) -> State:
    index = torch.as_tensor(rows, device=_state_device(state))
    return {key: value[index] for key, value in state.items()}


def _state_device(state: State) -> torch.device:
    for value in state.values():
        return value.device
    return torch.device("cpu")


def _merge(template: State, parts: Sequence[tuple[np.ndarray, State]]) -> State:
    """A batch whose ``rows`` come from each part's state, in place of ``template``'s."""
    merged: State = {}
    for key, value in template.items():
        out = value.clone()
        for rows, part in parts:
            if key not in part or part[key].shape[1:] != value.shape[1:]:
                got = None if key not in part else tuple(part[key].shape[1:])
                raise ValueError(
                    f"State '{key}' is {got} per dream for some dreams and "
                    f"{tuple(value.shape[1:])} for others, so one dream cannot "
                    "reset while the rest run on. World models whose state "
                    "grows until it fills their context window (IRIS, Genie) "
                    "need every dream to start from a full window: use the "
                    "default context (the model's max_context) and starts "
                    "that have actions."
                )
            out[torch.as_tensor(rows, device=out.device)] = part[key]
        merged[key] = out
    return merged


class DreamEnv(gym.Env):
    """One dream: a world model as a ``gymnasium.Env``.

    See the module docstring for where dreams start, what the agent observes,
    and where rewards come from.

    Args:
        world_model: A :class:`~simulusion.world_model.WorldModel`, a trained
            model :func:`~simulusion.world_model.as_world_model` accepts, or
            a path :func:`~simulusion.world_model.load_world_model` reads.
        starts: Real episodes dreams start from: a folder in Simulusion's
            episode format, an :class:`~simulusion.data.EpisodeFolder`, or
            :class:`~simulusion.data.Episode` objects.
        context: Real frames each dream starts from (default: the model's
            context window).
        max_episode_steps: Imagined steps before an episode is truncated.
            Without it, an episode only ends when the model predicts so,
            which a model without a termination head never does.
        reward_fn: ``reward_fn(frames, actions) -> rewards`` on the frames
            after a step, ``(B, H, W, C)`` uint8 with ``B = 1`` here. Replaces
            the model's reward head; required for a model without one.
        device: Device to load ``world_model`` on, when it is a path.
        render_mode: ``"rgb_array"`` to render the current frame.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        world_model: Any,
        starts: Starts,
        *,
        context: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        reward_fn: Optional[RewardFn] = None,
        device: Optional[Union[str, torch.device]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self._dreams = _Dreams(
            world_model, starts, context=context, reward_fn=reward_fn, device=device
        )
        self.world_model = self._dreams.model
        self.observation_space = self._dreams.observation_space
        self.action_space = self._dreams.action_space
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        if self._dreams.clips.fps:
            self.metadata = {**self.metadata, "render_fps": self._dreams.clips.fps}
        self._state: Optional[State] = None
        self._frame: Optional[np.ndarray] = None
        self._elapsed = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._dreams.seed(int(self.np_random.integers(2**63 - 1)))
        self._state, frames, picks = self._dreams.begin(self.np_random, 1)
        self._frame = frame = frames[0]
        self._elapsed = 0
        episode, start = picks[0]
        return frame, {"start_episode": episode, "start_frame": start}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        actions = np.asarray(action)[None]
        self._state, frames, rewards, terminated = self._dreams.advance(
            self._state, actions
        )
        self._frame = frame = frames[0]
        self._elapsed += 1
        truncated = (
            self.max_episode_steps is not None
            and self._elapsed >= self.max_episode_steps
        )
        return (
            frame,
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated) and not bool(terminated[0]),
            {"elapsed_steps": self._elapsed},
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        return self._frame

    # ------------------------------------------------------------- branching

    def snapshot(self) -> dict[str, Any]:
        """The dream so far, to :meth:`restore` and replay exactly from here."""
        if self._state is None:
            raise RuntimeError("Call reset() before snapshot().")
        return {
            "state": {k: v.clone() for k, v in self._state.items()},
            "frame": self._frame.copy() if self._frame is not None else None,
            "elapsed": self._elapsed,
            "torch_rng": self._dreams.rng_state,
            "np_random": self.np_random.bit_generator.state,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Return to a :meth:`snapshot`; the dream continues as it did then."""
        self._state = {k: v.clone() for k, v in snapshot["state"].items()}
        self._frame = None if snapshot["frame"] is None else snapshot["frame"].copy()
        self._elapsed = snapshot["elapsed"]
        self._dreams.rng_state = snapshot["torch_rng"]
        self.np_random.bit_generator.state = snapshot["np_random"]


class DreamVectorEnv(VectorEnv):
    """Many dreams stepped together: a world model as a ``gymnasium.vector`` env.

    Every step runs the model once on the whole batch, which is where the
    speed of training in imagination comes from. Episodes reset
    automatically in Gymnasium's default ``NEXT_STEP`` mode: the step after a
    dream ends returns its new first frame (reward 0), and that step's action
    for it is ignored.

    Args:
        world_model: As for :class:`DreamEnv`.
        starts: As for :class:`DreamEnv`.
        num_envs: Dreams in the batch.
        context: As for :class:`DreamEnv`.
        max_episode_steps: As for :class:`DreamEnv`, per dream.
        reward_fn: As for :class:`DreamEnv`, called on the dreams that
            stepped.
        device: As for :class:`DreamEnv`.
        render_mode: ``"rgb_array"`` to render every dream's current frame.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        world_model: Any,
        starts: Starts,
        num_envs: int,
        *,
        context: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
        reward_fn: Optional[RewardFn] = None,
        device: Optional[Union[str, torch.device]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1.")
        self._dreams = _Dreams(
            world_model, starts, context=context, reward_fn=reward_fn, device=device
        )
        self.world_model = self._dreams.model
        self.num_envs = num_envs
        self.single_observation_space = self._dreams.observation_space
        self.single_action_space = self._dreams.action_space
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        if self._dreams.clips.fps:
            self.metadata = {**self.metadata, "render_fps": self._dreams.clips.fps}
        self._state: Optional[State] = None
        self._frames: Optional[np.ndarray] = None
        self._elapsed = np.zeros(num_envs, dtype=np.int64)
        self._autoreset = np.zeros(num_envs, dtype=bool)

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[Optional[int]]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(seed, list):
            # One stream drives the whole batch; its first seed seeds it.
            seed = seed[0] if seed else None
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        self._dreams.seed(int(self.np_random.integers(2**63 - 1)))
        self._state, frames, _ = self._dreams.begin(self.np_random, self.num_envs)
        self._frames = frames
        self._elapsed[:] = 0
        self._autoreset[:] = False
        return frames.copy(), {}

    def step(
        self, actions: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        actions = np.asarray(actions)
        resetting = self._autoreset.copy()
        running = ~resetting
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)

        if running.all():
            self._state, frames, rewards, terminated = self._dreams.advance(
                self._state, actions
            )
        else:
            parts: list[tuple[np.ndarray, State]] = []
            frames = np.empty(
                (self.num_envs, *self.single_observation_space.shape), np.uint8
            )
            run_rows = np.flatnonzero(running)
            reset_rows = np.flatnonzero(resetting)
            if run_rows.size:
                state, stepped, r, t = self._dreams.advance(
                    _rows(self._state, run_rows), actions[run_rows]
                )
                parts.append((run_rows, state))
                frames[run_rows], rewards[run_rows], terminated[run_rows] = (
                    stepped,
                    r,
                    t,
                )
            fresh, frames[reset_rows], _ = self._dreams.begin(
                self.np_random, reset_rows.size
            )
            parts.append((reset_rows, fresh))
            self._state = _merge(self._state, parts)
        self._frames = frames

        self._elapsed[running] += 1
        self._elapsed[resetting] = 0
        if self.max_episode_steps is not None:
            truncated = (self._elapsed >= self.max_episode_steps) & ~terminated
        else:
            truncated = np.zeros(self.num_envs, dtype=bool)
        self._autoreset = terminated | truncated
        return frames.copy(), rewards, terminated, truncated, {}

    def render(self) -> Optional[tuple[np.ndarray, ...]]:
        if self.render_mode != "rgb_array" or self._frames is None:
            return None
        return tuple(self._frames)


__all__ = ["DreamEnv", "DreamVectorEnv"]
