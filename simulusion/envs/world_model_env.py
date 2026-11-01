"""Gymnasium-compatible environment wrapper for learned world models.

The wrapper intentionally supports a small adapter surface instead of assuming a
single Simulusion model architecture. Users can pass explicit reset/transition
callables, or rely on common model method names such as ``step``/``predict``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.util import find_spec
from inspect import signature
from typing import Any

import gymnasium as gym
import numpy as np

if find_spec("torch") is not None:
    import torch
else:
    torch = None  # type: ignore[assignment]


TransitionFn = Callable[..., Any]
ResetFn = Callable[..., Any]
RewardFn = Callable[..., Any]
TerminalFn = Callable[..., Any]
RenderFn = Callable[..., Any]
ActionTransformFn = Callable[..., Any]
_MISSING = object()


class WorldModelEnv(gym.Env):
    """Expose a trained world model through the Gymnasium ``Env`` API.

    ``WorldModelEnv`` keeps the current latent/model state and advances it with a
    transition callable or with a compatible method on ``world_model``. The
    wrapper returns Gymnasium-style ``(obs, info)`` from ``reset`` and
    ``(obs, reward, terminated, truncated, info)`` from ``step``, making learned
    model rollouts pluggable into RL libraries such as Stable-Baselines3,
    TorchRL, and CleanRL.

    Args:
        world_model: Trained model or lightweight adapter object used for
            simulated dynamics.
        observation_space: Gymnasium observation space emitted by the wrapper.
        action_space: Gymnasium action space accepted by the wrapper.
        initial_observation: Optional observation returned when no reset callable
            provides one. Defaults to ``observation_space.sample()``.
        initial_state: Optional latent/model state used at reset.
        reset_fn: Optional callable for resetting model state. Accepted return
            forms are ``obs``, ``(obs, info)``, ``(state, obs)``,
            ``(state, obs, info)``, or a mapping with ``state``/``observation``.
        transition_fn: Optional callable for one model step. If omitted, the
            wrapper tries common methods on ``world_model``: ``env_step``,
            ``step``, ``predict_step``, ``predict``, ``imagine_step``,
            ``transition``, then ``__call__``.
        reward_fn: Optional callable used when the transition output omits a
            reward.
        terminal_fn: Optional callable used when the transition output omits a
            termination flag.
        render_fn: Optional callable used by ``render``.
        action_transform_fn: Optional callable that converts library actions into
            the format expected by the world model.
        max_episode_steps: Optional time limit. Reaching it sets ``truncated``.
        render_mode: Optional Gymnasium render mode. ``rgb_array`` is supported
            by default when observations contain image-like data.
        device: Device used for tensor actions when ``torch_actions=True``.
        torch_actions: Convert actions to ``torch.Tensor`` before model calls.
        seed: Optional RNG seed for observation/action spaces and NumPy.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        world_model: Any,
        observation_space: gym.Space,
        action_space: gym.Space,
        *,
        initial_observation: Any | None = None,
        initial_state: Any | None = None,
        reset_fn: ResetFn | None = None,
        transition_fn: TransitionFn | None = None,
        reward_fn: RewardFn | None = None,
        terminal_fn: TerminalFn | None = None,
        render_fn: RenderFn | None = None,
        action_transform_fn: ActionTransformFn | None = None,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        device: Any | None = None,
        torch_actions: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.observation_space = observation_space
        self.action_space = action_space
        self.initial_observation = initial_observation
        self.initial_state = initial_state
        self.reset_fn = reset_fn
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.terminal_fn = terminal_fn
        if not isinstance(observation_space, gym.Space):
            raise TypeError("observation_space must be a gymnasium.Space instance.")
        if not isinstance(action_space, gym.Space):
            raise TypeError("action_space must be a gymnasium.Space instance.")

        self.render_fn = render_fn
        self.action_transform_fn = action_transform_fn
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.device = (
            torch.device(device) if torch is not None and device is not None else device
        )
        self.torch_actions = torch_actions

        self._state: Any = None
        self._last_observation: Any = None
        self._elapsed_steps = 0
        self._np_random = np.random.default_rng(seed)
        if seed is not None:
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

    @property
    def state(self) -> Any:
        """Current latent/model state tracked by the wrapper."""

        return self._state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the simulated rollout and return ``(observation, info)``."""

        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)

        self._elapsed_steps = 0
        result = self._call_reset(seed=seed, options=options or {})
        state, obs, info = self._parse_reset_result(result)
        self._state = state
        self._last_observation = obs
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Roll the learned model forward for one simulated environment step."""

        if self._last_observation is None and self._state is None:
            raise RuntimeError("Call reset() before step().")

        model_action = self._prepare_action(action)
        result = self._call_transition(model_action)
        obs, reward, terminated, truncated, info, next_state = self._parse_step_result(
            result, model_action
        )

        self._elapsed_steps += 1
        if (
            self.max_episode_steps is not None
            and self._elapsed_steps >= self.max_episode_steps
        ):
            truncated = True

        self._state = next_state
        self._last_observation = obs
        info = dict(info)
        info.setdefault("model_state", next_state)
        info.setdefault("elapsed_steps", self._elapsed_steps)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        """Render the latest simulated observation or delegate to ``render_fn``."""

        if self.render_fn is not None:
            return self._call_user_fn(
                self.render_fn,
                self.world_model,
                self._state,
                self._last_observation,
            )
        if isinstance(self._last_observation, Mapping):
            for key in ("image", "pixels", "rgb"):
                if key in self._last_observation:
                    return self._to_numpy(self._last_observation[key])
        return self._to_numpy(self._last_observation)

    def close(self) -> None:
        """Close the wrapped world model if it exposes ``close``."""

        close = getattr(self.world_model, "close", None)
        if callable(close):
            close()

    def _call_reset(self, *, seed: int | None, options: dict[str, Any]) -> Any:
        if self.reset_fn is not None:
            return self._call_user_fn(self.reset_fn, self.world_model, seed, options)

        for name in ("reset_env", "reset_state", "reset"):
            method = getattr(self.world_model, name, None)
            if callable(method):
                return self._call_user_fn(method, seed, options)

        obs = self.initial_observation
        if obs is None:
            obs = self.observation_space.sample()
        return {"state": self.initial_state, "observation": obs, "info": {}}

    def _call_transition(self, action: Any) -> Any:
        if self.transition_fn is not None:
            return self._call_user_fn(
                self.transition_fn, self.world_model, self._state, action
            )

        for name in (
            "env_step",
            "step",
            "predict_step",
            "predict",
            "imagine_step",
            "transition",
        ):
            method = getattr(self.world_model, name, None)
            if callable(method):
                return self._call_model_transition(method, action)

        if callable(self.world_model):
            return self._call_model_transition(self.world_model, action)
        raise TypeError(
            "World model must define a transition_fn or a callable "
            "step/predict/transition method."
        )

    def _parse_reset_result(self, result: Any) -> tuple[Any, Any, dict[str, Any]]:
        if isinstance(result, Mapping):
            state = result.get("state", self.initial_state)
            obs = self._extract_observation(result, default=self.initial_observation)
            info = dict(result.get("info", {}))
        elif isinstance(result, tuple):
            if len(result) == 3:
                state, obs, info = result
                info = dict(info or {})
            elif len(result) == 2 and self._looks_like_reset_info(result):
                state = self.initial_state
                obs, info = result
                info = dict(info or {})
            elif len(result) == 2:
                state, obs = result
                info = {}
            else:
                state = self.initial_state
                obs = result[0] if result else self.initial_observation
                info = {}
        else:
            state = self.initial_state
            obs = result
            info = {}

        if obs is None:
            obs = self.observation_space.sample()
        return state, self._to_numpy(obs), info

    def _parse_step_result(
        self, result: Any, action: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any], Any]:
        if isinstance(result, Mapping):
            next_state = result.get("state", result.get("next_state", self._state))
            obs = self._extract_observation(result, default=next_state)
            reward = result.get("reward", None)
            terminated = result.get("terminated", result.get("done", _MISSING))
            truncated = result.get("truncated", False)
            info = dict(result.get("info", {}))
        elif isinstance(result, tuple):
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                info = dict(info or {})
                next_state = info.get("model_state", obs)
            elif len(result) == 4:
                obs, reward, done, info = result
                info = dict(info or {})
                terminated, truncated = done, False
                next_state = info.get("model_state", obs)
            elif len(result) == 6:
                next_state, obs, reward, terminated, truncated, info = result
            elif len(result) == 3:
                next_state, obs, reward = result
                terminated, truncated, info = _MISSING, False, {}
            else:
                next_state = result[0] if result else self._state
                obs = next_state
                reward, terminated, truncated, info = None, _MISSING, False, {}
        else:
            next_state = result
            obs = result
            reward, terminated, truncated, info = None, _MISSING, False, {}

        if reward is None:
            reward = self._predict_reward(next_state, obs, action)
        if terminated is _MISSING:
            terminated = self._predict_terminal(next_state, obs, action)
        return (
            self._to_numpy(obs),
            float(self._scalar(reward)),
            bool(terminated),
            bool(truncated),
            info,
            next_state,
        )

    def _predict_reward(self, state: Any, obs: Any, action: Any) -> float:
        if self.reward_fn is None:
            return 0.0
        return self._scalar(
            self._call_user_fn(self.reward_fn, self.world_model, state, obs, action)
        )

    def _predict_terminal(self, state: Any, obs: Any, action: Any) -> bool:
        if self.terminal_fn is None:
            return False
        return bool(
            self._call_user_fn(self.terminal_fn, self.world_model, state, obs, action)
        )

    def _prepare_action(self, action: Any) -> Any:
        if self.action_transform_fn is not None:
            return self._call_user_fn(
                self.action_transform_fn, self.world_model, action
            )
        if not self.torch_actions:
            return action
        if torch is None:
            return action
        if torch.is_tensor(action):
            tensor = (
                action.detach().to(self.device)
                if self.device is not None
                else action.detach()
            )
        else:
            tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        return tensor

    @staticmethod
    def _extract_observation(result: Mapping[str, Any], *, default: Any) -> Any:
        for key in (
            "observation",
            "obs",
            "image",
            "pixels",
            "next_observation",
            "next_obs",
        ):
            if key in result:
                return result[key]
        return default

    def _looks_like_reset_info(self, result: tuple[Any, ...]) -> bool:
        obs, info = result
        if not isinstance(info, Mapping):
            return False
        try:
            return bool(self.observation_space.contains(self._to_numpy(obs)))
        except Exception:
            return True

    @classmethod
    def _to_numpy(cls, value: Any) -> Any:
        if torch is not None and torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {key: cls._to_numpy(inner) for key, inner in value.items()}
        if isinstance(value, tuple):
            return tuple(cls._to_numpy(inner) for inner in value)
        if isinstance(value, list):
            return [cls._to_numpy(inner) for inner in value]
        return value

    @classmethod
    def _scalar(cls, value: Any) -> float:
        value = cls._to_numpy(value)
        return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])

    def _call_model_transition(self, fn: Callable[..., Any], action: Any) -> Any:
        try:
            params = signature(fn).parameters
        except (TypeError, ValueError):
            return fn(self._state, action)
        if any(param.kind == param.VAR_POSITIONAL for param in params.values()):
            return fn(self._state, action)
        positional = [
            param
            for param in params.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) <= 1:
            return fn(action)
        return fn(self._state, action)

    @staticmethod
    def _call_user_fn(fn: Callable[..., Any], *args: Any) -> Any:
        """Call ``fn`` with as many positional args as it declares.

        This lets adapter functions choose a concise signature, e.g.
        ``transition(state, action)`` or ``transition(model, state, action)``.
        """

        try:
            params = signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args)

        if any(param.kind == param.VAR_POSITIONAL for param in params.values()):
            return fn(*args)
        positional = [
            param
            for param in params.values()
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        return fn(*args[: len(positional)])


def make_world_model_env(world_model: Any, **kwargs: Any) -> WorldModelEnv:
    """Create a :class:`WorldModelEnv` from a trained model and spaces."""

    return WorldModelEnv(world_model, **kwargs)
