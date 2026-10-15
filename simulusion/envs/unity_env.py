from __future__ import annotations

from typing import Any, Optional

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._observations import add_optional_state_space
import gymnasium as gym
import numpy as np
from PIL import Image


def make_unity_mlagents_env(
    env_id: Optional[str] = None, **kwargs: Any
) -> UnityMLAgentsEnv:
    """Create a Unity ML-Agents environment wrapper.

    Factory function that instantiates a UnityMLAgentsEnv with the provided
    keyword arguments. Suitable for integrating Unity-based environments
    with Dreamer-style world model pipelines.

    Args:
        **kwargs: Keyword arguments passed to UnityMLAgentsEnv, including:
            - file_name (str): Path to the Unity environment binary.
            - behavior_name (str, optional): Name of the behavior to use.
            - seed (int): Random seed (default: 0).
            - size (tuple): Image size as (height, width) (default: (64, 64)).
            - worker_id (int): Worker ID for multi-environment setup (default: 0).
            - base_port (int): Base port for communication (default: 5005).
            - no_graphics (bool): Disable graphics rendering (default: True).
            - time_scale (float): Simulation time scale (default: 20.0).
            - quality_level (int): Graphics quality level (default: 1).
            - max_episode_steps (int): Max steps per episode (default: 1000).

    Returns:
        UnityMLAgentsEnv: A Gym-compatible wrapper for Unity environments.
    """
    # `env_id` is accepted for API compatibility with generic factory
    # callers that forward an env id as the first positional argument.
    return UnityMLAgentsEnv(**kwargs)


class UnityMLAgentsEnv:
    """Gym-like wrapper for Unity ML-Agents environments.

    Provides a unified interface for Unity-based environments, converting
    observations to image format compatible with pixel-based world models.

    Features:
        - Supports single-agent control with continuous action spaces.
        - Returns observations as dicts with required key ``"image"``.
        - Normalizes actions to [-1, 1] range.
        - Exposes non-visual sensors in ``info["vector_observation"]`` for debugging.

    Args:
        file_name (str): Path to the Unity environment binary.
        behavior_name (str, optional): Name of the behavior to use. If None,
            uses the first available behavior.
        seed (int): Random seed for environment (default: 0).
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        worker_id (int): Worker ID for multi-environment setup (default: 0).
        base_port (int): Base port for Unity environment communication (default: 5005).
        no_graphics (bool): Disable graphics rendering for faster simulation (default: True).
        time_scale (float): Simulation time scale multiplier (default: 20.0).
        quality_level (int): Graphics quality level 0-5 (default: 1).
        max_episode_steps (int): Maximum steps per episode (default: 1000).
        include_state (bool): Include a flattened non-visual ``"state"`` key
            in observations when the Unity behavior exposes non-image sensors.

    Attributes:
        observation_space: Dict space with required ``"image"`` key containing
            ``(3, H, W)`` uint8 frames.
        action_space: Box space with actions in [-1, 1] range.
        max_episode_steps: Maximum steps per episode.

    Raises:
        ValueError: If no behaviors found or action space is not continuous.
        RuntimeError: If no agents available after reset.
    """

    def __init__(
        self,
        file_name: str,
        behavior_name: str | None = None,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        worker_id: int = 0,
        base_port: int = 5005,
        no_graphics: bool = True,
        time_scale: float = 20.0,
        quality_level: int = 1,
        max_episode_steps: int = 1000,
        include_state: bool = False,
    ) -> None:
        from mlagents_envs.base_env import ActionTuple
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import (
            EngineConfigurationChannel,
        )

        self._ActionTuple = ActionTuple
        self._UnityEnvironment = UnityEnvironment
        self._EngineConfigurationChannel = EngineConfigurationChannel
        self._size = (int(size[0]), int(size[1]))
        self._max_episode_steps = int(max_episode_steps)
        self._agent_id = None
        self._include_state = bool(include_state)
        self._last_image: Any = None
        self._seed = int(seed)
        self._file_name = file_name
        self._worker_id = int(worker_id)
        self._base_port = int(base_port)
        self._no_graphics = bool(no_graphics)
        self._time_scale = float(time_scale)
        self._quality_level = int(quality_level)

        self._engine_channel = self._EngineConfigurationChannel()
        self._configure_engine_channel()

        self._env = self._make_env(self._seed)

        behavior_names = list(self._env.behavior_specs.keys())
        if not behavior_names:
            raise ValueError("No Unity behaviors found in the environment.")
        if behavior_name is None:
            behavior_name = behavior_names[0]
        if behavior_name not in self._env.behavior_specs:
            raise ValueError(
                f"Behavior '{behavior_name}' not found. Available: {behavior_names}"
            )

        self._behavior_name = behavior_name
        self._spec = self._env.behavior_specs[self._behavior_name]

        action_spec = self._spec.action_spec
        if not action_spec.is_continuous():
            raise ValueError(
                "UnityMLAgentsEnv currently supports only continuous action spaces."
            )
        self._action_size = int(action_spec.continuous_size)
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._action_size,), dtype=np.float32
        )
        self._observation_space = self._build_observation_space()
        self._seed_spaces(self._seed)

    def _configure_engine_channel(self) -> None:
        self._engine_channel.set_configuration_parameters(
            width=self._size[1],
            height=self._size[0],
            quality_level=self._quality_level,
            time_scale=self._time_scale,
        )

    def _make_env(self, seed: int) -> Any:
        env = self._UnityEnvironment(
            file_name=self._file_name,
            seed=int(seed),
            worker_id=self._worker_id,
            base_port=self._base_port,
            no_graphics=self._no_graphics,
            side_channels=[self._engine_channel],
        )
        env.reset()
        return env

    def _build_observation_space(self) -> gym.spaces.Dict:
        state_space = None
        if self._include_state:
            vector_dim = 0
            for spec in getattr(self._spec, "observation_specs", []):
                shape = tuple(getattr(spec, "shape", ()))
                if len(shape) == 3 and (
                    shape[-1] in (1, 3, 4) or shape[0] in (1, 3, 4)
                ):
                    continue
                if len(shape) > 0:
                    vector_dim += int(np.prod(shape))
            if vector_dim > 0:
                state_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(vector_dim,),
                    dtype=np.float32,
                )
        return add_optional_state_space(
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, self._size[0], self._size[1]),
                dtype=np.uint8,
            ),
            state_space=state_space,
        )

    def _seed_spaces(self, seed: int | None) -> None:
        if seed is None:
            return
        for space in (self._action_space, self._observation_space):
            if hasattr(space, "seed"):
                try:
                    space.seed(seed)
                except Exception:
                    pass

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def _extract_agent_data(self, steps: Any, preferred_agent_id: Any) -> Any:
        agent_ids = np.asarray(getattr(steps, "agent_id", []))
        if agent_ids.size == 0:
            return None

        if preferred_agent_id is None:
            idx = 0
            agent_id = int(agent_ids[idx])
        else:
            matches = np.where(agent_ids == preferred_agent_id)[0]
            if matches.size == 0:
                return None
            idx = int(matches[0])
            agent_id = int(preferred_agent_id)

        obs_list = [np.asarray(obs[idx]) for obs in steps.obs]
        rewards = np.asarray(getattr(steps, "reward", np.zeros(agent_ids.size)))
        reward = float(rewards[idx]) if rewards.size > idx else 0.0

        interrupted = False
        if hasattr(steps, "interrupted"):
            interrupted_arr = np.asarray(steps.interrupted)
            if interrupted_arr.size > idx:
                interrupted = bool(interrupted_arr[idx])

        return agent_id, obs_list, reward, interrupted

    def _vector_to_image(self, vector: Any) -> np.ndarray:
        arr: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            arr_norm = (arr - vmin) / (vmax - vmin)
            arr = arr_norm
        else:
            arr = np.zeros_like(arr)

        image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        bands = min(arr.size, 8)
        band_w = max(1, self._size[1] // max(1, bands))
        for i in range(bands):
            start = i * band_w
            end = min(self._size[1], start + band_w)
            image[:, start:end, :] = int(255.0 * float(arr[i]))
        return image

    def _to_hwc_uint8(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)

        if arr.ndim == 1:
            image = self._vector_to_image(arr)
        elif arr.ndim == 2:
            image = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3:
            image = arr
            # Handle CHW -> HWC if needed.
            if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
        else:
            image = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size > 0 and image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        if image.shape[0] != self._size[0] or image.shape[1] != self._size[1]:
            image = np.array(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return image

    def _obs_list_to_chw_image(self, obs_list: Any) -> np.ndarray:
        visual = None
        for obs in obs_list:
            arr = np.asarray(obs)
            if arr.ndim == 3:
                visual = arr
                break
        if visual is None and obs_list:
            visual = np.asarray(obs_list[0])
        if visual is None:
            visual = np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        image = self._to_hwc_uint8(visual)
        return image.transpose(2, 0, 1).copy()

    def _extract_vector_observation(self, obs_list: Any) -> np.ndarray | None:
        vectors: list[np.ndarray] = []
        for obs in obs_list:
            arr = np.asarray(obs)
            if arr.ndim == 0:
                continue
            if arr.ndim == 3 and (
                arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4)
            ):
                continue
            vectors.append(arr.astype(np.float32, copy=False).reshape(-1))
        if not vectors:
            return None
        return np.concatenate(vectors, axis=0)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._seed = int(seed)
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = self._make_env(self._seed)
            self._spec = self._env.behavior_specs[self._behavior_name]
            self._observation_space = self._build_observation_space()
            self._seed_spaces(self._seed)
        self._agent_id = None
        self._last_image = None
        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)

        data = self._extract_agent_data(decision_steps, preferred_agent_id=None)
        if data is None:
            data = self._extract_agent_data(terminal_steps, preferred_agent_id=None)
        if data is None:
            raise RuntimeError("No Unity agents were available after reset.")

        self._agent_id, obs_list, _, _ = data
        image = self._obs_list_to_chw_image(obs_list)
        self._last_image = image
        observation = {"image": image}
        if self._include_state:
            vector_observation = self._extract_vector_observation(obs_list)
            if vector_observation is not None:
                observation["state"] = vector_observation.copy()
        return observation

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._agent_id is None:
            raise RuntimeError(
                "Environment has terminated. Call reset() before step()."
            )

        clipped = clip_box_action(
            action,
            -np.ones((self._action_size,), dtype=np.float32),
            np.ones((self._action_size,), dtype=np.float32),
        )
        action = clipped.reshape(1, self._action_size)

        self._env.set_actions(
            self._behavior_name,
            self._ActionTuple(continuous=action),
        )
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self._behavior_name)
        terminal_data = self._extract_agent_data(terminal_steps, self._agent_id)

        interrupted = False
        if terminal_data is not None:
            _, obs_list, reward, interrupted = terminal_data
            done = True
            self._agent_id = None
        else:
            decision_data = self._extract_agent_data(decision_steps, self._agent_id)
            if decision_data is None:
                decision_data = self._extract_agent_data(decision_steps, None)
            if decision_data is None:
                raise RuntimeError("No decision step data found after Unity step.")
            self._agent_id, obs_list, reward, _ = decision_data
            done = False

        image = self._obs_list_to_chw_image(obs_list)
        self._last_image = image

        info = {
            "discount": np.array(0.0 if done else 1.0, dtype=np.float32),
            "action": clipped.copy(),
            "executed_action": clipped.copy(),
        }
        vector_observation = self._extract_vector_observation(obs_list)
        if vector_observation is not None:
            info["vector_observation"] = vector_observation.copy()
        if done:
            info["interrupted"] = bool(interrupted)
        info = finalize_step_info(
            info,
            done=done,
            terminated=done and not interrupted,
            truncated=done and interrupted,
        )

        observation = {"image": image}
        if self._include_state and vector_observation is not None:
            observation["state"] = vector_observation.copy()
        return observation, float(reward), bool(done), info

    def render(self, *args: Any, **kwargs: Any) -> Any:
        if self._last_image is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_image.transpose(1, 2, 0).copy()

    def close(self) -> None:
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()
