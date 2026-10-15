from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.envs._observations import add_optional_state_space
import gymnasium as gym
import numpy as np
from PIL import Image


from simulusion.envs.gym_env import GymImageEnv
from simulusion.envs.robotics_env import (
    make_gymnasium_env_with_robotics_fallback,
    register_gymnasium_robotics_envs,
)

RewardFn = Callable[[Any, Any, np.ndarray, dict[str, Any]], float]
TerminalFn = Callable[[Any, Any, dict[str, Any]], bool]


def _load_mujoco() -> Any:
    if "mujoco" in sys.modules:
        return sys.modules["mujoco"]
    if importlib.util.find_spec("mujoco") is None:
        raise ImportError(
            "The native MuJoCo bindings are required for MuJoCoImageEnv. "
            "Install them with `pip install mujoco` or `pip install simulusion[mujoco]`."
        )
    return importlib.import_module("mujoco")


def _validate_model_source(
    *,
    xml_path: str | Path | None,
    xml_string: str | None,
    binary_path: str | Path | None,
) -> None:
    sources = [xml_path is not None, xml_string is not None, binary_path is not None]
    if sum(sources) != 1:
        raise ValueError(
            "Provide exactly one of xml_path, xml_string, or binary_path for MuJoCoImageEnv."
        )


def _is_native_model_source(model: str | Path) -> bool:
    model_text = str(model)
    return (
        model_text.lstrip().startswith("<")
        or model_text.endswith((".xml", ".mjb"))
        or Path(model_text).exists()
    )


def _infer_model_source(model: str | Path) -> dict[str, str | Path]:
    model_text = str(model)
    if model_text.lstrip().startswith("<"):
        return {"xml_string": model_text}
    if model_text.endswith(".mjb"):
        return {"binary_path": model}
    return {"xml_path": model}


def make_mujoco_env_from_config(args: Any, size: tuple[int, int]) -> Any:
    """Build a MuJoCo image environment from a DreamerConfig-like object."""
    native_kwargs = {
        "seed": args.seed,
        "size": size,
        "camera": getattr(args, "mujoco_camera", None),
        "frame_skip": int(getattr(args, "mujoco_frame_skip", 1)),
        "reset_noise_scale": float(getattr(args, "mujoco_reset_noise_scale", 0.0)),
    }
    if getattr(args, "mujoco_xml_string", None) is not None:
        return make_mujoco_env(xml_string=args.mujoco_xml_string, **native_kwargs)
    if getattr(args, "mujoco_binary_path", None) is not None:
        return make_mujoco_env(binary_path=args.mujoco_binary_path, **native_kwargs)
    if getattr(args, "mujoco_xml_path", None) is not None:
        return make_mujoco_env(xml_path=args.mujoco_xml_path, **native_kwargs)

    gym_kwargs = {}
    reset_noise_scale = native_kwargs["reset_noise_scale"]
    if reset_noise_scale != 0.0:
        gym_kwargs["reset_noise_scale"] = reset_noise_scale
    return make_mujoco_env(
        args.env,
        seed=args.seed,
        size=size,
        gym_kwargs=gym_kwargs,
    )


class MuJoCoImageEnv:
    """Native MuJoCo environment adapter for pixel-based world-model training.

    The adapter uses the low-level ``mujoco`` Python package directly: models are
    compiled from MJCF XML strings/files or MJB binaries via ``mujoco.MjModel``;
    simulation state lives in ``mujoco.MjData``; actions are written to
    ``data.ctrl``; and images are produced with ``mujoco.Renderer``. Observations
    follow Simulusion's Dreamer-style contract: a dict with required key
    ``"image"`` containing ``uint8[C, H, W]`` frames.

    Native MuJoCo models do not define task rewards or episode termination by
    themselves, so callers can supply ``reward_fn`` and ``terminal_fn`` callbacks.
    By default, rewards are ``0.0`` and episodes terminate only through external
    wrappers such as ``TimeLimit``.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        xml_string: str | None = None,
        binary_path: str | Path | None = None,
        assets: dict[str, bytes] | None = None,
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        camera: str | int | None = None,
        reward_fn: RewardFn | None = None,
        terminal_fn: TerminalFn | None = None,
        frame_skip: int = 1,
        reset_noise_scale: float = 0.0,
        default_control_range: tuple[float, float] = (-1.0, 1.0),
        include_state: bool = False,
    ) -> None:
        _validate_model_source(
            xml_path=xml_path,
            xml_string=xml_string,
            binary_path=binary_path,
        )

        self._mujoco = _load_mujoco()
        self._size = (int(size[0]), int(size[1]))
        self._camera = camera
        self._reward_fn = reward_fn
        self._terminal_fn = terminal_fn
        self._frame_skip = max(1, int(frame_skip))
        self._reset_noise_scale = float(reset_noise_scale)
        self._rng = np.random.default_rng(seed)
        self._include_state = bool(include_state)
        self._closed = False

        if xml_string is not None:
            self.model = self._mujoco.MjModel.from_xml_string(xml_string, assets)
        elif binary_path is not None:
            self.model = self._mujoco.MjModel.from_binary_path(str(binary_path))
        else:
            self.model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self._mujoco.MjData(self.model)

        height, width = self._size
        self._renderer = self._mujoco.Renderer(self.model, height=height, width=width)

        state_space = None
        if self._include_state:
            state_dim = int(getattr(self.model, "nq", 0)) + int(
                getattr(self.model, "nv", 0)
            )
            state_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32,
            )
        self._observation_space = add_optional_state_space(
            gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, height, width),
                dtype=np.uint8,
            ),
            state_space=state_space,
        )
        self._action_space = self._build_action_space(default_control_range)
        self.max_episode_steps = 1000

    def _build_action_space(
        self, default_control_range: tuple[float, float]
    ) -> gym.spaces.Box:
        action_dim = int(getattr(self.model, "nu", 0))
        if action_dim <= 0:
            return gym.spaces.Box(
                low=np.zeros((0,), dtype=np.float32),
                high=np.zeros((0,), dtype=np.float32),
                dtype=np.float32,
            )

        ctrlrange = np.asarray(
            getattr(self.model, "actuator_ctrlrange", []), dtype=np.float32
        )
        limited = np.asarray(
            getattr(self.model, "actuator_ctrllimited", []), dtype=bool
        )
        if ctrlrange.shape == (action_dim, 2) and limited.shape[0] == action_dim:
            default_low, default_high = default_control_range
            low = np.where(limited, ctrlrange[:, 0], float(default_low)).astype(
                np.float32
            )
            high = np.where(limited, ctrlrange[:, 1], float(default_high)).astype(
                np.float32
            )
        else:
            low = np.full(
                (action_dim,), float(default_control_range[0]), dtype=np.float32
            )
            high = np.full(
                (action_dim,), float(default_control_range[1]), dtype=np.float32
            )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def _vector_state(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(getattr(self.data, "qpos", []), dtype=np.float32).reshape(
                    -1
                ),
                np.asarray(getattr(self.data, "qvel", []), dtype=np.float32).reshape(
                    -1
                ),
            ],
            axis=0,
        )

    def _build_observation(self) -> dict[str, np.ndarray]:
        observation = {"image": self._render_chw()}
        if self._include_state:
            observation["state"] = self._vector_state().copy()
        return observation

    def _render_chw(self) -> np.ndarray:
        if self._camera is None:
            self._renderer.update_scene(self.data)
        else:
            self._renderer.update_scene(self.data, camera=self._camera)
        image = np.asarray(self._renderer.render())
        if image.shape[:2] != self._size:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image.astype(np.uint8, copy=False).transpose(2, 0, 1).copy()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        if self._reset_noise_scale > 0.0:
            if getattr(self.model, "nq", 0):
                self.data.qpos[:] += self._rng.normal(
                    0.0, self._reset_noise_scale, size=self.data.qpos.shape
                )
            if getattr(self.model, "nv", 0):
                self.data.qvel[:] += self._rng.normal(
                    0.0, self._reset_noise_scale, size=self.data.qvel.shape
                )
        self._mujoco.mj_forward(self.model, self.data)
        return self._build_observation()

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        clipped = clip_box_action(
            action,
            self.action_space.low,
            self.action_space.high,
        )
        if clipped.size:
            self.data.ctrl[:] = clipped
        self._mujoco.mj_step(self.model, self.data, nstep=self._frame_skip)

        info = {
            "action": clipped.astype(np.float32, copy=True),
            "executed_action": clipped.astype(np.float32, copy=True),
            "time": float(getattr(self.data, "time", 0.0)),
            "qpos": np.asarray(getattr(self.data, "qpos", []), dtype=np.float64).copy(),
            "qvel": np.asarray(getattr(self.data, "qvel", []), dtype=np.float64).copy(),
        }
        info["vector_observation"] = self._vector_state().copy()
        if self._reward_fn is None:
            reward = 0.0
        else:
            reward = float(self._reward_fn(self.model, self.data, clipped, info))
        done = (
            bool(self._terminal_fn(self.model, self.data, info))
            if self._terminal_fn
            else False
        )
        info = finalize_step_info(
            info,
            done=done,
            terminated=done,
            truncated=False,
        )
        return self._build_observation(), reward, done, info

    def render(self) -> Any:
        return self._render_chw().transpose(1, 2, 0).copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._renderer, "close", None)
        if callable(close):
            close()


def make_mujoco_env(
    model: str | Path | None = None,
    *,
    backend: str = "auto",
    seed: int = 0,
    size: tuple[int, int] = (64, 64),
    render_mode: str = "rgb_array",
    gym_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> GymImageEnv | MuJoCoImageEnv:
    """Create one MuJoCo image environment factory for tasks and MJCF/MJB models.

    Args:
        model: Either a Gymnasium MuJoCo task id such as ``"Humanoid-v4"``,
            an MJCF XML path/string, or an MJB binary path.
        backend: ``"auto"`` infers native vs Gymnasium task mode. Use
            ``"native"`` for MJCF/MJB, ``"gymnasium"`` for task ids, or
            ``"robotics"`` for Gymnasium Robotics registrations.
        seed: Seed forwarded to the image wrapper.
        size: Target ``(height, width)`` image size.
        render_mode: Render mode used for Gymnasium MuJoCo task ids.
        gym_kwargs: Optional keyword arguments forwarded to ``gymnasium.make``
            in task-id mode. Extra ``**kwargs`` are also forwarded there.
        **kwargs: Native ``MuJoCoImageEnv`` options for MJCF/MJB mode, or
            environment-constructor options for Gymnasium task-id mode.

    Returns:
        A Simulusion image environment returning ``{"image": uint8[C, H, W]}``.
    """
    backend = backend.lower()
    explicit_native_source = any(
        key in kwargs for key in ("xml_path", "xml_string", "binary_path")
    )
    use_native = backend in {"native", "mjcf", "mjb"} or (
        backend == "auto"
        and (
            explicit_native_source
            or (model is not None and _is_native_model_source(model))
        )
    )

    if use_native:
        if model is not None:
            kwargs.update(
                {k: v for k, v in _infer_model_source(model).items() if k not in kwargs}
            )
        return MuJoCoImageEnv(seed=seed, size=size, **kwargs)

    if backend not in {
        "auto",
        "gym",
        "gymnasium",
        "task",
        "robotics",
        "gymnasium_robotics",
    }:
        raise ValueError(
            f"Unknown MuJoCo backend={backend!r}. Use 'auto', 'native', 'gymnasium', or 'robotics'."
        )
    if model is None:
        raise ValueError(
            "A Gymnasium MuJoCo environment id, XML path/string, or MJB path is required."
        )

    env_kwargs = dict(gym_kwargs or {})
    env_kwargs.update(kwargs)
    if backend in {"robotics", "gymnasium_robotics"}:
        register_gymnasium_robotics_envs()
    env = make_gymnasium_env_with_robotics_fallback(
        str(model),
        render_mode=render_mode,
        gym_kwargs=env_kwargs,
    )
    return GymImageEnv(env, seed=seed, size=size, render_mode=render_mode)
