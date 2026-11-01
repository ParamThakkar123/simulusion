"""Minecraft environment adapters for Simulusion image-based agents.

Minecraft is the natural stress test for a world model: partially observable,
visually rich, and with reward structures far sparser than Atari. The IRIS
authors themselves worked on the MineRL Diamond competition (Kanervisto et al.,
2022), which the IRIS paper cites as one of the benchmarks beyond Atari 100k.

Two backends are supported:

* **MineRL** (``minerl``) -- the competition environment. Observations are a dict
  containing a ``pov`` RGB image; actions are a ``Dict`` space mixing binary
  keypresses with a continuous camera delta.
* **MineDojo** (``minedojo``) -- built on the same Malmo foundation, with a
  larger task suite and channels-first RGB observations.

Neither exposes what an image-based discrete-action agent such as IRIS or
DreamerV2 needs, so :class:`MinecraftDiscreteEnv` adapts them:

* the ``Dict`` action space collapses to ``Discrete(n)`` over a curated action
  set (see :data:`MINECRAFT_ACTION_SET`), the standard approach for MineRL
  agents -- a raw factorised space would need a multi-head policy;
* the observation dict reduces to a plain ``uint8`` HWC image;
* the Gymnasium 5-tuple step API is presented regardless of which API the
  underlying package uses.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from simulusion.envs._contract import finalize_step_info

_MINERL_PACKAGE = "minerl"
_MINEDOJO_PACKAGE = "minedojo"

# Camera deltas are in degrees; 10 degrees per step is the granularity the MineRL
# competition baselines converged on. Larger steps turn faster but make fine aim
# impossible; smaller ones waste the agent's limited step budget.
DEFAULT_CAMERA_DELTA = 10.0

# A curated discrete action set. MineRL's native space is a Dict of nine binary
# keypresses plus a 2-D continuous camera delta, i.e. 2^9 x R^2 -- far too large
# to enumerate and not directly usable by a categorical policy. This set covers
# navigation, looking around, and the two interaction verbs, which is what the
# navigation and treechop tasks need.
#
# Each entry is (name, {action_key: value}). Keys absent from a dict take the
# environment's no-op value.
MINECRAFT_ACTION_SET: tuple[tuple[str, Mapping[str, Any]], ...] = (
    ("noop", {}),
    ("forward", {"forward": 1}),
    ("back", {"back": 1}),
    ("left", {"left": 1}),
    ("right", {"right": 1}),
    ("jump", {"jump": 1}),
    ("forward_jump", {"forward": 1, "jump": 1}),
    ("attack", {"attack": 1}),
    ("use", {"use": 1}),
    ("camera_left", {"camera": (0.0, -DEFAULT_CAMERA_DELTA)}),
    ("camera_right", {"camera": (0.0, DEFAULT_CAMERA_DELTA)}),
    ("camera_up", {"camera": (-DEFAULT_CAMERA_DELTA, 0.0)}),
    ("camera_down", {"camera": (DEFAULT_CAMERA_DELTA, 0.0)}),
)

# MineRL environment ids that expose only navigation/interaction actions, i.e.
# the ones MINECRAFT_ACTION_SET fully covers. The Obtain* tasks additionally
# need craft/place/equip actions; they still run here, but the agent will never
# emit those, so it cannot progress past what tool-free play allows.
MINERL_NAVIGATION_ENVS = (
    "MineRLTreechop-v0",
    "MineRLNavigate-v0",
    "MineRLNavigateDense-v0",
    "MineRLNavigateExtreme-v0",
    "MineRLNavigateExtremeDense-v0",
)

MINERL_OBTAIN_ENVS = (
    "MineRLObtainDiamond-v0",
    "MineRLObtainDiamondDense-v0",
    "MineRLObtainIronPickaxe-v0",
    "MineRLObtainIronPickaxeDense-v0",
)

MINERL_ENVS = MINERL_NAVIGATION_ENVS + MINERL_OBTAIN_ENVS


def list_minecraft_envs() -> list[str]:
    """Return the MineRL environment ids this adapter is known to handle."""
    return list(MINERL_ENVS)


def list_minecraft_actions() -> list[str]:
    """Return the names of the discrete actions, indexed by action id."""
    return [name for name, _ in MINECRAFT_ACTION_SET]


def _require_package(package: str, extra: str) -> Any:
    """Import an optional Minecraft backend with an actionable install message."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        spec = None
    if spec is None and package not in sys.modules:
        raise ImportError(
            f"Minecraft support requires the optional '{package}' package. "
            f"Install it with `pip install simulusion[{extra}]` or "
            f"`pip install {package}`. Note that it also needs a Java runtime "
            "(JDK 8 for MineRL) and launches a real Minecraft client, so it "
            "cannot run in a headless container without a virtual display."
        )
    if package in sys.modules:
        return sys.modules[package]
    return importlib.import_module(package)


def _extract_pov(observation: Any) -> NDArray[np.uint8]:
    """Pull the RGB frame out of a Minecraft observation.

    MineRL returns ``{"pov": (H, W, 3) uint8, ...}``; MineDojo returns
    ``{"rgb": (3, H, W) uint8, ...}``. Both are normalised to HWC uint8.
    """
    frame: Any = observation
    if isinstance(observation, Mapping):
        for key in ("pov", "rgb", "image"):
            if key in observation:
                frame = observation[key]
                break
        else:
            raise KeyError(
                "Minecraft observation had no 'pov', 'rgb' or 'image' key; got "
                f"{sorted(observation)}."
            )

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.ndim == 2:
        array = array[:, :, None]
    if array.dtype != np.uint8:
        # MineDojo can hand back float images in [0, 1].
        if (
            np.issubdtype(array.dtype, np.floating)
            and float(array.max(initial=0.0)) <= 1.0
        ):
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


class MinecraftDiscreteEnv(gym.Env):
    """Discrete-action, image-observation view of a MineRL or MineDojo task.

    Args:
        env: An already-constructed backend environment. When omitted, one is
            built from ``env_id`` using the requested backend.
        env_id: Environment id, e.g. ``"MineRLTreechop-v0"``.
        backend: ``"minerl"`` or ``"minedojo"``.
        action_set: Overrides :data:`MINECRAFT_ACTION_SET`. Supply a task-specific
            set to expose craft/place actions for the Obtain* tasks.
        seed: Seed forwarded to the backend when it supports seeding.
        env_kwargs: Extra keyword arguments for the backend constructor.

    Attributes:
        action_names: Action id -> human-readable name, useful when inspecting
            what a trained policy actually does.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env: Any | None = None,
        env_id: str = "MineRLTreechop-v0",
        backend: str = "minerl",
        action_set: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
        seed: int | None = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()

        self.env_id = env_id
        self.backend = backend
        self._action_set = tuple(action_set or MINECRAFT_ACTION_SET)
        self.action_names = [name for name, _ in self._action_set]

        self.env = env if env is not None else self._make_backend_env(**env_kwargs)

        if seed is not None:
            self._seed_backend(seed)

        self.action_space = gym.spaces.Discrete(len(self._action_set))

        # Probe the real observation rather than trusting the declared space:
        # backends disagree on layout, and a wrong shape here would surface much
        # later as a confusing error inside the encoder.
        probe = _extract_pov(self._reset_backend())
        self._frame_shape = probe.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=probe.shape, dtype=np.uint8
        )
        self._pending_reset_obs: NDArray[np.uint8] | None = probe

    def _make_backend_env(self, **env_kwargs: Any) -> Any:
        if self.backend == "minerl":
            _require_package(_MINERL_PACKAGE, "minerl")
            # Importing minerl registers its Gym ids as a side effect.
            import gym as legacy_gym  # MineRL targets the pre-Gymnasium API

            return legacy_gym.make(self.env_id, **env_kwargs)
        if self.backend == "minedojo":
            minedojo = _require_package(_MINEDOJO_PACKAGE, "minedojo")
            return minedojo.make(task_id=self.env_id, **env_kwargs)
        raise ValueError(
            f"Unknown Minecraft backend {self.backend!r}; expected 'minerl' or "
            "'minedojo'."
        )

    def _seed_backend(self, seed: int) -> None:
        seeder = getattr(self.env, "seed", None)
        if callable(seeder):
            try:
                seeder(seed)
                return
            except Exception:  # noqa: BLE001 - seeding is best-effort
                pass
        space = getattr(self.env, "action_space", None)
        if space is not None and hasattr(space, "seed"):
            space.seed(seed)

    def _reset_backend(self) -> Any:
        result = self.env.reset()
        # Gymnasium returns (obs, info); the older Gym API returns just obs.
        if isinstance(result, tuple) and len(result) == 2:
            return result[0]
        return result

    def _noop_action(self) -> Any:
        """A no-op action in the backend's own action space."""
        space = self.env.action_space
        sampler = getattr(space, "no_op", None)
        if callable(sampler):
            return sampler()
        if isinstance(space, gym.spaces.Dict) or hasattr(space, "spaces"):
            # Values are per-subspace: a zero array for shaped subspaces, a
            # plain 0 for scalar ones.
            action: dict[str, Any] = {}
            for key, subspace in space.spaces.items():
                if hasattr(subspace, "shape") and subspace.shape:
                    action[key] = np.zeros(subspace.shape, dtype=np.float32)
                else:
                    action[key] = 0
            return action
        return 0

    def translate_action(self, action: int) -> Any:
        """Map a discrete action id to the backend's native action.

        Exposed publicly so the mapping can be inspected or reused when driving
        the environment outside this wrapper.
        """
        index = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= index < len(self._action_set):
            raise ValueError(
                f"Action {index} out of range for {len(self._action_set)} actions."
            )

        native = self._noop_action()
        _name, settings = self._action_set[index]
        if not isinstance(native, dict):
            # A backend that already uses a flat discrete space: pass through.
            return index

        for key, value in settings.items():
            if key not in native:
                # Skip actions the task does not support (e.g. "use" in
                # Treechop) rather than crashing -- they become no-ops.
                continue
            if key == "camera":
                native[key] = np.asarray(value, dtype=np.float32)
            else:
                native[key] = value
        return native

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        del options
        if seed is not None:
            self._seed_backend(seed)
            self._pending_reset_obs = None

        # The constructor already reset the backend to probe the frame shape;
        # reuse that observation instead of paying for a second world load,
        # which for Minecraft costs seconds, not milliseconds.
        if self._pending_reset_obs is not None:
            observation = self._pending_reset_obs
            self._pending_reset_obs = None
            return observation, {}

        return _extract_pov(self._reset_backend()), {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        result = self.env.step(self.translate_action(action))

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = bool(terminated) or bool(truncated)
        else:
            observation, reward, done, info = result
            terminated, truncated = None, None

        info = finalize_step_info(
            info, done=bool(done), terminated=terminated, truncated=truncated
        )
        return (
            _extract_pov(observation),
            float(reward),
            bool(info["terminated"]),
            bool(info["truncated"]),
            info,
        )

    def render(self) -> Any:
        renderer = getattr(self.env, "render", None)
        if callable(renderer):
            try:
                rendered = renderer()
            except Exception:  # noqa: BLE001 - rendering is optional
                return None
            if rendered is not None:
                return _extract_pov(rendered)
        return None

    def close(self) -> None:
        closer = getattr(self.env, "close", None)
        if callable(closer):
            closer()


def make_minecraft_env(
    env_id: str = "MineRLTreechop-v0",
    backend: str = "minerl",
    action_set: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    seed: int | None = None,
    **env_kwargs: Any,
) -> MinecraftDiscreteEnv:
    """Create a Minecraft environment usable by Simulusion's image agents.

    The result has a ``Discrete`` action space and ``uint8`` HWC observations, so
    it can be handed straight to :class:`~simulusion.training.train_iris.IRISTrainer`
    via its ``env`` argument::

        from simulusion.envs import make_minecraft_env
        from simulusion.training.train_iris import IRISTrainer

        env = make_minecraft_env("MineRLTreechop-v0")
        trainer = IRISTrainer(game="MineRLTreechop-v0", config=cfg, env=env)

    Minecraft is far from the Atari 100k regime the paper's defaults target:
    episodes are long, rewards are sparse, and each environment step is orders of
    magnitude slower. Expect to raise ``imagination_horizon`` and the collection
    budget, and to need many more environment steps than 100k.
    """
    return MinecraftDiscreteEnv(
        env_id=env_id,
        backend=backend,
        action_set=action_set,
        seed=seed,
        **env_kwargs,
    )


__all__ = [
    "DEFAULT_CAMERA_DELTA",
    "MINECRAFT_ACTION_SET",
    "MINERL_ENVS",
    "MINERL_NAVIGATION_ENVS",
    "MINERL_OBTAIN_ENVS",
    "MinecraftDiscreteEnv",
    "list_minecraft_actions",
    "list_minecraft_envs",
    "make_minecraft_env",
]
