from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulusion.envs._actions import clip_box_action
from simulusion.envs._contract import finalize_step_info
from simulusion.utils.gym_compat import gym
import numpy as np

# MuJoCo's mjtJoint, and how many qpos / qvel entries each joint type owns.
_FREE, _BALL, _SLIDE, _HINGE = 0, 1, 2, 3
_QPOS_PARTS = {
    _FREE: (("x", "free_pos"), ("y", "free_pos"), ("z", "free_pos"))
    + tuple((q, "free_quat") for q in ("qw", "qx", "qy", "qz")),
    _BALL: tuple((q, "ball_quat") for q in ("qw", "qx", "qy", "qz")),
    _SLIDE: (("", "slide"),),
    _HINGE: (("", "hinge"),),
}
_QVEL_PARTS = {
    _FREE: tuple((v, "free_lin") for v in ("vx", "vy", "vz"))
    + tuple((w, "free_ang") for w in ("wx", "wy", "wz")),
    _BALL: tuple((w, "ball_ang") for w in ("wx", "wy", "wz")),
    _SLIDE: (("", "slide"),),
    _HINGE: (("", "hinge"),),
}


@dataclass(frozen=True)
class PhysicsStateSpec:
    """Names and kinds of each entry of :meth:`DeepMindControlEnv.physics_state`.

    ``*_names`` label every dimension (``"rootx"``, ``"root/qw"``); ``*_kinds``
    say how to treat it: ``"hinge"`` is an angle in radians, ``"slide"`` a
    linear position, ``"free_pos"``/``"free_quat"``/``"ball_quat"`` the parts
    of free and ball joints, and the ``qvel`` kinds mirror them as
    ``"hinge"``, ``"slide"``, ``"free_lin"``, ``"free_ang"`` and ``"ball_ang"``.
    """

    qpos_names: tuple[str, ...]
    qpos_kinds: tuple[str, ...]
    qvel_names: tuple[str, ...]
    qvel_kinds: tuple[str, ...]
    act_names: tuple[str, ...]


class DeepMindControlEnv:
    """Gym-style adapter for DeepMind Control Suite tasks.

    The wrapper exposes DMC observations and actions through Gym spaces and
    adds a rendered RGB image to each observation dict so image-based world
    model pipelines can train consistently across backends.

    Features:
        - Parses domain-task names (e.g., "cheetah-run" -> domain="cheetah", task="run")
        - Automatically handles special cases like "cup" -> "ball_in_cup"
        - Renders RGB images at configurable resolution
        - Returns observations as dict with both state vectors and images
        - Reports the true MuJoCo state (``qpos``, ``qvel``, ``act``) in
          ``info["physics_state"]``; see :meth:`physics_state` and
          :meth:`physics_state_spec`

    Args:
        name (str): Environment name in format "domain-task" (e.g., "cheetah-run").
        seed (int): Random seed for environment initialization.
        size (tuple): Target image size as (height, width) (default: (64, 64)).
        camera (int, optional): Camera ID for rendering. Defaults to 0 for most
            domains, 2 for quadruped.

    Attributes:
        observation_space (gym.spaces.Dict): Dict space with state keys and "image".
        action_space (gym.spaces.Box): Continuous action space from DMC spec.

    Example:
        >>> env = DeepMindControlEnv("cheetah-run", seed=0, size=(64, 64))
        >>> obs = env.reset()
        >>> print(obs.keys())  # dict_keys(['position', 'velocity', 'image'])
    """

    def __init__(
        self,
        name: str,
        seed: int,
        size: tuple[int, int] = (64, 64),
        camera: int | None = None,
    ) -> None:
        if "-" not in name:
            raise ValueError(
                f"DeepMind Control task names use 'domain-task' form, e.g. "
                f"'cartpole-balance' or 'walker-walk'; got {name!r}."
            )
        domain, task = name.split("-", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._name = name
        self._domain = domain
        self._task = task
        self._seed = int(seed)
        self._size = (int(size[0]), int(size[1]))
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self._env = self._make_env(self._seed)

        spaces: dict[str, gym.spaces.Space[Any]] = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8)
        self._observation_space = gym.spaces.Dict(spaces)
        spec = self._env.action_spec()
        self._action_space = gym.spaces.Box(
            spec.minimum, spec.maximum, dtype=np.float32
        )
        self._seed_spaces(self._seed)

    def _make_env(self, seed: int) -> Any:
        try:
            from dm_control import suite
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The DeepMind Control backend requires the 'dm_control' package, "
                "which is not installed. Install it with:\n\n"
                "    pip install simulusion[dmc]          # Python <= 3.12\n"
                "    python -m simulusion.install_dmc     # any version, incl. 3.13\n\n"
                "On CPython 3.13 the extra alone is not enough: dm-control pins "
                "`labmaze`, which has no 3.13 wheel and builds with Bazel. The "
                "installer above uses the pure-Python `labmaze-new` instead.\n\n"
                "or pick a backend that is already available (for example a "
                "Gymnasium task via env_backend='gym', such as 'Pendulum-v1')."
            ) from exc

        return suite.load(self._domain, self._task, task_kwargs={"random": int(seed)})

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

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        clipped = clip_box_action(action, self.action_space.low, self.action_space.high)
        time_step = self._env.step(clipped)
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        vector_observation = np.concatenate(
            [
                np.asarray(value, dtype=np.float32).reshape(-1)
                for value in time_step.observation.values()
            ],
            axis=0,
        )
        info = finalize_step_info(
            {
                "discount": np.array(time_step.discount, np.float32),
                "action": clipped.copy(),
                "executed_action": clipped.copy(),
                "vector_observation": vector_observation,
                "physics_state": self.physics_state(),
            },
            done=done,
            terminated=done,
            truncated=False,
        )
        return obs, float(reward), done, info

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._seed = int(seed)
            self._env = self._make_env(self._seed)
            self._seed_spaces(self._seed)
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs["image"] = self.render().transpose(2, 0, 1).copy()
        return obs

    def physics_state(self) -> dict[str, np.ndarray]:
        """The simulator's current state as float32 copies.

        ``qpos`` holds generalized positions, ``qvel`` velocities and ``act``
        actuator activations (empty for most suite domains). Unlike the task
        observations, these are the quantities the dynamics evolve, so they are
        the ground truth for probing what a world model's latent recovers.
        :meth:`step` returns the same dict in ``info["physics_state"]``; call
        this after :meth:`reset` for the initial state.
        """
        data = self._env.physics.data
        return {
            key: np.array(getattr(data, key), dtype=np.float32).reshape(-1)
            for key in ("qpos", "qvel", "act")
        }

    def physics_state_spec(self) -> PhysicsStateSpec:
        """Name and kind of every dimension of :meth:`physics_state`."""
        model = self._env.physics.model
        qpos: list[tuple[int, str, str]] = []
        qvel: list[tuple[int, str, str]] = []
        for joint in range(int(model.njnt)):
            name = model.id2name(joint, "joint") or f"joint{joint}"
            kind = int(model.jnt_type[joint])
            for layout, start, out in (
                (_QPOS_PARTS, int(model.jnt_qposadr[joint]), qpos),
                (_QVEL_PARTS, int(model.jnt_dofadr[joint]), qvel),
            ):
                for offset, (part, label) in enumerate(layout[kind]):
                    out.append(
                        (start + offset, f"{name}/{part}" if part else name, label)
                    )
        qpos.sort()
        qvel.sort()
        act = sorted(
            (int(model.actuator_actadr[i]), model.id2name(i, "actuator") or f"act{i}")
            for i in range(int(model.nu))
            if int(model.actuator_actadr[i]) >= 0
        )
        return PhysicsStateSpec(
            qpos_names=tuple(name for _, name, _ in qpos),
            qpos_kinds=tuple(kind for _, _, kind in qpos),
            qvel_names=tuple(name for _, name, _ in qvel),
            qvel_kinds=tuple(kind for _, _, kind in qvel),
            act_names=tuple(name for _, name in act),
        )

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


def make_dmc_env(
    env_id: str,
    seed: int = 0,
    size: tuple[int, int] = (64, 64),
    camera: int | None = None,
) -> DeepMindControlEnv:
    """Create a DeepMind Control Suite task, e.g. ``make_dmc_env("walker-walk")``."""
    return DeepMindControlEnv(env_id, seed=seed, size=size, camera=camera)
