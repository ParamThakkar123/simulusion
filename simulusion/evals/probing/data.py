"""Episodes of a DeepMind Control task labelled with the simulator's true state.

A probe dataset pairs what a world model sees (frames and actions) with what
generated them (MuJoCo's ``qpos``, ``qvel`` and ``act``). Probing a latent means
regressing these targets from it, so the dataset also turns the raw state into
regression targets: hinge angles become ``sin``/``cos`` pairs, quaternions get
a fixed sign, and dimensions the camera cannot show are left out.

.. code-block:: python

    data = collect_probe_episodes("cartpole-swingup", episodes=50, seed=0)
    train, val = data.split(0.2)
    targets = train.targets()          # StateTargets: (N, K) values and names
    data.save("data/probe/cartpole")   # an ordinary episode folder
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeFolder, EpisodeWriter
from simulusion.envs.dmc import PhysicsStateSpec

Policy = Callable[[dict], np.ndarray]

STATE_KEYS = ("qpos", "qvel", "act")
TASK_PREFIX = "task."
SOURCE_PREFIX = "dmc:"

# Domains whose default cameras (``trackcom``) follow the body's centre of
# mass. Horizontal root position then shows only as the phase of a repeating
# floor pattern, so no encoder can be expected to recover it. Keyed by domain:
# the camera ids that track, and the qpos dimensions they hide.
_TRACKED_POSITION: dict[str, tuple[frozenset[int], tuple[str, ...]]] = {
    "walker": (frozenset({0, 1}), ("rootx",)),
    "cheetah": (frozenset({0, 1}), ("rootx",)),
    "hopper": (frozenset({0, 1}), ("rootx",)),
    "swimmer": (frozenset({0, 1}), ("rootx", "rooty")),
    "humanoid": (frozenset({0, 1}), ("root/x", "root/y")),
    "quadruped": (frozenset({0, 1, 2}), ("root/x", "root/y")),
    "fish": (frozenset({0, 1, 2}), ("root/x", "root/y")),
}


def unobservable_dims(domain: str, camera: int) -> tuple[str, ...]:
    """The ``qpos`` names a tracking ``camera`` of ``domain`` cannot show."""
    cameras, names = _TRACKED_POSITION.get(domain, (frozenset(), ()))
    return names if camera in cameras else ()


@dataclass(frozen=True)
class StateTargets:
    """Regression targets for every frame of a probe dataset.

    Attributes:
        values: ``(N, K)`` float32, one row per frame, episodes in order.
        names: The ``K`` target names, by group (``"qpos/hinge_1/sin"``,
            ``"qvel/slider"``, ``"task/position/0"``).
        episode: ``(N,)`` index of each row's episode in the dataset.
        step: ``(N,)`` index of each row's frame within its episode.
    """

    values: np.ndarray
    names: tuple[str, ...]
    episode: np.ndarray
    step: np.ndarray

    def __len__(self) -> int:
        return int(self.values.shape[0])


@dataclass
class ProbeDataset(Sequence[Episode]):
    """Episodes of one DMC task with the true state stored per frame.

    Each episode keeps ``qpos``, ``qvel`` and ``act`` as ``(T, n)``
    observations, and the task's own observations as ``task.<name>``.

    Attributes:
        task: The ``domain-task`` name.
        spec: Names and kinds of the state dimensions.
        episodes: The episodes, in memory or as an :class:`EpisodeFolder`.
        camera: The camera the frames were rendered from.
        action_repeat: Simulator steps per recorded frame.
    """

    task: str
    spec: PhysicsStateSpec
    episodes: Sequence[Episode]
    camera: int = 0
    action_repeat: int = 1

    @property
    def domain(self) -> str:
        domain = self.task.split("-", 1)[0]
        return "ball_in_cup" if domain == "cup" else domain

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:  # type: ignore[override]
        return self.episodes[index]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def split(
        self, val_fraction: float = 0.2, seed: int = 0
    ) -> tuple["ProbeDataset", "ProbeDataset"]:
        """Disjoint train / validation datasets, split by whole episodes.

        Neighbouring frames are near duplicates, so a split by frame would put
        almost the same image on both sides and overstate every probe.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        if len(self) < 2:
            raise ValueError("Splitting needs at least two episodes.")
        order = np.random.default_rng(seed).permutation(len(self))
        n_val = min(max(1, int(round(len(self) * val_fraction))), len(self) - 1)
        val, train = sorted(order[:n_val].tolist()), sorted(order[n_val:].tolist())
        return self._subset(train), self._subset(val)

    def _subset(self, indices: list[int]) -> "ProbeDataset":
        if isinstance(self.episodes, EpisodeFolder):
            episodes: Sequence[Episode] = self.episodes._subset(indices)
        else:
            episodes = [self.episodes[i] for i in indices]
        return ProbeDataset(
            self.task, self.spec, episodes, self.camera, self.action_repeat
        )

    def targets(
        self,
        include: Iterable[str] = ("qpos", "qvel"),
        *,
        unobservable: Union[str, Iterable[str], None] = "auto",
        wrap_angles: bool = True,
    ) -> StateTargets:
        """The regression targets for every frame.

        Args:
            include: Which groups to use, in order: ``"qpos"``, ``"qvel"``,
                ``"act"`` and ``"task"`` (the task's own observations).
            unobservable: ``qpos`` names to leave out. ``"auto"`` drops the
                horizontal root position under a tracking camera (see
                :func:`unobservable_dims`); ``None`` keeps everything.
            wrap_angles: Replace each hinge angle by its ``sin`` and ``cos``,
                which are continuous where the raw angle wraps around.
        """
        include = tuple(include)
        unknown = set(include) - {*STATE_KEYS, "task"}
        if unknown:
            raise ValueError(
                f"Unknown target groups {sorted(unknown)}; use qpos, qvel, act or task."
            )
        if unobservable == "auto":
            skip = set(unobservable_dims(self.domain, self.camera))
        else:
            skip = set(unobservable or ())
        missing = skip - set(self.spec.qpos_names)
        if missing:
            raise ValueError(
                f"{sorted(missing)} are not qpos names of {self.task}: "
                f"{list(self.spec.qpos_names)}."
            )

        columns: list[Callable[[Episode], np.ndarray]] = []
        names: list[str] = []
        for group in include:
            if group == "qpos":
                self._qpos_columns(skip, wrap_angles, columns, names)
            elif group == "task":
                for key in sorted(self._task_keys()):
                    width = int(np.prod(self[0].observations[key].shape[1:]))
                    columns.append(_flat(key))
                    label = "task/" + key[len(TASK_PREFIX) :]
                    names.extend(
                        [label]
                        if width == 1
                        else [f"{label}/{i}" for i in range(width)]
                    )
            else:
                labels = (
                    self.spec.qvel_names if group == "qvel" else self.spec.act_names
                )
                columns.append(_flat(group))
                names.extend(f"{group}/{label}" for label in labels)

        values, episode, step = [], [], []
        for index, ep in enumerate(self):
            parts = [column(ep) for column in columns]
            values.append(
                np.concatenate(parts, axis=1) if parts else np.zeros((len(ep), 0))
            )
            episode.append(np.full(len(ep), index, dtype=np.int64))
            step.append(np.arange(len(ep), dtype=np.int64))
        stacked = np.concatenate(values).astype(np.float32)
        if stacked.shape[1] != len(names):
            raise ValueError(
                f"Targets have {stacked.shape[1]} columns for {len(names)} names; "
                "the episodes do not match this dataset's spec."
            )
        return StateTargets(
            stacked, tuple(names), np.concatenate(episode), np.concatenate(step)
        )

    def _qpos_columns(
        self,
        skip: set[str],
        wrap_angles: bool,
        columns: list[Callable[[Episode], np.ndarray]],
        names: list[str],
    ) -> None:
        kinds = self.spec.qpos_kinds
        for index, (name, kind) in enumerate(zip(self.spec.qpos_names, kinds)):
            if name in skip:
                continue
            if kind == "hinge" and wrap_angles:
                columns.append(_hinge(index))
                names.extend((f"qpos/{name}/sin", f"qpos/{name}/cos"))
            elif kind.endswith("_quat") and name.endswith("/qw"):
                # q and -q are the same rotation: keep the half with qw >= 0.
                columns.append(_quaternion(index))
                names.extend(
                    f"qpos/{q}" for q in self.spec.qpos_names[index : index + 4]
                )
            elif not kind.endswith("_quat"):
                columns.append(_column("qpos", index))
                names.append(f"qpos/{name}")

    def _task_keys(self) -> list[str]:
        if not len(self):
            return []
        return [k for k in self[0].observations if k.startswith(TASK_PREFIX)]

    def save(self, root: str | Path, *, compress: bool = True) -> Path:
        """Write the episodes as an episode folder that :meth:`load` reads."""
        meta = DatasetMeta(
            source=SOURCE_PREFIX + self.task,
            info={
                "task": self.task,
                "camera": self.camera,
                "action_repeat": self.action_repeat,
                "physics_state_spec": asdict(self.spec),
            },
        )
        with EpisodeWriter(root, meta=meta, compress=compress) as writer:
            for episode in self:
                writer.write(episode)
        return Path(root)

    @classmethod
    def load(cls, root: str | Path) -> "ProbeDataset":
        """Open a folder written by :meth:`save`; episodes load on access."""
        folder = EpisodeFolder(root)
        info = folder.meta.info if folder.meta is not None else {}
        if "physics_state_spec" not in info:
            raise ValueError(
                f"{root} has no physics state spec in meta.json; it was not "
                "written by ProbeDataset.save."
            )
        spec = PhysicsStateSpec(
            **{k: tuple(v) for k, v in info["physics_state_spec"].items()}
        )
        return cls(
            task=info["task"],
            spec=spec,
            episodes=folder,
            camera=int(info.get("camera", 0)),
            action_repeat=int(info.get("action_repeat", 1)),
        )


def collect_probe_episodes(
    task: str,
    episodes: int,
    *,
    policy: Union[str, Policy] = "random",
    seed: int = 0,
    size: tuple[int, int] = (64, 64),
    camera: Optional[int] = None,
    action_repeat: int = 1,
    max_steps: Optional[int] = None,
) -> ProbeDataset:
    """Play ``episodes`` episodes of a DMC task and record the true state.

    Args:
        task: A ``domain-task`` name, such as ``"cartpole-swingup"``.
        episodes: How many episodes to record.
        policy: ``"random"`` for uniform random actions, or a callable from
            the observation dict to an action. Random actions cover little of
            the state space in some tasks (a swing-up pole rarely gets up), so
            mixing in a trained agent's episodes gives a fairer probe.
        seed: Seeds the simulator and the random policy.
        size: Frame ``(height, width)``; match the world model's input.
        camera: Camera id; the environment's default when ``None``.
        action_repeat: Simulator steps per recorded frame. Match the world
            model's training setup (Dreamer uses 2 on DMC).
        max_steps: Cut each episode after this many recorded steps.
    """
    from simulusion.envs.dmc import DeepMindControlEnv

    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if action_repeat < 1:
        raise ValueError("action_repeat must be at least 1.")
    env = DeepMindControlEnv(task, seed=seed, size=size, camera=camera)
    if policy == "random":
        space = env.action_space
        space.seed(seed)
        act: Policy = lambda obs: space.sample()  # noqa: E731
    elif callable(policy):
        act = policy
    else:
        raise ValueError(f"policy must be 'random' or a callable, got {policy!r}.")

    recorded = [
        _record_episode(env, act, action_repeat, max_steps) for _ in range(episodes)
    ]
    return ProbeDataset(
        task=task,
        spec=env.physics_state_spec(),
        episodes=recorded,
        camera=int(env._camera),
        action_repeat=action_repeat,
    )


def _record_episode(
    env: Any, act: Policy, action_repeat: int, max_steps: Optional[int]
) -> Episode:
    obs = env.reset()
    frames, states, task_obs = [_frame(obs)], [env.physics_state()], [_task(obs)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    done = False
    while not done and (max_steps is None or len(actions) < max_steps):
        action = np.asarray(act(obs), dtype=np.float32)
        reward, info = 0.0, {}
        for _ in range(action_repeat):
            obs, step_reward, done, info = env.step(action)
            reward += step_reward
            if done:
                break
        frames.append(_frame(obs))
        states.append(info["physics_state"])
        task_obs.append(_task(obs))
        actions.append(action)
        rewards.append(reward)
        terminated.append(bool(info.get("terminated", done)))
        truncated.append(bool(info.get("truncated", False)))

    observations = {key: np.stack([s[key] for s in states]) for key in STATE_KEYS}
    observations.update(
        {key: np.stack([o[key] for o in task_obs]) for key in task_obs[0]}
    )
    return Episode(
        frames=np.stack(frames),
        actions=np.stack(actions) if actions else None,
        rewards=np.asarray(rewards, dtype=np.float32) if actions else None,
        terminated=np.asarray(terminated) if actions else None,
        truncated=np.asarray(truncated) if actions else None,
        observations=observations,
    )


def _frame(obs: dict) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs["image"]).transpose(1, 2, 0))


def _task(obs: dict) -> dict[str, np.ndarray]:
    return {
        TASK_PREFIX + key: np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in obs.items()
        if key != "image"
    }


def _flat(key: str) -> Callable[[Episode], np.ndarray]:
    def column(ep: Episode) -> np.ndarray:
        value = ep.observations[key]
        return value.reshape(len(value), -1)

    return column


def _column(key: str, index: int) -> Callable[[Episode], np.ndarray]:
    return lambda ep: ep.observations[key][:, index : index + 1]


def _hinge(index: int) -> Callable[[Episode], np.ndarray]:
    def column(ep: Episode) -> np.ndarray:
        angle = ep.observations["qpos"][:, index : index + 1]
        return np.concatenate([np.sin(angle), np.cos(angle)], axis=1)

    return column


def _quaternion(index: int) -> Callable[[Episode], np.ndarray]:
    def column(ep: Episode) -> np.ndarray:
        quat = ep.observations["qpos"][:, index : index + 4]
        return np.where(quat[:, :1] < 0, -quat, quat)

    return column
