"""Minecraft Diamond: the DreamerV3 benchmark environment.

DreamerV3 (Hafner et al., 2023) was the first agent to collect diamonds in
Minecraft from scratch. It did so on an environment of its own, built on
MineRL 0.4.4 (Minecraft 1.11.2), that fixes problems of the competition task
(paper, "Minecraft" appendix). This module reproduces that environment
(``embodied/envs/minecraft_flat.py`` of the reference implementation):

* **Actions** -- a flat categorical space of 25 actions: 12 basic ones
  (movement, camera, attack, jump, place dirt) plus 13 abstract crafting,
  placing, equipping and smelting actions (:data:`DIAMOND_ACTIONS`). Camera
  moves 15 degrees; pitch is kept within +-60 degrees; ``jump`` also moves
  forward and is held for 10 steps, since a single-step press does not jump.
* **Block breaking** -- blocks break 100x faster (``break_speed=100``, the
  "block breaking setting of prior work"), which needs the patched MineRL
  wheel used by the reference (see :func:`make_minerl_backend`). With a
  speed-up the attack key is not held (``sticky_attack`` is disabled).
* **Observations** -- a 64x64 RGB image plus vectors: inventory counts,
  the maximum inventory counts so far this episode, a one-hot of the equipped
  item, and health, hunger and breath scaled to [0, 1].
* **Rewards** -- +1 the first time in an episode each of the 12 milestone
  items is obtained (:data:`DIAMOND_MILESTONES`), plus 0.01 times the change
  of health (health in [0, 1]), as in the reference code. (The paper text
  describes this as +-0.01 per heart; the code, which produced the results,
  scales by health in [0, 1], so a heart is worth 0.001.)
* **Episodes** end when the player dies or after 36000 steps (30 minutes at
  20 Hz); breaking diamond ore does not end the episode.

MineRL 0.4.4 needs Linux, Python <= 3.11 and a JDK 8; the ``docker/envs``
image installs it. Everything except the backend is plain Python, and the
adapter accepts any backend with the MineRL 0.4.4 interface through ``env=``.

Reference:
    Mastering Diverse Domains through World Models
    Hafner et al., 2023 - https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info

__all__ = [
    "BASIC_ACTIONS",
    "DIAMOND_ACTIONS",
    "DIAMOND_MILESTONES",
    "MINECRAFT_DIAMOND_TASKS",
    "MINERL_PATCHED_WHEEL",
    "MinecraftDiamondEnv",
    "make_minecraft_diamond_env",
    "make_minerl_backend",
]

# The patched MineRL 0.4.4 the reference installs: it adds the
# BreakSpeedMultiplier mission handler. CPython 3.11, Linux x86-64 only.
MINERL_PATCHED_WHEEL = (
    "https://github.com/danijar/minerl/releases/download/v0.4.4-patched/"
    "minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl"
)

CAMERA_STEP = 15.0  # degrees per camera action

# name -> native action fields; omitted fields take their NOOP value.
BASIC_ACTIONS: dict[str, dict[str, Any]] = {
    "noop": {},
    "attack": {"attack": 1},
    "turn_up": {"camera": (-CAMERA_STEP, 0.0)},
    "turn_down": {"camera": (CAMERA_STEP, 0.0)},
    "turn_left": {"camera": (0.0, -CAMERA_STEP)},
    "turn_right": {"camera": (0.0, CAMERA_STEP)},
    "forward": {"forward": 1},
    "back": {"back": 1},
    "left": {"left": 1},
    "right": {"right": 1},
    "jump": {"jump": 1, "forward": 1},
    "place_dirt": {"place": "dirt"},
}

DIAMOND_ACTIONS: dict[str, dict[str, Any]] = {
    **BASIC_ACTIONS,
    "craft_planks": {"craft": "planks"},
    "craft_stick": {"craft": "stick"},
    "craft_crafting_table": {"craft": "crafting_table"},
    "place_crafting_table": {"place": "crafting_table"},
    "craft_wooden_pickaxe": {"nearbyCraft": "wooden_pickaxe"},
    "craft_stone_pickaxe": {"nearbyCraft": "stone_pickaxe"},
    "craft_iron_pickaxe": {"nearbyCraft": "iron_pickaxe"},
    "equip_stone_pickaxe": {"equip": "stone_pickaxe"},
    "equip_wooden_pickaxe": {"equip": "wooden_pickaxe"},
    "equip_iron_pickaxe": {"equip": "iron_pickaxe"},
    "craft_furnace": {"nearbyCraft": "furnace"},
    "place_furnace": {"place": "furnace"},
    "smelt_iron_ingot": {"nearbySmelt": "iron_ingot"},
}

# The 12 items rewarded once per episode, in technology-tree order.
DIAMOND_MILESTONES = (
    "log",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "stone_pickaxe",
    "iron_ore",
    "furnace",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
)

# "diamond" is the benchmark; "wood" (repeated reward per log, basic actions)
# is the reference's quick sanity task.
MINECRAFT_DIAMOND_TASKS = ("diamond", "wood")

NOOP: dict[str, Any] = {
    "camera": (0.0, 0.0),
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "attack": 0,
    "sprint": 0,
    "jump": 0,
    "sneak": 0,
    "craft": "none",
    "nearbyCraft": "none",
    "nearbySmelt": "none",
    "place": "none",
    "equip": "none",
}

DEFAULT_LENGTH = 36_000


class _CollectReward:
    """``once`` the first time ``item`` is held, ``repeated`` per new unit."""

    def __init__(self, item: str, once: float = 0.0, repeated: float = 0.0):
        self.item = item
        self.once = float(once)
        self.repeated = float(repeated)
        self.previous = 0.0
        self.maximum = 0.0

    def __call__(self, inventory: Mapping[str, float], is_first: bool) -> float:
        current = float(inventory.get(self.item, 0.0))
        if is_first:
            self.previous = current
            self.maximum = current
            return 0.0
        reward = self.repeated * max(0.0, current - self.previous)
        if self.maximum == 0 and current > 0:
            reward += self.once
        self.previous = current
        self.maximum = max(self.maximum, current)
        return reward


class _HealthReward:
    def __init__(self, scale: float = 0.01):
        self.scale = float(scale)
        self.previous = 0.0

    def __call__(self, health: float, is_first: bool) -> float:
        if is_first:
            self.previous = health
            return 0.0
        reward = self.scale * (health - self.previous)
        self.previous = health
        return reward


class _OneHotActionSpace(gym.spaces.Box):
    """Box of one-hot vectors in ``[-1, 1]``; sampling draws a random action."""

    one_hot = True  # a one-hot discrete choice (categorical, not Gaussian)

    def __init__(self, n: int, seed: int | None = None):
        self.n = int(n)
        self._rng = np.random.default_rng(seed)
        super().__init__(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def seed(self, seed: int | None = None) -> list[int]:
        self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def sample(self, mask: Any = None, probability: Any = None) -> np.ndarray:
        del mask, probability
        action = np.full((self.n,), -1.0, dtype=np.float32)
        action[int(self._rng.integers(self.n))] = 1.0
        return action


def _flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a/b": 1}``, for observations and spaces."""
    children = getattr(tree, "spaces", None)
    if children is None and isinstance(tree, Mapping):
        children = tree
    if children is None:
        return {prefix: tree}
    flat: dict[str, Any] = {}
    for key, value in dict(children).items():
        flat.update(_flatten(value, f"{prefix}/{key}" if prefix else str(key)))
    return flat


def _enum_values(space: Any) -> list[str]:
    values = getattr(space, "values", None)
    if values is None:
        raise ValueError(
            "The backend's equipped_items/mainhand/type space has no enum values."
        )
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def make_minerl_backend(
    size: tuple[int, int] = (64, 64), break_speed: float = 100.0
) -> Any:
    """Build the reference's custom MineRL 0.4.4 environment.

    A ``minerl.herobraine`` :class:`EnvSpec` with the full flat inventory, the
    equipped item, life stats and position as observations; every keyboard,
    camera and abstract crafting action; default world generation; and the
    block break speed multiplier, which only the patched wheel
    (:data:`MINERL_PATCHED_WHEEL`) understands.
    """
    if importlib.util.find_spec("minerl") is None:
        raise ImportError(
            "Minecraft Diamond needs MineRL 0.4.4 with the break-speed patch, "
            "which runs on Linux with Python 3.11 and a JDK 8. Use the "
            "docker/envs image, or install it with "
            f"`pip install {MINERL_PATCHED_WHEEL}` (and openjdk-8-jdk, xvfb)."
        )
    # MineRL 0.4.4 still uses the NumPy aliases removed in NumPy 1.24.
    for alias, builtin in (("float", float), ("int", int), ("bool", bool)):
        if alias not in np.__dict__:
            setattr(np, alias, builtin)
    env_spec = importlib.import_module("minerl.herobraine.env_spec")
    hero = importlib.import_module("minerl.herobraine.hero")
    handler = importlib.import_module("minerl.herobraine.hero.handler")
    handlers = hero.handlers
    mc = hero.mc
    keymap = mc.INVERSE_KEYMAP

    class BreakSpeedMultiplier(handler.Handler):  # type: ignore[name-defined]
        def __init__(self, multiplier: float = 1.0):
            self.multiplier = multiplier

        def to_string(self) -> str:
            return f"break_speed({self.multiplier})"

        def xml_template(self) -> str:
            return "<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>"

    class DiamondSpec(env_spec.EnvSpec):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__(name="MineRLEnv-v1")

        def create_agent_start(self) -> list[Any]:
            return [BreakSpeedMultiplier(break_speed)]

        def create_agent_handlers(self) -> list[Any]:
            return []

        def create_server_world_generators(self) -> list[Any]:
            return [handlers.DefaultWorldGenerator(force_reset=True)]

        def create_server_quit_producers(self) -> list[Any]:
            return [handlers.ServerQuitWhenAnyAgentFinishes()]

        def create_server_initial_conditions(self) -> list[Any]:
            return [
                handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=0),
                handlers.SpawningInitialCondition(allow_spawning=True),
            ]

        def create_observables(self) -> list[Any]:
            return [
                handlers.POVObservation(tuple(size)),
                handlers.FlatInventoryObservation(mc.ALL_ITEMS),
                handlers.EquippedItemObservation(
                    mc.ALL_ITEMS, _default="air", _other="other"
                ),
                handlers.ObservationFromCurrentLocation(),
                handlers.ObservationFromLifeStats(),
            ]

        def create_actionables(self) -> list[Any]:
            kw = dict(_other="none", _default="none")
            items = ["none"] + list(mc.ALL_ITEMS)
            return [
                handlers.KeybasedCommandAction("forward", keymap["forward"]),
                handlers.KeybasedCommandAction("back", keymap["back"]),
                handlers.KeybasedCommandAction("left", keymap["left"]),
                handlers.KeybasedCommandAction("right", keymap["right"]),
                handlers.KeybasedCommandAction("jump", keymap["jump"]),
                handlers.KeybasedCommandAction("sneak", keymap["sneak"]),
                handlers.KeybasedCommandAction("attack", keymap["attack"]),
                handlers.CameraAction(),
                handlers.PlaceBlock(items, **kw),
                handlers.EquipAction(items, **kw),
                handlers.CraftAction(items, **kw),
                handlers.CraftNearbyAction(items, **kw),
                handlers.SmeltItemNearby(items, **kw),
            ]

        def is_from_folder(self, folder: str) -> bool:
            return folder == "none"

        def get_docstring(self) -> str:
            return ""

        def determine_success_from_rewards(self, rewards: Any) -> bool:
            return True

        def create_rewardables(self) -> list[Any]:
            return []

        def create_server_decorators(self) -> list[Any]:
            return []

        def create_mission_handlers(self) -> list[Any]:
            return []

        def create_monitors(self) -> list[Any]:
            return []

    return DiamondSpec().make()


class MinecraftDiamondEnv:
    """DreamerV3's Minecraft Diamond task with Simulusion's env interface.

    ``reset()`` returns an observation dict and ``step(action)`` returns
    ``(obs, reward, done, info)``, like the other Dreamer backends. Actions are
    one-hot vectors (or integer indices) over :attr:`action_names`.

    Observations: ``image`` ``(3, H, W)`` uint8; ``inventory``,
    ``inventory_max`` ``(num_items,)``; ``equipped`` ``(num_equip_types,)``;
    ``health``, ``hunger``, ``breath`` ``(1,)``, all float32.

    Args:
        task: ``"diamond"`` (the benchmark) or ``"wood"``.
        seed: Seeds action sampling; worlds are generated by Minecraft.
        size: Image size ``(H, W)``. The backend renders at this resolution.
        break_speed: Block breaking speed multiplier (100 in the paper).
        sticky_attack: Steps to hold attack after an attack action; ignored
            (0) when ``break_speed != 1``, as in the reference.
        sticky_jump: Steps to hold jump (+ forward) after a jump action.
        pitch_limit: Camera pitch bounds in degrees, or ``None``.
        length: Episode step limit (36000 = 30 minutes at 20 Hz).
        env: A ready backend with the MineRL 0.4.4 interface; built with
            :func:`make_minerl_backend` when omitted.
    """

    def __init__(
        self,
        task: str = "diamond",
        seed: int = 0,
        size: tuple[int, int] = (64, 64),
        break_speed: float = 100.0,
        sticky_attack: int = 30,
        sticky_jump: int = 10,
        pitch_limit: tuple[float, float] | None = (-60.0, 60.0),
        length: int | None = DEFAULT_LENGTH,
        env: Any | None = None,
    ):
        task = str(task).lower().removeprefix("minecraft_").removeprefix("minecraft-")
        if task not in MINECRAFT_DIAMOND_TASKS:
            raise ValueError(
                f"Unknown Minecraft task {task!r}; options: "
                f"{', '.join(MINECRAFT_DIAMOND_TASKS)}."
            )
        self.task = task
        self._size = (int(size[0]), int(size[1]))
        self._length = None if length is None else int(length)
        self._env = (
            env if env is not None else make_minerl_backend(self._size, break_speed)
        )

        actions = DIAMOND_ACTIONS if task == "diamond" else BASIC_ACTIONS
        self.action_names = tuple(actions)
        self._action_values = tuple({**NOOP, **value} for value in actions.values())
        self._sticky_attack_length = (
            0 if float(break_speed) != 1.0 else int(sticky_attack)
        )
        self._sticky_jump_length = int(sticky_jump)
        self._pitch_limit = pitch_limit

        flat_space = _flatten(self._env.observation_space)
        self._inv_keys = [
            key
            for key in flat_space
            if key.startswith("inventory/") and key != "inventory/log2"
        ]
        if not self._inv_keys:
            raise ValueError("The Minecraft backend observes no inventory.")
        self._equip_enum = _enum_values(flat_space["equipped_items/mainhand/type"])
        native_action_space = getattr(self._env, "action_space", None)
        native_keys = _flatten(native_action_space) if native_action_space else {}
        # Send only the fields the backend declares (it has no "sprint").
        self._native_keys = set(native_keys) or set(NOOP)

        self._action_space = _OneHotActionSpace(len(self.action_names), seed=seed)
        n_inv, n_equip = len(self._inv_keys), len(self._equip_enum)

        def vector(n: int, high: float = np.inf) -> gym.spaces.Box:
            return gym.spaces.Box(0.0, high, (n,), dtype=np.float32)

        self._observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255, (3, self._size[0], self._size[1]), dtype=np.uint8
                ),
                "inventory": vector(n_inv),
                "inventory_max": vector(n_inv),
                "equipped": vector(n_equip, 1.0),
                "health": vector(1, 1.0),
                "hunger": vector(1, 1.0),
                "breath": vector(1, 1.0),
            }
        )
        if task == "diamond":
            self._rewards = [
                _CollectReward(item, once=1) for item in DIAMOND_MILESTONES
            ]
        else:
            self._rewards = [_CollectReward("log", repeated=1)]
        self._health_reward = _HealthReward()
        self._reset_counters()
        self._last_obs: dict[str, np.ndarray] | None = None

    # -- Gym-style interface -------------------------------------------------

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> _OneHotActionSpace:
        return self._action_space

    @property
    def discrete_actions(self) -> bool:
        """Minecraft actions are discrete; the Box space holds their one-hot form."""
        return True

    @property
    def max_episode_steps(self) -> int:
        return self._length or DEFAULT_LENGTH

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._action_space.seed(seed)
        raw = self._env.reset()
        if isinstance(raw, tuple) and len(raw) == 2:
            raw = raw[0]
        self._reset_counters()
        obs, _ = self._process(raw, is_first=True)
        return obs

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        index, one_hot = encode_discrete_action(action, len(self.action_names))
        native = self._apply_sticky(dict(self._action_values[index]))
        result = self._env.step(self._to_backend(native))
        if len(result) == 5:
            raw, _, terminated, truncated, info = result
            backend_done = bool(terminated) or bool(truncated)
        else:
            raw, _, backend_done, info = result
        info = dict(info or {})
        self._step += 1
        if "error" in info:
            # Malmo failed. The reference discards this step and resets in
            # place, so the episode stops there without a terminal state and
            # without reward. Here the episode is cut short (truncated, so it
            # is bootstrapped) on the last good observation; the next reset()
            # restarts the backend.
            if self._last_obs is None:
                raise RuntimeError("Call reset() before step().")
            obs = {key: value.copy() for key, value in self._last_obs.items()}
            info = finalize_step_info(
                {
                    **{k: v for k, v in info.items() if k != "error"},
                    "backend_error": str(info["error"]),
                    "action": one_hot,
                    "action_name": self.action_names[index],
                    "achievements": self.achievements(),
                },
                done=True,
                terminated=False,
                truncated=True,
            )
            return obs, 0.0, True, info
        obs, reward = self._process(raw, is_first=False)
        timeout = self._length is not None and self._step >= self._length
        done = bool(backend_done or timeout)
        info = finalize_step_info(
            {
                **info,
                "action": one_hot,
                "action_name": self.action_names[index],
                "achievements": self.achievements(),
            },
            done=done,
            terminated=bool(backend_done),
            truncated=bool(timeout and not backend_done),
        )
        return obs, reward, done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        if self._last_obs is None:
            return np.zeros((self._size[0], self._size[1], 3), dtype=np.uint8)
        return self._last_obs["image"].transpose(1, 2, 0).copy()

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def achievements(self) -> dict[str, bool]:
        """Which milestone items were obtained this episode."""
        return {
            item: self._inventory_max.get(item, 0.0) > 0 for item in DIAMOND_MILESTONES
        }

    # -- internals -------------------------------------------------------------

    def _reset_counters(self) -> None:
        self._step = 0
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0.0
        self._max_inventory: np.ndarray | None = None
        self._inventory_max: dict[str, float] = {}

    def _apply_sticky(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = 1
                action["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = 1
                action["forward"] = 1
                self._sticky_jump_counter -= 1
        pitch = float(action["camera"][0])
        if self._pitch_limit and pitch:
            low, high = self._pitch_limit
            if not low <= self._pitch + pitch <= high:
                action["camera"] = (0.0, action["camera"][1])
                pitch = 0.0
            self._pitch += pitch
        return action

    def _to_backend(self, action: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in action.items():
            if key not in self._native_keys:
                continue
            out[key] = np.asarray(value, dtype=np.float32) if key == "camera" else value
        return out

    def _process(self, raw: Any, is_first: bool) -> tuple[dict[str, np.ndarray], float]:
        flat = _flatten(raw)
        if "inventory/log2" in flat and "inventory/log" in flat:
            flat["inventory/log"] = flat["inventory/log"] + flat["inventory/log2"]
        inventory = np.array(
            [
                float(np.asarray(flat.get(key, 0)).reshape(-1)[0])
                for key in self._inv_keys
            ],
            dtype=np.float32,
        )
        if self._max_inventory is None:
            self._max_inventory = inventory.copy()
        else:
            self._max_inventory = np.maximum(self._max_inventory, inventory)
        items = {
            key.split("/", 1)[1]: float(value)
            for key, value in zip(self._inv_keys, inventory)
            if key != "inventory/air"
        }
        self._inventory_max = {
            key.split("/", 1)[1]: float(value)
            for key, value in zip(self._inv_keys, self._max_inventory)
        }
        equipped = np.zeros(len(self._equip_enum), dtype=np.float32)
        held = str(np.asarray(flat.get("equipped_items/mainhand/type", "air")).item())
        equipped[self._equip_enum.index(held) if held in self._equip_enum else 0] = 1.0

        def stat(key: str, scale: float) -> np.ndarray:
            value = float(np.asarray(flat.get(key, 0.0)).reshape(-1)[0]) / scale
            return np.array([value], dtype=np.float32)

        health = stat("life_stats/life", 20.0)
        obs = {
            "image": self._image(flat["pov"]),
            "inventory": inventory,
            "inventory_max": self._max_inventory.copy(),
            "equipped": equipped,
            "health": health,
            "hunger": stat("life_stats/food", 20.0),
            "breath": stat("life_stats/air", 300.0),
        }
        reward = sum(fn(items, is_first) for fn in self._rewards)
        reward += self._health_reward(float(health[0]), is_first)
        self._last_obs = obs
        return obs, float(reward)

    def _image(self, pov: Any) -> np.ndarray:
        image = np.asarray(pov, dtype=np.uint8)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.shape[:2] != self._size:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self._size[1], self._size[0]), Image.Resampling.BILINEAR
                )
            )
        return np.ascontiguousarray(image.transpose(2, 0, 1))


def make_minecraft_diamond_env(
    task: str = "diamond", **kwargs: Any
) -> MinecraftDiamondEnv:
    """Create DreamerV3's Minecraft Diamond (or Wood) environment."""
    return MinecraftDiamondEnv(task=task, **kwargs)
