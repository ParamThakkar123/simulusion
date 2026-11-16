"""Atari 100k with the exact preprocessing of IRIS, DIAMOND, STORM and DreamerV3.

Atari 100k (Kaiser et al., 2020) trains on 26 games for 100k agent steps
(400k frames). Papers that report on it agree on the game list and on a frame
skip of 4, but not on the rest of the preprocessing, and the differences move
scores. This module drives ALE directly, as the DreamerV3 reference does, and
offers each paper's settings as a named protocol, taken from the reference
implementations:

==================  ======  ==========  =============  ============  ==============
Setting             IRIS    DIAMOND     STORM          DreamerV3     Source note
==================  ======  ==========  =============  ============  ==============
sticky actions      0       0           0.25           0             STORM keeps the
                                                                     ALE v5 default
no-op starts        1-30    1-30        none           0-30
life loss           ends    ends        ends           ignored       "ends": episode
                    episode episode     episode                      ends, game goes on
frame limit         20000   108000      108000         108000        IRIS: 108000 in
                                                                     evaluation
resize              pillow  area        area           pillow
                    before
                    pooling
==================  ======  ==========  =============  ============  ==============

All four act on ALE's minimal action set, max-pool the last two frames of each
4-frame step, keep RGB at 64x64 and leave rewards unclipped in the environment
(IRIS and DIAMOND clip, or sign, rewards inside their training code).

The ``*-eval`` protocols are the evaluation settings of IRIS and DIAMOND: life
loss does not end the episode, and IRIS evaluates with 108000 frames and a
single no-op.

``life_loss="terminate"`` follows the OpenAI Baselines ``EpisodicLifeEnv`` these
codebases use: losing a life returns ``done=True`` with ``terminated=True`` and
``info["life_lost"] = True``, but the next :meth:`Atari100kEnv.reset` continues
the same game rather than starting a new one. Only a real game over, or the
frame limit, resets the emulator. ``info["game_over"]`` tells the two apart, and
``info["episode_frame_number"]`` counts frames since the last real reset.

DreamerV3's reference marks frame-limit endings as terminal; this adapter
reports them as truncations, consistent with every other Simulusion backend.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from simulusion.envs._actions import encode_discrete_action
from simulusion.envs._contract import finalize_step_info

#: The 26 Atari 100k games, as ALE ROM ids.
ATARI_100K_ROMS = (
    "alien",
    "amidar",
    "assault",
    "asterix",
    "bank_heist",
    "battle_zone",
    "boxing",
    "breakout",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "private_eye",
    "qbert",
    "road_runner",
    "seaquest",
    "up_n_down",
)

LIFE_LOSS_MODES = ("ignore", "terminate")
RESIZE_MODES = ("pillow", "area")


@dataclass(frozen=True)
class AtariProtocol:
    """Preprocessing settings of one Atari benchmark protocol.

    Attributes:
        sticky: ``repeat_action_probability``: chance that ALE repeats the
            previous action instead of the chosen one, each frame.
        frame_skip: Frames per agent step; the action repeats on each.
        pool_last: Frames at the end of a step that are max-pooled into the
            observation (2 removes Atari sprite flicker).
        noop_min: Fewest NOOP frames taken after a reset.
        noop_max: Most NOOP frames taken after a reset; the count is uniform in
            ``[noop_min, noop_max]``.
        life_loss: ``"terminate"`` ends the episode on a lost life without
            resetting the game; ``"ignore"`` does nothing.
        continue_noop: After a life-loss episode end, take one NOOP agent step
            before returning the next first observation (Baselines'
            ``EpisodicLifeEnv`` does this; DIAMOND and STORM do not).
        max_frames: Frames before the game is cut off as a truncation.
        resize: ``"pillow"`` (PIL bilinear) or ``"area"`` (OpenCV INTER_AREA).
        resize_before_pool: Resize each frame before max-pooling (IRIS) rather
            than pooling the full-resolution frames first.
        size: Default observation size ``(height, width)``.
    """

    sticky: float = 0.0
    frame_skip: int = 4
    pool_last: int = 2
    noop_min: int = 0
    noop_max: int = 30
    life_loss: str = "ignore"
    continue_noop: bool = False
    max_frames: int = 108000
    resize: str = "pillow"
    resize_before_pool: bool = False
    size: tuple[int, int] = (64, 64)

    def __post_init__(self) -> None:
        if not 0.0 <= self.sticky <= 1.0:
            raise ValueError(f"sticky must lie in [0, 1], got {self.sticky}.")
        if self.frame_skip < 1 or not 1 <= self.pool_last <= self.frame_skip:
            raise ValueError("Need frame_skip >= 1 and 1 <= pool_last <= frame_skip.")
        if not 0 <= self.noop_min <= self.noop_max:
            raise ValueError("Need 0 <= noop_min <= noop_max.")
        if self.life_loss not in LIFE_LOSS_MODES:
            raise ValueError(f"life_loss must be one of {LIFE_LOSS_MODES}.")
        if self.resize not in RESIZE_MODES:
            raise ValueError(f"resize must be one of {RESIZE_MODES}.")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive.")


#: Named protocols. ``*-eval`` variants are the papers' evaluation settings.
ATARI_100K_PROTOCOLS: dict[str, AtariProtocol] = {
    # eloialonso/iris config/env/default.yaml (train) and envs/wrappers.py: a
    # v4 NoFrameskip env, TimeLimit of 20000 frames, NoopReset in [1, 30],
    # MaxAndSkip(4) over frames already resized with PIL, EpisodicLifeEnv.
    "iris": AtariProtocol(
        noop_min=1,
        noop_max=30,
        life_loss="terminate",
        continue_noop=True,
        max_frames=20000,
        resize="pillow",
        resize_before_pool=True,
    ),
    # Same file, test split: 108000 frames, one no-op, no life-loss ends.
    "iris-eval": AtariProtocol(
        noop_min=1,
        noop_max=1,
        life_loss="ignore",
        max_frames=108000,
        resize="pillow",
        resize_before_pool=True,
    ),
    # eloialonso/diamond src/envs/env.py and atari_preprocessing.py: v4
    # NoFrameskip (108000-frame cap), noops in [1, 30], max-pool then
    # cv2.INTER_AREA resize, DoneOnLifeLoss without resetting the game.
    "diamond": AtariProtocol(
        noop_min=1,
        noop_max=30,
        life_loss="terminate",
        max_frames=108000,
        resize="area",
    ),
    "diamond-eval": AtariProtocol(
        noop_min=1,
        noop_max=30,
        life_loss="ignore",
        max_frames=108000,
        resize="area",
    ),
    # weipu-zhang/STORM train.py: gymnasium.make("ALE/<game>-v5",
    # frameskip=1), which keeps v5's sticky actions (0.25), then
    # MaxLast2FrameSkip(4), ResizeObservation (INTER_AREA) and LifeLossInfo
    # used as the termination signal. No no-op starts.
    "storm": AtariProtocol(
        sticky=0.25,
        noop_min=0,
        noop_max=0,
        life_loss="terminate",
        max_frames=108000,
        resize="area",
    ),
    # danijar/dreamerv3 configs.yaml env.atari100k and embodied/envs/atari.py:
    # no sticky actions, noops in [0, 30], lives unused, 108000 frames, max
    # over the last two frames, PIL bilinear resize.
    "dreamerv3": AtariProtocol(
        noop_min=0,
        noop_max=30,
        life_loss="ignore",
        max_frames=108000,
        resize="pillow",
    ),
}
ATARI_100K_PROTOCOLS["atari100k"] = ATARI_100K_PROTOCOLS["dreamerv3"]


def list_atari100k_games() -> list[str]:
    """The 26 Atari 100k games as ALE ROM ids (``"ms_pacman"``, ...)."""
    return list(ATARI_100K_ROMS)


def normalize_atari_game(game: str) -> str:
    """Map a game name in any common spelling to its ALE ROM id.

    Accepts ``"MsPacman"``, ``"ALE/MsPacman-v5"``, ``"MsPacmanNoFrameskip-v4"``,
    ``"MsPacman-v5"``, ``"ms_pacman"`` and DreamerV3's ``"atari100k_ms_pacman"``.
    ``"james_bond"`` is accepted for ``"jamesbond"``.
    """
    name = str(game).strip()
    if "/" in name:
        name = name.split("/", 1)[1]
    name = re.sub(r"(NoFrameskip|Deterministic)?-v\d+$", "", name)
    for prefix in ("atari100k_", "atari_"):
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
    if not name.islower():
        # "MsPacman" -> "Ms_Pacman", "UpNDown" -> "Up_N_Down".
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    name = name.lower().replace("-", "_")
    aliases = {"james_bond": "jamesbond", "up_and_down": "up_n_down"}
    name = aliases.get(name, name)
    try:
        from ale_py import roms

        known = set(roms.get_all_rom_ids())
    except Exception:  # noqa: BLE001 - ale_py absent: trust the Atari 100k list
        known = set(ATARI_100K_ROMS)
    if name not in known:
        raise ValueError(f"Unknown Atari game {game!r} (resolved to ROM id {name!r}).")
    return name


def resolve_atari_protocol(
    protocol: str | AtariProtocol, **overrides: Any
) -> AtariProtocol:
    """Look up a named protocol, optionally overriding individual fields."""
    if isinstance(protocol, AtariProtocol):
        base = protocol
    else:
        key = str(protocol).strip().lower().replace("_", "-")
        if key not in ATARI_100K_PROTOCOLS:
            valid = ", ".join(ATARI_100K_PROTOCOLS)
            raise ValueError(f"Unknown Atari protocol {protocol!r}. Valid: {valid}.")
        base = ATARI_100K_PROTOCOLS[key]
    if overrides:
        if "size" in overrides:
            size = overrides["size"]
            overrides["size"] = (int(size[0]), int(size[1]))
        base = replace(base, **overrides)
    return base


def _resize(frame: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    height, width = size
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    if mode == "area":
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "resize='area' (the DIAMOND and STORM protocols) uses OpenCV's "
                "INTER_AREA. Install it with `pip install opencv-python-headless`."
            ) from exc
        return np.asarray(
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        )
    from PIL import Image

    image = Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


class Atari100kEnv:
    """One Atari game under a benchmark protocol, with Simulusion's image contract.

    Observations are ``{"image": (3, H, W) uint8}``; actions are one-hot-like
    over ALE's minimal action set. Rewards are the raw game score change.

    Args:
        game: Game name, see :func:`normalize_atari_game`.
        protocol: A key of :data:`ATARI_100K_PROTOCOLS` (default
            ``"atari100k"``, DreamerV3's settings) or an :class:`AtariProtocol`.
        seed: Seeds ALE (sticky actions) and the no-op draws.
        size: Observation size; defaults to the protocol's (64x64).
        **overrides: Replace individual protocol fields, e.g.
            ``max_frames=27000``.
    """

    def __init__(
        self,
        game: str,
        protocol: str | AtariProtocol = "atari100k",
        seed: int = 0,
        size: tuple[int, int] | None = None,
        **overrides: Any,
    ) -> None:
        try:
            import ale_py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Atari 100k backend requires ale-py. Install it with "
                "`pip install simulusion[gym]`."
            ) from exc
        from simulusion.envs._spaces import OneHotActionSpace, image_space
        from simulusion.utils.gym_compat import gym

        if size is not None:
            overrides["size"] = size
        self.protocol = resolve_atari_protocol(protocol, **overrides)
        self.game = normalize_atari_game(game)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

        self._ale = ale_py.ALEInterface()
        self._ale.setLoggerMode(ale_py.LoggerMode.Error)
        self._ale.setInt("random_seed", int(self._rng.integers(0, 2**31)))
        self._ale.setFloat("repeat_action_probability", float(self.protocol.sticky))
        # The frame limit is enforced here, so it can be reported as a
        # truncation; ALE's own cap would look like a game over.
        self._ale.setInt("max_num_frames_per_episode", 0)
        self._ale.loadROM(_rom_path(self.game))
        self._actions = list(self._ale.getMinimalActionSet())
        self._noop = self._actions.index(ale_py.Action.NOOP)

        height, width = self._ale.getScreenDims()
        self._screen = np.zeros((height, width, 3), np.uint8)
        self._pool: list[np.ndarray] = []
        self._frames = 0
        self._lives = 0
        self._needs_reset = True
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_full_frame: np.ndarray | None = None

        self._action_space = OneHotActionSpace(len(self._actions), seed=self._seed)
        self._observation_space = gym.spaces.Dict(
            {"image": image_space(self.protocol.size)}
        )

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def action_meanings(self) -> list[str]:
        return [str(action).split(".")[-1] for action in self._actions]

    @property
    def max_episode_steps(self) -> int:
        return self.protocol.max_frames // self.protocol.frame_skip

    def _grab(self) -> np.ndarray:
        self._ale.getScreenRGB(self._screen)
        frame = self._screen.copy()
        if self.protocol.resize_before_pool:
            frame = _resize(frame, self.protocol.size, self.protocol.resize)
        return frame

    def _observation(self) -> dict[str, np.ndarray]:
        frame = (
            np.max(np.stack(self._pool), axis=0)
            if len(self._pool) > 1
            else self._pool[0]
        )
        self._last_full_frame = frame
        if not self.protocol.resize_before_pool:
            frame = _resize(frame, self.protocol.size, self.protocol.resize)
        obs = {"image": np.ascontiguousarray(frame.transpose(2, 0, 1))}
        self._last_obs = obs
        return obs

    def _act(self, action: Any) -> float:
        self._frames += 1
        return float(self._ale.act(action))

    def _reset_game(self) -> None:
        self._ale.reset_game()
        self._frames = 0
        noops = int(
            self._rng.integers(self.protocol.noop_min, self.protocol.noop_max + 1)
        )
        for _ in range(noops):
            self._act(self._actions[self._noop])
            if self._ale.game_over():
                self._ale.reset_game()
                self._frames = 0
        self._lives = int(self._ale.lives())
        self._pool = [self._grab()]

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
            self._ale.setInt("random_seed", int(self._rng.integers(0, 2**31)))
            # ALE reads random_seed when a ROM loads.
            self._ale.loadROM(_rom_path(self.game))
            self._action_space.seed(self._seed)
            self._needs_reset = True
        if self._needs_reset or self._last_obs is None:
            self._reset_game()
            self._needs_reset = False
            return self._observation()
        # The previous episode ended on a lost life: carry on with the game.
        if self.protocol.continue_noop:
            obs, _, done, info = self._step_native(self._noop)
            if done and info.get("game_over", False):
                self._reset_game()
                return self._observation()
            return obs
        return self._last_obs

    def _step_native(
        self, index: int
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = self._actions[index]
        reward = 0.0
        game_over = False
        life_lost = False
        truncated = False
        skip = self.protocol.frame_skip
        pool: list[np.ndarray] = []
        for repeat in range(skip):
            reward += self._act(action)
            if repeat >= skip - self.protocol.pool_last:
                pool.append(self._grab())
            game_over = bool(self._ale.game_over())
            lives = int(self._ale.lives())
            if 0 < lives < self._lives:
                life_lost = True
            self._lives = lives
            truncated = self._frames >= self.protocol.max_frames
            if game_over or truncated:
                break
            if life_lost and self.protocol.life_loss == "terminate":
                break
        if not pool:
            pool.append(self._grab())
        self._pool = pool
        obs = self._observation()

        ends_on_life = life_lost and self.protocol.life_loss == "terminate"
        terminated = game_over or ends_on_life
        truncated = truncated and not terminated
        done = terminated or truncated
        if game_over or truncated:
            self._needs_reset = True
        info: dict[str, Any] = {
            "lives": self._lives,
            "life_lost": life_lost,
            "game_over": game_over,
            "episode_frame_number": self._frames,
        }
        return (
            obs,
            reward,
            done,
            finalize_step_info(
                info, done=done, terminated=terminated, truncated=truncated
            ),
        )

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(): the game is over.")
        index, model_action = encode_discrete_action(action, len(self._actions))
        obs, reward, done, info = self._step_native(index)
        info["action"] = model_action
        info["executed_action"] = index
        return obs, reward, done, info

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """The last pooled frame at full 210x160 resolution (for videos)."""
        if self._last_full_frame is None:
            raise RuntimeError("No frame available. Call reset() before render().")
        return self._last_full_frame.copy()

    def close(self) -> None:
        """Nothing to release: ALE frees its emulator when garbage collected."""


def _rom_path(game: str) -> str:
    from ale_py import roms

    path = roms.get_rom_path(game)
    if path is None:
        raise ValueError(f"ale-py has no ROM for {game!r}.")
    return str(path)


def make_atari100k_env(game: str, **kwargs: Any) -> Atari100kEnv:
    """Create an Atari 100k game, e.g. ``make_atari100k_env("Pong", protocol="iris")``."""
    return Atari100kEnv(game, **kwargs)


def atari_protocol_settings(protocol: str | AtariProtocol) -> dict[str, Any]:
    """A protocol's settings as a plain dict, for logging with a run."""
    return asdict(resolve_atari_protocol(protocol))


__all__ = [
    "ATARI_100K_PROTOCOLS",
    "ATARI_100K_ROMS",
    "Atari100kEnv",
    "AtariProtocol",
    "atari_protocol_settings",
    "list_atari100k_games",
    "make_atari100k_env",
    "normalize_atari_game",
    "resolve_atari_protocol",
]
