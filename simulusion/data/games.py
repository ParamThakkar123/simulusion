"""Named game presets: where a game's data comes from and how it is prepared.

A :class:`GamePreset` ties a short name (``"street-fighter-2"``) to its
source and a :class:`~simulusion.data.preprocess.FramePipeline`:

* **Emulated retro games** (NES, SNES, Genesis, ...) run in `stable-retro`_
  (``pip install "simulusion[retro]"``). ``retro_game`` is the integration
  name; :func:`~simulusion.data.retro.record_retro` plays them and
  :func:`~simulusion.data.retro.import_retro_movies` replays recorded human
  play (``.bk2`` files). Apart from ``airstriker`` (a free homebrew game that
  ships with stable-retro) the ROMs are not included: import ones you own
  with ``python -m retro.import /path/to/roms``.
* **PC games without an emulator** (Terraria) are captured as gameplay
  video, optionally with a per-frame input log, and converted with
  :func:`~simulusion.data.gameplay.import_gameplay`.

Presets are defaults, not requirements: every importer also takes a
pipeline of its own, and :func:`register_game` adds presets.

.. _stable-retro: https://github.com/Farama-Foundation/stable-retro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulusion.data.preprocess import FramePipeline


@dataclass(frozen=True)
class GamePreset:
    """One game's data source and preprocessing.

    Attributes:
        name: Short name used by the CLI and the importers.
        title: The game's full title.
        source: ``"retro"`` (emulated in stable-retro) or ``"video"``
            (captured gameplay).
        pipeline: Default preprocessing of its frames.
        retro_game: The stable-retro integration, for ``source="retro"``.
        retro_state: Save state to start from; the integration's default
            when ``None``.
        fps: Native frame rate, recorded in ``meta.json``.
        notes: Anything worth knowing when collecting the data.
    """

    name: str
    title: str
    source: str
    pipeline: FramePipeline = field(default_factory=FramePipeline)
    retro_game: Optional[str] = None
    retro_state: Optional[str] = None
    fps: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source not in ("retro", "video"):
            raise ValueError(f"source must be 'retro' or 'video'; got '{self.source}'.")
        if self.source == "retro" and not self.retro_game:
            raise ValueError(
                f"Preset '{self.name}' is a retro game without retro_game."
            )


# Emulated games keep their own frame (Genesis 320 x 224, NES 256 x 240) at
# 64 x 64, four emulator frames apart: the setting Atari world models use.
_RETRO = FramePipeline(size=(64, 64), frame_skip=4)

_GAMES: Dict[str, GamePreset] = {}


def register_game(preset: GamePreset, *, overwrite: bool = False) -> GamePreset:
    """Add a preset under ``preset.name``; returns it."""
    if preset.name in _GAMES and not overwrite:
        raise ValueError(f"A game preset named '{preset.name}' already exists.")
    _GAMES[preset.name] = preset
    return preset


def get_game(name: str) -> GamePreset:
    """The preset called ``name``."""
    try:
        return _GAMES[name]
    except KeyError:
        raise KeyError(
            f"No game preset '{name}'; known: {', '.join(list_games())}"
        ) from None


def list_games(source: Optional[str] = None) -> List[str]:
    """Preset names, optionally only those with the given ``source``."""
    return sorted(n for n, p in _GAMES.items() if source in (None, p.source))


for _preset in (
    GamePreset(
        name="street-fighter-2",
        title="Street Fighter II' - Special Champion Edition (Genesis)",
        source="retro",
        pipeline=_RETRO,
        retro_game="StreetFighterIISpecialChampionEdition-Genesis",
        fps=60.0,
        notes="Fighting game: two characters, health bars, a round timer.",
    ),
    GamePreset(
        name="super-mario-bros",
        title="Super Mario Bros. (NES)",
        source="retro",
        pipeline=_RETRO,
        retro_game="SuperMarioBros-Nes",
        fps=60.0,
        notes="Side-scrolling platformer.",
    ),
    GamePreset(
        name="super-mario-world",
        title="Super Mario World (SNES)",
        source="retro",
        pipeline=_RETRO,
        retro_game="SuperMarioWorld-Snes",
        fps=60.0,
        notes="Side-scrolling platformer.",
    ),
    GamePreset(
        name="sonic",
        title="Sonic The Hedgehog (Genesis)",
        source="retro",
        pipeline=_RETRO,
        retro_game="SonicTheHedgehog-Genesis",
        fps=60.0,
        notes="Fast side-scroller; the game of the OpenAI Retro Contest.",
    ),
    GamePreset(
        name="airstriker",
        title="Airstriker (Genesis)",
        source="retro",
        pipeline=_RETRO,
        retro_game="Airstriker-Genesis",
        fps=60.0,
        notes="Homebrew shooter whose ROM ships with stable-retro: works "
        "without importing anything.",
    ),
    GamePreset(
        name="terraria",
        title="Terraria",
        source="video",
        # 16:9 captures keep their aspect ratio; 60 fps footage becomes 15.
        pipeline=FramePipeline(size=(72, 128), frame_skip=4, action_reduce="any"),
        fps=60.0,
        notes="Side-scrolling sandbox; no emulator, so data is captured "
        "gameplay video with an optional per-frame input log.",
    ),
):
    register_game(_preset)


__all__ = ["GamePreset", "get_game", "list_games", "register_game"]
