"""Retro console games (NES, SNES, Genesis, ...) as episodes, via stable-retro.

`stable-retro`_ (``pip install "simulusion[retro]"``) emulates over a
thousand console games behind a Gymnasium interface, Street Fighter II,
Super Mario Bros. and Sonic among them. Two ways to get episodes from it:

* :func:`record_retro` plays a game with a policy (random by default) and
  writes what it sees: frames, button presses, the integration's rewards and
  episode ends.
* :func:`import_retro_movies` replays ``.bk2`` movies, the button-by-button
  recordings of human play stable-retro and BizHawk write, through the
  emulator and keeps every frame; human play explores far more of a game
  than random buttons do.

.. code-block:: python

    from simulusion.data import import_retro_movies, record_retro

    record_retro("street-fighter-2", "data/sf2", episodes=20, max_steps=2000)
    import_retro_movies("recordings/", "data/sf2_human", game="street-fighter-2")

or from the shell: ``simulusion datasets record street-fighter-2 data/sf2``.

Only ``Airstriker-Genesis`` ships with its ROM; import ROMs you own with
``python -m retro.import /path/to/roms``.

Actions are the console's buttons, one ``0 / 1`` column each (``meta.json``
lists them under ``info.buttons``), so a fighting game's combinations stay
expressible; ``actions="discrete"`` records an index into the integration's
valid combinations instead, which suits models with a discrete action head.

.. _stable-retro: https://github.com/Farama-Foundation/stable-retro
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeWriter
from simulusion.data.games import GamePreset, get_game
from simulusion.data.preprocess import FramePipeline

logger = logging.getLogger(__name__)

Policy = Callable[[np.ndarray], Any]


def record_retro(
    game: str,
    out_dir: str | Path,
    *,
    episodes: int = 10,
    max_steps: int = 4500,
    state: Optional[str] = None,
    policy: Optional[Policy] = None,
    actions: str = "buttons",
    pipeline: Optional[FramePipeline] = None,
    seed: int = 0,
) -> Path:
    """Play a retro game and write each episode to ``out_dir``.

    Args:
        game: A preset name (``"street-fighter-2"``) or a stable-retro
            integration (``"StreetFighterIISpecialChampionEdition-Genesis"``).
        out_dir: Episode folder to write (appended to if it exists).
        episodes: Episodes to record.
        max_steps: Steps per episode before it is cut off (``truncated``);
            each step is ``pipeline.frame_skip`` emulator frames.
        state: Save state to start from; the preset's, else the game's default.
        policy: Maps the current raw frame to an action; random when ``None``.
        actions: ``"buttons"`` (a ``0 / 1`` vector per step) or
            ``"discrete"`` (an index into the valid button combinations).
        pipeline: Preprocessing; the preset's when ``None``. Its
            ``frame_skip`` is used as action repeat, the way Atari agents
            act: each action is held for that many frames, their rewards
            summed, and only the last frame kept.
        seed: Seeds the random policy.

    Returns:
        ``out_dir``.
    """
    retro = _import_retro()
    preset = _preset(game)
    integration = _integration(retro, (preset.retro_game if preset else None) or game)
    pipeline = pipeline or (preset.pipeline if preset else FramePipeline())
    repeat = pipeline.frame_skip
    per_frame = dataclasses.replace(pipeline, frame_skip=1)
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be at least 1.")
    if actions not in ("buttons", "discrete"):
        raise ValueError(f"actions must be 'buttons' or 'discrete'; got '{actions}'.")

    env = retro.make(
        integration,
        state=state or (preset.retro_state if preset else None) or retro.State.DEFAULT,
        use_restricted_actions=(
            retro.Actions.DISCRETE if actions == "discrete" else retro.Actions.FILTERED
        ),
        render_mode="rgb_array",
    )
    env.action_space.seed(seed)
    buttons = [str(b) for b in getattr(env, "buttons", []) or []]
    meta = _meta(
        integration,
        preset,
        pipeline,
        buttons=buttons if actions == "buttons" else None,
        num_actions=int(env.action_space.n) if actions == "discrete" else None,
        source=f"retro:{integration}",
        extra={"state": state or (preset.retro_state if preset else None)},
    )
    try:
        with EpisodeWriter(out_dir, meta=meta) as writer:
            for index in range(episodes):
                obs, _ = env.reset(seed=seed + index)
                frames: List[np.ndarray] = [per_frame.frame(obs)]
                taken: List[Any] = []
                rewards: List[float] = []
                ends: List[bool] = []
                for _ in range(max_steps):
                    action = (
                        env.action_space.sample() if policy is None else policy(obs)
                    )
                    total, done = 0.0, False
                    for _ in range(repeat):
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total += float(reward)
                        if terminated or truncated:
                            done = True
                            break
                    frames.append(per_frame.frame(obs))
                    if actions == "buttons":
                        # Restricted action sets drop some presses (UP with
                        # DOWN, say); keep the buttons the console received.
                        action = env.action_to_array(action)[0]
                    taken.append(action)
                    rewards.append(total)
                    ends.append(done)
                    if done:
                        break
                cut = np.zeros(len(ends), dtype=bool)
                if ends and not ends[-1]:
                    cut[-1] = True
                writer.write(
                    Episode(
                        frames=np.stack(frames),
                        actions=_actions(taken, actions),
                        rewards=np.asarray(rewards, np.float32),
                        terminated=np.asarray(ends, bool),
                        truncated=cut,
                    )
                )
                logger.info(
                    "%s: episode %d/%d, %d steps, return %.1f",
                    integration,
                    index + 1,
                    episodes,
                    len(taken),
                    sum(rewards),
                )
    finally:
        env.close()
    return Path(out_dir)


def episode_from_movie(
    path: str | Path,
    *,
    pipeline: Optional[FramePipeline] = None,
    players: Optional[int] = None,
) -> Episode:
    """Replay one ``.bk2`` movie through the emulator as an episode.

    Every emulator frame of the movie is a step: ``actions[t]`` are the
    buttons held on frame ``t`` (all players' buttons side by side), and
    ``rewards`` / ``terminated`` come from the game's integration.

    Args:
        path: The ``.bk2`` file.
        pipeline: Preprocessing applied after the replay; button presses
            merged by frame skip combine as its ``action_reduce`` says
            (``"any"`` keeps a short press).
        players: Players whose buttons to keep; all the movie has when ``None``.
    """
    episode, _ = _replay(path, players=players, pipeline=pipeline)
    return episode


def import_retro_movies(
    source: str | Path,
    out_dir: str | Path,
    *,
    game: Optional[str] = None,
    pipeline: Optional[FramePipeline] = None,
    players: Optional[int] = None,
) -> Path:
    """Replay ``.bk2`` movies (a file, or a folder searched recursively) into
    an episode folder; returns ``out_dir``.

    Args:
        source: A ``.bk2`` file or a folder of them.
        out_dir: Episode folder to write.
        game: A preset whose pipeline to use; the movies name their own game.
        pipeline: Preprocessing; overrides the preset's. Movies are human
            play, so the preset pipeline's actions combine with ``"any"``.
        players: Players whose buttons to keep; all when ``None``.
    """
    paths = _movie_paths(source)
    preset = get_game(game) if game else None
    if pipeline is None:
        pipeline = (
            dataclasses.replace(preset.pipeline, action_reduce="any")
            if preset
            else FramePipeline()
        )
    writer: Optional[EpisodeWriter] = None
    try:
        for path in paths:
            episode, info = _replay(path, players=players, pipeline=pipeline)
            if writer is None:
                meta = _meta(
                    info["game"],
                    preset,
                    pipeline,
                    buttons=info["buttons"],
                    num_actions=None,
                    source=f"retro-movie:{source}",
                    extra={"players": info["players"]},
                )
                writer = EpisodeWriter(out_dir, meta=meta)
            writer.write(episode)
            logger.info("%s: %d frames", path.name, len(episode))
    finally:
        if writer is not None:
            writer.close()
    return Path(out_dir)


def _replay(
    path: str | Path,
    *,
    players: Optional[int],
    pipeline: Optional[FramePipeline] = None,
) -> tuple[Episode, dict]:
    """Replay a movie; frames are cropped and resized as they are rendered
    (a long movie at full resolution would not fit in memory), frame skip is
    applied once the replay ends."""
    retro = _import_retro()
    pipeline = pipeline or FramePipeline()
    per_frame = dataclasses.replace(pipeline, frame_skip=1)
    movie = retro.Movie(str(path))
    movie.step()  # load the movie's first input frame, as playback does
    game = _integration(retro, movie.get_game())
    total_players = int(movie.players)
    keep_players = total_players if players is None else min(players, total_players)
    env = retro.make(
        game,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        players=total_players,
        render_mode="rgb_array",
    )
    try:
        env.initial_state = movie.get_state()
        obs, _ = env.reset()
        num_buttons = int(env.num_buttons)
        frames = [per_frame.frame(obs)]
        keys: List[List[bool]] = []
        rewards: List[float] = []
        ends: List[bool] = []
        while movie.step():
            pressed = [
                bool(movie.get_key(i, p))
                for p in range(total_players)
                for i in range(num_buttons)
            ]
            obs, reward, terminated, truncated, _ = env.step(pressed)
            frames.append(per_frame.frame(obs))
            keys.append(pressed[: keep_players * num_buttons])
            rewards.append(float(np.sum(reward)))
            ends.append(bool(terminated))
            if terminated or truncated:
                break
        buttons = [str(b) for b in env.buttons]
    finally:
        env.close()
    if not keys:
        raise ValueError(f"{path} holds no input frames.")
    names = [
        f"P{p + 1}_{b}" if keep_players > 1 else b
        for p in range(keep_players)
        for b in buttons
    ]
    episode = Episode(
        frames=np.stack(frames),
        actions=np.asarray(keys, np.float32),
        rewards=np.asarray(rewards, np.float32),
        terminated=np.asarray(ends, bool),
    )
    skip = FramePipeline(
        frame_skip=pipeline.frame_skip, action_reduce=pipeline.action_reduce
    )
    return skip.apply(episode), {
        "game": game,
        "buttons": names,
        "players": keep_players,
    }


def _meta(
    integration: str,
    preset: Optional[GamePreset],
    pipeline: FramePipeline,
    *,
    buttons: Optional[Sequence[str]],
    num_actions: Optional[int],
    source: str,
    extra: dict,
) -> DatasetMeta:
    fps = preset.fps if preset and preset.fps else 60.0
    info = {
        "game": integration,
        "preset": preset.name if preset else None,
        "pipeline": pipeline.to_json(),
        **extra,
    }
    if buttons is not None:
        info["buttons"] = list(buttons)
        return DatasetMeta(
            action_kind="continuous",
            action_dim=len(buttons),
            action_low=[0.0] * len(buttons),
            action_high=[1.0] * len(buttons),
            fps=fps / pipeline.frame_skip,
            source=source,
            info=info,
        )
    return DatasetMeta(
        action_kind="discrete",
        action_dim=num_actions,
        fps=fps / pipeline.frame_skip,
        source=source,
        info=info,
    )


def _actions(taken: List[Any], kind: str) -> np.ndarray:
    if kind == "discrete":
        return np.asarray([int(np.asarray(a)) for a in taken], dtype=np.int64)
    return np.asarray([np.asarray(a).reshape(-1) for a in taken], dtype=np.float32)


def _preset(game: str) -> Optional[GamePreset]:
    try:
        preset = get_game(game)
    except KeyError:
        return None
    if preset.source != "retro":
        raise ValueError(
            f"'{game}' is not an emulated game; import its gameplay video with "
            "simulusion.data.import_gameplay instead."
        )
    return preset


def _movie_paths(source: str | Path) -> List[Path]:
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"No such movie or folder: {root}")
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.bk2")
    found = sorted(p for p in paths if p.suffix.lower() == ".bk2")
    if not found:
        raise ValueError(f"No .bk2 movies under {root}.")
    return found


def _integration(retro: Any, name: str) -> str:
    """stable-retro 1.0 names integrations ``<Game>-<System>-v0``; earlier
    releases had no suffix. Accept either spelling with either release."""
    known = set(retro.data.list_games(retro.data.Integrations.ALL))
    for candidate in (name, f"{name}-v0", name.removesuffix("-v0")):
        if candidate in known:
            return candidate
    return name  # retro.make explains what is missing


def _import_retro() -> Any:
    try:
        import stable_retro as retro  # stable-retro >= 1.0
    except ImportError:
        try:
            import retro  # stable-retro < 1.0 (and gym-retro)
        except ImportError as exc:
            raise ImportError(
                "Retro games need stable-retro: pip install simulusion[retro]"
            ) from exc
    return retro


__all__ = ["episode_from_movie", "import_retro_movies", "record_retro"]
