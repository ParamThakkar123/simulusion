"""Game datasets: frame preprocessing, presets, retro games and gameplay video.

stable-retro has no wheel for every platform, so the retro tests run against
a stand-in ``stable_retro`` module with the same calls the importers make.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
from click.testing import CliRunner

from simulusion import cli
from simulusion.data import (
    Episode,
    EpisodeFolder,
    FramePipeline,
    GamePreset,
    get_game,
    list_games,
    read_meta,
    register_game,
)

# ------------------------------------------------------------------ pipeline


def _counting_episode(T: int = 9) -> Episode:
    """Frame t is filled with t; transition t has action t, reward 1."""
    return Episode(
        frames=np.stack([np.full((20, 30, 3), t, np.uint8) for t in range(T)]),
        actions=(np.arange(T - 1)[:, None] == np.arange(3)).astype(np.float32),
        rewards=np.ones(T - 1, np.float32),
        terminated=np.r_[np.zeros(T - 2, bool), True],
        observations={"state": np.arange(T, dtype=np.float32)},
    )


def test_frame_skip_merges_transitions() -> None:
    ep = _counting_episode(9)  # frames 0..8, transitions 0..7
    out = FramePipeline(frame_skip=4).apply(ep)
    assert ep.actions is not None
    np.testing.assert_array_equal(out.frames[:, 0, 0, 0], [0, 4, 8])
    np.testing.assert_array_equal(out.rewards, [4.0, 4.0])
    np.testing.assert_array_equal(out.terminated, [False, True])
    np.testing.assert_array_equal(out.actions, ep.actions[[0, 4]])  # "first"
    np.testing.assert_array_equal(out.observations["state"], [0, 4, 8])

    any_ = FramePipeline(frame_skip=4, action_reduce="any").apply(ep)
    # Transitions 0-3 press buttons 0, 1, 2 (then none); 4-7 press none.
    np.testing.assert_array_equal(any_.actions, [[1, 1, 1], [0, 0, 0]])


def test_crop_resize_and_grayscale() -> None:
    frames = np.zeros((2, 20, 30, 3), np.uint8)
    frames[:, :4] = 255  # a white score bar along the top
    pipeline = FramePipeline(crop=(4, 0, 0, 0), size=(8, 12), grayscale=True)
    out = pipeline.frames(frames)
    assert out.shape == (2, 8, 12, 1) and out.max() == 0
    assert pipeline.frame(frames[0]).shape == (8, 12, 1)
    assert FramePipeline.from_json(pipeline.to_json()) == pipeline


def test_pipeline_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        FramePipeline(frame_skip=0)
    with pytest.raises(ValueError, match="action_reduce"):
        FramePipeline(action_reduce="last")
    with pytest.raises(ValueError, match="leaves nothing"):
        FramePipeline(crop=(10, 10, 0, 0)).frames(np.zeros((1, 20, 20, 3), np.uint8))


# ------------------------------------------------------------------- presets


def test_presets_cover_the_requested_games() -> None:
    assert {"street-fighter-2", "terraria", "super-mario-bros", "sonic"} <= set(
        list_games()
    )
    assert get_game("street-fighter-2").retro_game == (
        "StreetFighterIISpecialChampionEdition-Genesis"
    )
    assert get_game("terraria").source == "video"
    assert "terraria" not in list_games("retro")
    with pytest.raises(KeyError, match="known"):
        get_game("pong-deluxe")


def test_register_game_refuses_duplicates() -> None:
    preset = GamePreset(name="my-game", title="Mine", source="video")
    register_game(preset, overwrite=True)
    with pytest.raises(ValueError, match="already exists"):
        register_game(preset)
    with pytest.raises(ValueError, match="retro_game"):
        GamePreset(name="x", title="x", source="retro")


# --------------------------------------------------------------------- retro

BUTTONS = ["B", "A", "UP", "DOWN"]


class _Space:
    def __init__(self, discrete: bool) -> None:
        self.n = 6 if discrete else None
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> Any:
        if self.n is not None:
            return int(self._rng.integers(self.n))
        return self._rng.integers(0, 2, len(BUTTONS)).astype(np.int8)


class _Env:
    """Frame t shows the value t; the game ends after ``length`` frames."""

    length = 10

    def __init__(self, game: str, **kwargs: Any) -> None:
        self.game, self.kwargs = game, kwargs
        self.buttons = BUTTONS
        self.num_buttons = len(BUTTONS)
        discrete = kwargs.get("use_restricted_actions") == "DISCRETE"
        self.action_space = _Space(discrete)
        self.initial_state = None
        self.received: List[np.ndarray] = []

    def _obs(self) -> np.ndarray:
        return np.full((24, 32, 3), self.t, np.uint8)

    def reset(self, seed: Any = None) -> tuple:
        self.t = 0
        return self._obs(), {}

    def action_to_array(self, action: Any) -> List[np.ndarray]:
        pressed = np.asarray(action, np.uint8).copy()
        pressed[3] = 0 if pressed[2] else pressed[3]  # no UP with DOWN
        return [pressed]

    def step(self, action: Any) -> tuple:
        self.received.append(np.asarray(action))
        self.t += 1
        return self._obs(), 1.0, self.t >= self.length, False, {}

    def close(self) -> None:
        pass


class _Movie:
    """A movie whose frame t presses button t % 4."""

    frames = 9

    def __init__(self, path: str) -> None:
        self.path, self.cursor, self.players = path, -1, 1

    def step(self) -> bool:
        self.cursor += 1
        return self.cursor <= self.frames

    def get_game(self) -> str:
        return "Fake-Genesis"

    def get_state(self) -> bytes:
        return b"state"

    def get_key(self, button: int, player: int) -> bool:
        return button == (self.cursor - 1) % len(BUTTONS)


@pytest.fixture
def fake_retro(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("stable_retro")
    module.made = []  # type: ignore[attr-defined]

    def make(game: str, **kwargs: Any) -> _Env:
        env = _Env(game, **kwargs)
        module.made.append(env)  # type: ignore[attr-defined]
        return env

    module.make = make  # type: ignore[attr-defined]
    module.Movie = _Movie  # type: ignore[attr-defined]
    module.State = types.SimpleNamespace(DEFAULT="DEFAULT", NONE="NONE")  # type: ignore[attr-defined]
    module.Actions = types.SimpleNamespace(  # type: ignore[attr-defined]
        ALL="ALL", FILTERED="FILTERED", DISCRETE="DISCRETE"
    )
    module.data = types.SimpleNamespace(  # type: ignore[attr-defined]
        Integrations=types.SimpleNamespace(ALL="ALL"),
        # stable-retro 1.0 spells integrations with a -v0 suffix.
        list_games=lambda inttype: [
            "Fake-Genesis-v0",
            "StreetFighterIISpecialChampionEdition-Genesis-v0",
        ],
    )
    monkeypatch.setitem(sys.modules, "stable_retro", module)
    return module


def test_record_retro_writes_button_episodes(fake_retro, tmp_path: Path) -> None:
    from simulusion.data import record_retro

    record_retro(
        "street-fighter-2",
        tmp_path,
        episodes=2,
        max_steps=4,
        pipeline=FramePipeline(size=(12, 16), frame_skip=2),
    )
    env = fake_retro.made[0]
    assert env.game == "StreetFighterIISpecialChampionEdition-Genesis-v0"
    assert env.kwargs["use_restricted_actions"] == "FILTERED"
    folder = EpisodeFolder(tmp_path)
    ep = folder[0]
    # Four steps of two frames each, the last frame of each kept.
    assert ep.frames.shape == (5, 12, 16, 3)
    np.testing.assert_array_equal(ep.frames[:, 0, 0, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(ep.rewards, [2, 2, 2, 2])
    np.testing.assert_array_equal(ep.truncated, [False, False, False, True])
    # Actions are the buttons the console received, after filtering (the
    # first episode's four steps, each held for two frames).
    sent = [env.action_to_array(a)[0] for a in env.received[:8:2]]
    assert ep.actions is not None
    np.testing.assert_array_equal(ep.actions, np.stack(sent).astype(np.float32))
    assert not (ep.actions[:, 2].astype(bool) & ep.actions[:, 3].astype(bool)).any()
    meta = read_meta(tmp_path)
    assert meta is not None
    assert meta.action_kind == "continuous" and meta.action_dim == 4
    assert meta.info["buttons"] == BUTTONS and meta.fps == 30.0


def test_record_retro_ends_with_the_game(fake_retro, tmp_path: Path) -> None:
    from simulusion.data import record_retro

    record_retro(
        "Fake-Genesis", tmp_path, episodes=1, max_steps=100, actions="discrete"
    )
    ep = EpisodeFolder(tmp_path)[0]
    assert len(ep) == _Env.length + 1
    assert ep.terminated is not None and ep.truncated is not None
    assert ep.terminated[-1] and not ep.truncated.any()
    assert ep.actions is not None and ep.actions.dtype == np.int64
    meta = read_meta(tmp_path)
    assert meta is not None and meta.action_kind == "discrete" and meta.action_dim == 6


def test_record_retro_refuses_a_video_game(fake_retro, tmp_path: Path) -> None:
    from simulusion.data import record_retro

    with pytest.raises(ValueError, match="import_gameplay"):
        record_retro("terraria", tmp_path)


def test_movies_replay_into_episodes(fake_retro, tmp_path: Path) -> None:
    from simulusion.data import episode_from_movie, import_retro_movies

    (tmp_path / "movies").mkdir()
    for name in ("a.bk2", "b.bk2"):
        (tmp_path / "movies" / name).write_bytes(b"")
    ep = episode_from_movie(tmp_path / "movies" / "a.bk2")
    env = fake_retro.made[0]
    assert env.game == "Fake-Genesis-v0" and env.initial_state == b"state"
    assert env.kwargs["use_restricted_actions"] == "ALL"
    assert len(ep) == _Env.length  # the game ends before the movie does
    np.testing.assert_array_equal(ep.frames[:, 0, 0, 0], np.arange(_Env.length))
    assert ep.actions is not None
    np.testing.assert_array_equal(ep.actions.argmax(1), np.arange(9) % 4)

    import_retro_movies(tmp_path / "movies", tmp_path / "out", game="sonic")
    folder = EpisodeFolder(tmp_path / "out")
    assert len(folder) == 2
    # The preset's 64x64 frames, four frames apart; a press anywhere in
    # those four frames counts ("any").
    assert folder[0].frames.shape == (3, 64, 64, 3)
    np.testing.assert_array_equal(folder[0].actions, np.ones((2, 4), np.float32))
    meta = read_meta(tmp_path / "out")
    assert meta is not None and meta.info["pipeline"]["action_reduce"] == "any"


def test_retro_needs_stable_retro(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from simulusion.data import record_retro

    monkeypatch.setitem(sys.modules, "stable_retro", None)
    monkeypatch.setitem(sys.modules, "retro", None)
    with pytest.raises(ImportError, match=r"simulusion\[retro\]"):
        record_retro("sonic", tmp_path)


# ------------------------------------------------------------ gameplay video


def _write_capture(path: Path, values: List[int], fps: float = 60.0) -> None:
    from simulusion.data.video_io import write_video

    frames = np.stack([np.full((36, 64, 3), v, np.uint8) for v in values])
    write_video(path, frames, fps=fps)


def test_gameplay_video_with_input_log(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from simulusion.data import import_gameplay

    captures = tmp_path / "captures"
    values = [10 * i for i in range(12)]
    _write_capture(captures / "session.mp4", values)
    rows = ["frame,left,right,mouse_x"] + [
        f"{i},{int(i % 4 == 1)},0,{i}" for i in range(12)
    ]
    (captures / "session.inputs.csv").write_text("\n".join(rows) + "\n")

    import_gameplay(captures, tmp_path / "out", game="terraria", clip_length=2)
    folder = EpisodeFolder(tmp_path / "out")
    # 12 frames, every 4th kept: frames 0, 4, 8, cut into clips of two
    # (the last clip, a single frame, is dropped).
    assert len(folder) == 1
    ep = folder[0]
    assert ep.frames.shape == (2, 72, 128, 3)
    np.testing.assert_allclose(ep.frames[:, 0, 0, 0], [0, 40], atol=3)
    # Rows 0-3 merged with "any" (left pressed on frame 1); mouse_x non-zero too.
    np.testing.assert_array_equal(ep.actions, [[1, 0, 1]])
    meta = read_meta(tmp_path / "out")
    assert meta is not None
    assert meta.info["inputs"] == ["left", "right", "mouse_x"]
    assert meta.action_kind == "continuous" and meta.fps == pytest.approx(15.0)


def test_gameplay_video_without_logs_and_mixed_logs(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from simulusion.data import import_gameplay

    _write_capture(tmp_path / "a.mp4", [0, 50, 100, 150])
    _write_capture(tmp_path / "b.mp4", [0, 50, 100, 150])
    np.save(tmp_path / "a.inputs.npy", np.zeros((4, 2), np.float32))
    with pytest.raises(ValueError, match="Missing: b.mp4"):
        import_gameplay(tmp_path, tmp_path / "out")
    import_gameplay(tmp_path, tmp_path / "out", inputs=False)
    folder = EpisodeFolder(tmp_path / "out")
    assert len(folder) == 2 and folder[0].actions is None
    assert folder[0].frames.shape == (4, 36, 64, 3)  # no preset: kept as is


def test_input_log_shorter_than_the_video_truncates(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from simulusion.data import import_gameplay

    _write_capture(tmp_path / "a.mp4", [0, 50, 100, 150, 200])
    (tmp_path / "a.inputs.csv").write_text("jump\n1\n0\n1\n")
    import_gameplay(tmp_path, tmp_path / "out")
    ep = EpisodeFolder(tmp_path / "out")[0]
    assert len(ep) == 3
    np.testing.assert_array_equal(ep.actions, [[1], [0]])


# ----------------------------------------------------------------------- CLI


def test_cli_lists_games() -> None:
    result = CliRunner().invoke(cli.app, ["datasets", "games"])
    assert result.exit_code == 0, result.output
    assert "street-fighter-2" in result.output and "terraria" in result.output


def test_cli_imports_video_with_options(tmp_path: Path) -> None:
    pytest.importorskip("av")
    _write_capture(tmp_path / "clip.mp4", [0, 50, 100, 150])
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "import",
            "video",
            str(tmp_path / "clip.mp4"),
            str(tmp_path / "out"),
            "--size",
            "16x32",
            "--frame-skip",
            "2",
            "--grayscale",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 episodes, 2 frames" in result.output
    assert EpisodeFolder(tmp_path / "out")[0].frames.shape == (2, 16, 32, 1)


def test_cli_record_and_movie_import(fake_retro, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "record",
            "Fake-Genesis",
            str(tmp_path / "rec"),
            "--episodes",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "actions: continuous (4)" in result.output

    (tmp_path / "m.bk2").write_bytes(b"")
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "import",
            "retro-movie",
            str(tmp_path / "m.bk2"),
            str(tmp_path / "mv"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_reports_unknown_presets(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"")
    result = CliRunner().invoke(
        cli.app,
        [
            "datasets",
            "import",
            "video",
            str(tmp_path),
            str(tmp_path / "o"),
            "--game",
            "nope",
        ],
    )
    assert result.exit_code == 1
    assert "No game preset 'nope'" in result.output
