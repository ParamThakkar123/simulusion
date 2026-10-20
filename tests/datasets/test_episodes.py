"""simulusion.data: the episode format, folders, windows, LeRobot and RLDS readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.data import (
    DatasetMeta,
    Episode,
    EpisodeFolder,
    EpisodeWriter,
    SequenceDataset,
    collate_sequences,
)
from simulusion.data.episode import npz_length


def _episode(T: int = 6, *, actions: bool = True, offset: int = 0) -> Episode:
    frames = np.stack(
        [np.full((8, 8, 3), (offset + t) % 256, np.uint8) for t in range(T)]
    )
    return Episode(
        frames=frames,
        actions=np.arange(T - 1) % 3 if actions else None,
        rewards=np.arange(T - 1, dtype=np.float32) if actions else None,
        terminated=np.r_[np.zeros(T - 2, bool), True] if actions else None,
        observations={"state": np.arange(T * 2, dtype=np.float32).reshape(T, 2)},
        language="push the block",
    )


def test_episode_round_trips_and_reads_length_from_header(tmp_path: Path) -> None:
    ep = _episode()
    path = ep.save(tmp_path / "ep")
    assert path.suffix == ".npz" and npz_length(path) == 6
    back = Episode.load(path)
    np.testing.assert_array_equal(back.frames, ep.frames)
    np.testing.assert_array_equal(back.actions, ep.actions)
    np.testing.assert_array_equal(back.observations["state"], ep.observations["state"])
    assert back.language == "push the block"
    assert back.num_transitions == 5


def test_episode_validates_transition_lengths() -> None:
    with pytest.raises(ValueError, match="T - 1"):
        Episode(frames=np.zeros((4, 8, 8, 3), np.uint8), actions=np.zeros(4))


def test_float_and_channel_first_frames_are_normalised() -> None:
    ep = Episode(frames=np.ones((3, 3, 8, 8), np.float32))
    assert ep.frames.shape == (3, 8, 8, 3) and ep.frames.dtype == np.uint8
    assert int(ep.frames.max()) == 255


def test_writer_numbers_episodes_and_infers_meta(tmp_path: Path) -> None:
    with EpisodeWriter(tmp_path) as writer:
        writer.write(_episode())
        writer.write(_episode(offset=10))
    with EpisodeWriter(tmp_path) as writer:  # appends
        writer.write(_episode(offset=20))
    folder = EpisodeFolder(tmp_path)
    assert [p.name for p in folder.files] == [
        "episode_000000.npz",
        "episode_000001.npz",
        "episode_000002.npz",
    ]
    assert folder.lengths == [6, 6, 6]
    assert folder.meta is not None
    assert folder.meta.action_kind == "discrete" and folder.meta.action_dim == 3
    assert folder.meta.observation_keys == ["state"]
    train, val = folder.split(0.34)
    assert len(train) == 2 and len(val) == 1
    assert set(train.files).isdisjoint(val.files)


def test_folder_reads_loose_npy_frames(tmp_path: Path) -> None:
    np.save(tmp_path / "clip.npy", np.zeros((5, 8, 8, 3), np.uint8))
    folder = EpisodeFolder(tmp_path)
    assert folder.lengths == [5] and folder[0].actions is None


def test_sequence_windows_align_actions_with_frames(tmp_path: Path) -> None:
    EpisodeWriter(tmp_path).write(_episode(T=6))
    data = SequenceDataset(
        EpisodeFolder(tmp_path), sequence_length=3, observation_keys=["state"]
    )
    assert len(data) == 4  # starts 0..3
    item = data[2]
    assert item["frames"].shape == (3, 3, 8, 8) and item["frames"].dtype == torch.uint8
    # Frame t is filled with value t, so the window starts at frame 2.
    assert item["frames"][:, 0, 0, 0].tolist() == [2, 3, 4]
    # Actions/rewards between those frames: transitions 2 and 3.
    assert item["actions"].tolist() == [2 % 3, 3 % 3]
    assert item["actions"].dtype == torch.int64
    assert item["rewards"].tolist() == [2.0, 3.0]
    assert item["obs.state"].shape == (3, 2)
    last = data[3]
    assert last["terminated"].tolist() == [False, True]


def test_sequence_resizes_and_skips_short_episodes(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path)
    writer.write(_episode(T=6))
    writer.write(_episode(T=2))
    data = SequenceDataset(
        EpisodeFolder(tmp_path), sequence_length=4, stride=2, image_size=4
    )
    assert len(data) == 2  # starts 0, 2 of the long episode only
    batch = collate_sequences([data[0], data[1]])
    assert batch["frames"].shape == (2, 4, 3, 4, 4)


def test_video_only_episodes_have_no_actions(tmp_path: Path) -> None:
    EpisodeWriter(tmp_path).write(_episode(actions=False))
    item = SequenceDataset(EpisodeFolder(tmp_path), sequence_length=2)[0]
    assert "actions" not in item
    with pytest.raises(ValueError, match="no actions"):
        SequenceDataset(
            EpisodeFolder(tmp_path), sequence_length=2, require_actions=True
        )[0]


# ------------------------------------------------------------------ LeRobot

FPS = 10


@pytest.fixture
def arrow():
    """PyArrow and PyAV come with the `lerobot` extra; skip without them."""
    pytest.importorskip("av")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    return pa, pq


def _gray_frames(values: list[int]) -> np.ndarray:
    return np.stack([np.full((16, 16, 3), v, np.uint8) for v in values])


def _features() -> dict:
    return {
        "observation.images.top": {"dtype": "video", "shape": [16, 16, 3]},
        "observation.images.wrist": {"dtype": "video", "shape": [16, 16, 3]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }


def _rows(ep: int, T: int, first_index: int) -> dict:
    return {
        "observation.state": [[float(ep), float(t)] for t in range(T)],
        "action": [[float(t), -float(t)] for t in range(T)],
        "next.reward": [float(t) for t in range(T)],
        "next.done": [t == T - 1 for t in range(T)],
        "timestamp": [t / FPS for t in range(T)],
        "frame_index": list(range(T)),
        "episode_index": [ep] * T,
        "index": list(range(first_index, first_index + T)),
        "task_index": [0] * T,
    }


def _make_v21(root: Path, lengths: list[int], arrow) -> None:
    pa, pq = arrow
    from simulusion.data.video_io import write_video

    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    episodes, tasks = [], [{"task_index": 0, "task": "stack the cubes"}]
    first = 0
    for ep, T in enumerate(lengths):
        path = root / info["data_path"].format(episode_chunk=0, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(_rows(ep, T, first)), path)
        for key, base in (
            ("observation.images.top", 0),
            ("observation.images.wrist", 100),
        ):
            write_video(
                root
                / info["video_path"].format(
                    episode_chunk=0, video_key=key, episode_index=ep
                ),
                _gray_frames([base + 10 * t for t in range(T)]),
                fps=FPS,
            )
        episodes.append(
            {"episode_index": ep, "tasks": ["stack the cubes"], "length": T}
        )
        first += T
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes)
    )
    (root / "meta" / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))


def _make_v30(root: Path, lengths: list[int], arrow) -> None:
    pa, pq = arrow
    from simulusion.data.video_io import write_video

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _features(),
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    columns: dict = {}
    records = []
    frames = {"observation.images.top": [], "observation.images.wrist": []}
    first, t0 = 0, 0.0
    for ep, T in enumerate(lengths):
        for key, value in _rows(ep, T, first).items():
            columns.setdefault(key, []).extend(value)
        frames["observation.images.top"].extend(10 * t for t in range(T))
        frames["observation.images.wrist"].extend(100 + 10 * t for t in range(T))
        record = {
            "episode_index": ep,
            "tasks": ["stack the cubes"],
            "length": T,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": first,
            "dataset_to_index": first + T,
        }
        for key in frames:
            record[f"videos/{key}/chunk_index"] = 0
            record[f"videos/{key}/file_index"] = 0
            record[f"videos/{key}/from_timestamp"] = t0
            record[f"videos/{key}/to_timestamp"] = t0 + T / FPS
        records.append(record)
        first += T
        t0 += T / FPS
    data = root / "data" / "chunk-000" / "file-000.parquet"
    data.parent.mkdir(parents=True)
    pq.write_table(pa.table(columns), data)
    for key, values in frames.items():
        write_video(
            root
            / info["video_path"].format(video_key=key, chunk_index=0, file_index=0),
            _gray_frames(values),
            fps=FPS,
        )
    pq.write_table(
        pa.Table.from_pylist(records),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": ["stack the cubes"]}),
        root / "meta" / "tasks.parquet",
    )


@pytest.mark.parametrize("make", [_make_v21, _make_v30], ids=["v2.1", "v3.0"])
def test_lerobot_episodes_map_onto_the_episode_format(
    tmp_path: Path, make, arrow
) -> None:
    from simulusion.data import LeRobotEpisodes

    make(tmp_path / "ds", [5, 4], arrow)
    data = LeRobotEpisodes(tmp_path / "ds", camera="top", extra_cameras="all")
    assert len(data) == 2 and data.lengths == [5, 4]
    ep = data[1]
    assert ep.frames.shape == (4, 16, 16, 3)
    # Gray levels 0, 10, 20, 30 survive the (near-lossless) encode.
    np.testing.assert_allclose(ep.frames[:, 0, 0, 0], [0, 10, 20, 30], atol=3)
    np.testing.assert_allclose(
        ep.cameras["wrist"][:, 0, 0, 0], [100, 110, 120, 130], atol=3
    )
    # The last action leads past the final frame and is dropped.
    np.testing.assert_array_equal(ep.actions, [[0, 0], [1, -1], [2, -2]])
    np.testing.assert_array_equal(ep.rewards, [0, 1, 2])
    # The final next.done moves onto the last transition.
    assert ep.terminated.tolist() == [False, False, True]
    np.testing.assert_array_equal(ep.observations["state"][:, 0], [1, 1, 1, 1])
    assert ep.language == "stack the cubes"
    meta = data.meta
    assert meta.action_kind == "continuous" and meta.action_dim == 2
    assert meta.cameras == ["wrist"] and meta.fps == FPS


def test_import_lerobot_writes_an_episode_folder(tmp_path: Path, arrow) -> None:
    from simulusion.data import import_lerobot

    _make_v30(tmp_path / "ds", [5, 6], arrow)
    out = import_lerobot(tmp_path / "ds", tmp_path / "eps")
    folder = EpisodeFolder(out)
    assert folder.lengths == [5, 6]
    assert folder.meta.source == f"lerobot:{tmp_path / 'ds'}"
    windows = SequenceDataset(folder, sequence_length=4, observation_keys=["state"])
    assert (
        windows[0]["actions"].shape == (3, 2)
        and windows[0]["actions"].dtype == torch.float32
    )


# --------------------------------------------------------------------- RLDS


def test_rlds_episode_conversion() -> None:
    from simulusion.data import RLDSKeys, episode_from_rlds

    steps = [
        {
            "observation": {
                "image": np.full((8, 8, 3), t, np.uint8),
                "wrist": np.zeros((8, 8, 3), np.uint8),
                "state": np.array([t, t], np.float32),
            },
            "action": {"world_vector": np.ones(3) * t, "gripper": np.array([0.5])},
            "reward": np.float32(t),
            "is_first": t == 0,
            "is_last": t == 3,
            "is_terminal": t == 3,
            "language_instruction": b"open the drawer",
        }
        for t in range(4)
    ]
    keys = RLDSKeys(
        cameras=("wrist",), state=("state",), action=("world_vector", "gripper")
    )
    ep = episode_from_rlds({"steps": steps}, keys)
    assert len(ep) == 4 and ep.actions.shape == (3, 4)
    np.testing.assert_array_equal(ep.actions[2], [2, 2, 2, 0.5])
    np.testing.assert_array_equal(ep.rewards, [0, 1, 2])
    assert ep.terminated.tolist() == [False, False, True]
    assert ep.truncated.tolist() == [False, False, False]
    assert ep.language == "open the drawer"
    assert set(ep.cameras) == {"wrist"} and ep.observations["state"].shape == (4, 2)


def test_rlds_episode_without_terminal_is_truncated() -> None:
    from simulusion.data import episode_from_rlds

    steps = [
        {
            "observation": {"image": np.zeros((4, 4, 3), np.uint8)},
            "action": np.zeros(2),
            "is_terminal": False,
        }
        for _ in range(3)
    ]
    ep = episode_from_rlds(steps)
    assert ep.terminated.tolist() == [False, False]
    assert ep.truncated.tolist() == [False, True]


def test_dataset_meta_round_trips(tmp_path: Path) -> None:
    from simulusion.data import read_meta, write_meta

    meta = DatasetMeta(action_kind="continuous", action_dim=7, fps=5.0, source="x")
    write_meta(tmp_path, meta)
    assert read_meta(tmp_path) == meta
