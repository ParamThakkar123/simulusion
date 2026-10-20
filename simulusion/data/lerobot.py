"""Read LeRobot datasets as Simulusion episodes.

`LeRobot`_ is the Hugging Face format most open robot datasets now ship in.
:class:`LeRobotEpisodes` reads one, from a local folder or the Hub, without
the ``lerobot`` package: the format is Parquet for the low-dimensional signals
plus MP4 per camera, described by ``meta/info.json``. Both layouts are
supported:

* **v2.x**: one Parquet file and one MP4 per camera per episode, listed in
  ``meta/episodes.jsonl``.
* **v3.0**: many episodes per Parquet / MP4 file, located through the
  ``meta/episodes/`` tables (row ranges and video timestamps).

.. code-block:: python

    from simulusion.data import LeRobotEpisodes, SequenceDataset

    episodes = LeRobotEpisodes("lerobot/pusht")           # downloads on first use
    episodes[0].frames.shape                               # (T, 96, 96, 3)
    episodes[0].actions.shape                              # (T - 1, 2)
    windows = SequenceDataset(episodes, sequence_length=16)

or convert once to an episode folder: ``simulusion datasets import lerobot
lerobot/pusht data/pusht``. Needs ``pip install simulusion[lerobot]``
(PyArrow and PyAV).

How LeRobot maps onto :class:`~simulusion.data.episode.Episode`: the chosen
camera becomes ``frames`` and the others ``cameras``; ``action`` becomes
``actions`` minus its last row (it leads past the final frame);
``next.reward`` and ``next.done`` become ``rewards`` and ``terminated`` the
same way, with the final ``next.done`` moved onto the last transition;
other numeric ``observation.*`` features become ``observations`` (so
``observation.state`` is ``observations["state"]``); the episode's first task
becomes ``language``.

.. _LeRobot: https://github.com/huggingface/lerobot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from simulusion.data.episode import Episode
from simulusion.data.folder import DatasetMeta, EpisodeWriter
from simulusion.data.video_io import read_video

logger = logging.getLogger(__name__)

IMAGE_PREFIX = "observation.images."


class LeRobotEpisodes(Sequence[Episode]):
    """The episodes of a LeRobot dataset; ``dataset[i]`` decodes episode ``i``.

    Args:
        source: A local dataset folder (holding ``meta/info.json``) or a Hub
            repo id such as ``"lerobot/pusht"``.
        camera: Feature key of the view to use as ``frames``
            (``"observation.images.top"``, or just ``"top"``). Defaults to the
            first camera in name order.
        extra_cameras: Further views to keep as ``cameras``: ``"all"``, or a
            list of keys. ``()`` keeps none (less decoding).
        episodes: Episode indices to read; all when ``None``.
        revision: Hub revision (branch, tag or commit).
        cache_dir: Where Hub downloads go (the Hugging Face cache by default).
    """

    def __init__(
        self,
        source: str | Path,
        *,
        camera: Optional[str] = None,
        extra_cameras: str | Sequence[str] = (),
        episodes: Optional[Iterable[int]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        _require_pyarrow()
        self.source = str(source)
        local = Path(source)
        self._repo_id: Optional[str] = None
        if (local / "meta" / "info.json").exists():
            self.root = local
        else:
            self._repo_id = self.source
            self.root = _download(self.source, ["meta/*"], revision, cache_dir)
        self._revision = revision
        self._cache_dir = cache_dir
        self.info: Dict[str, Any] = json.loads(
            (self.root / "meta" / "info.json").read_text(encoding="utf-8")
        )
        self.version = str(self.info.get("codebase_version", "v2.1"))
        self.fps = float(self.info.get("fps", 30))
        self.features: Dict[str, Any] = self.info.get("features", {})
        self._v3 = self.version.startswith("v3")
        self._episodes = self._read_episode_table()
        self._tasks = self._read_tasks()
        if episodes is not None:
            wanted = set(int(e) for e in episodes)
            self._episodes = [
                e for e in self._episodes if int(e["episode_index"]) in wanted
            ]
        if not self._episodes:
            raise ValueError(f"{self.source} has no episodes to read.")

        views = sorted(
            key
            for key, spec in self.features.items()
            if spec.get("dtype") in ("video", "image")
        )
        if not views:
            raise ValueError(f"{self.source} has no camera features.")
        self.camera = _match_key(camera, views) if camera else views[0]
        if extra_cameras == "all":
            extras = [v for v in views if v != self.camera]
        else:
            extras = [_match_key(c, views) for c in extra_cameras]
        self.extra_cameras = extras
        self.action_key = "action" if "action" in self.features else None
        self.observation_keys = sorted(
            key
            for key, spec in self.features.items()
            if key.startswith("observation.")
            and spec.get("dtype") not in ("video", "image", "string")
        )
        if self._repo_id is not None:
            self._fetch_files()

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Episode:  # type: ignore[override]
        return self._read_episode(self._episodes[index])

    @property
    def lengths(self) -> List[int]:
        return [int(e["length"]) for e in self._episodes]

    @property
    def meta(self) -> DatasetMeta:
        """The episode-folder metadata this dataset converts to."""
        action = self.features.get(self.action_key or "", {})
        shape = action.get("shape") or [0]
        return DatasetMeta(
            action_kind="continuous" if self.action_key else None,
            action_dim=int(np.prod(shape)) if self.action_key else None,
            fps=self.fps,
            cameras=[_short(c) for c in self.extra_cameras],
            observation_keys=[k[len("observation.") :] for k in self.observation_keys],
            source=f"lerobot:{self.source}",
            info={
                "codebase_version": self.version,
                "robot_type": self.info.get("robot_type"),
                "camera": self.camera,
            },
        )

    # ------------------------------------------------------------- reading

    def _read_episode(self, record: Dict[str, Any]) -> Episode:
        import pyarrow.compute as pc

        index = int(record["episode_index"])
        table = self._read_table(self._data_path(record))
        table = table.filter(pc.equal(table["episode_index"], index))
        if "frame_index" in table.column_names:
            table = table.sort_by("frame_index")
        rows = table.num_rows
        if rows == 0:
            raise ValueError(
                f"Episode {index} has no rows in {self._data_path(record)}."
            )

        frames = self._read_view(self.camera, record, table)
        cameras = {
            _short(c): self._read_view(c, record, table) for c in self.extra_cameras
        }
        T = min([rows, len(frames), *(len(v) for v in cameras.values())])
        if T != rows or T != len(frames):
            logger.warning(
                "Episode %d: %d rows but %d decoded frames; keeping the first %d.",
                index,
                rows,
                len(frames),
                T,
            )

        def column(name: str) -> Optional[np.ndarray]:
            if name not in table.column_names:
                return None
            return _column_array(table, name)[:T]

        actions = column(self.action_key) if self.action_key else None
        rewards = column("next.reward")
        done = column("next.done")
        terminated = None
        if done is not None and T > 1:
            terminated = done[: T - 1].astype(bool).reshape(-1)
            terminated[-1] = terminated[-1] or bool(done[T - 1])
        observations = {
            key[len("observation.") :]: value
            for key in self.observation_keys
            if (value := column(key)) is not None
        }
        tasks = record.get("tasks") or []
        language = None
        if tasks:
            language = str(tasks[0])
        elif "task_index" in table.column_names and self._tasks:
            language = self._tasks.get(int(_column_array(table, "task_index")[0]))
        return Episode(
            frames=frames[:T],
            actions=None if actions is None else actions[: T - 1],
            rewards=None if rewards is None else rewards[: T - 1].reshape(-1),
            terminated=terminated,
            cameras={k: v[:T] for k, v in cameras.items()},
            observations=observations,
            language=language,
        )

    def _read_view(self, key: str, record: Dict[str, Any], table: Any) -> np.ndarray:
        if self.features[key].get("dtype") == "image":
            from io import BytesIO

            from PIL import Image

            images = []
            for cell in table.column(key).to_pylist():
                data = cell.get("bytes") if isinstance(cell, dict) else cell
                if data is None and isinstance(cell, dict) and cell.get("path"):
                    data = (self.root / cell["path"]).read_bytes()
                if data is None:
                    raise ValueError(
                        f"An image in '{key}' has neither bytes nor a path."
                    )
                images.append(np.asarray(Image.open(BytesIO(data)).convert("RGB")))
            return np.stack(images)
        path = self.root / self._video_path(key, record)
        if self._v3:
            start = float(record.get(f"videos/{key}/from_timestamp", 0.0))
            end = record.get(f"videos/{key}/to_timestamp")
            if end is None:
                end = start + int(record["length"]) / self.fps
            return read_video(path, start=start, end=float(end))
        return read_video(path)

    def _data_path(self, record: Dict[str, Any]) -> Path:
        template = self.info.get("data_path")
        if self._v3:
            template = (
                template or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            )
            return self.root / template.format(
                chunk_index=int(record.get("data/chunk_index", 0)),
                file_index=int(record.get("data/file_index", 0)),
            )
        template = template or (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        index = int(record["episode_index"])
        return self.root / template.format(
            episode_chunk=index // int(self.info.get("chunks_size", 1000)),
            episode_index=index,
        )

    def _video_path(self, key: str, record: Dict[str, Any]) -> str:
        template = self.info.get("video_path")
        if self._v3:
            template = template or (
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )
            return template.format(
                video_key=key,
                chunk_index=int(record.get(f"videos/{key}/chunk_index", 0)),
                file_index=int(record.get(f"videos/{key}/file_index", 0)),
            )
        template = template or (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        index = int(record["episode_index"])
        return template.format(
            episode_chunk=index // int(self.info.get("chunks_size", 1000)),
            video_key=key,
            episode_index=index,
        )

    def _read_episode_table(self) -> List[Dict[str, Any]]:
        meta = self.root / "meta"
        if self._v3:
            import pyarrow as pa

            files = sorted((meta / "episodes").rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No meta/episodes/*.parquet in {self.root}.")
            table = pa.concat_tables([self._read_table(f) for f in files])
            records = table.to_pylist()
        else:
            path = meta / "episodes.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"No meta/episodes.jsonl in {self.root}.")
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return sorted(records, key=lambda r: int(r["episode_index"]))

    def _read_tasks(self) -> Dict[int, str]:
        meta = self.root / "meta"
        if (meta / "tasks.parquet").exists():
            table = self._read_table(meta / "tasks.parquet")
            names = table.column_names
            task_col = next(
                (c for c in ("task", "__index_level_0__") if c in names),
                None,
            )
            if task_col is None or "task_index" not in names:
                return {}
            return {
                int(i): str(t)
                for i, t in zip(
                    table["task_index"].to_pylist(), table[task_col].to_pylist()
                )
            }
        if (meta / "tasks.jsonl").exists():
            tasks = {}
            for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    tasks[int(row["task_index"])] = str(row["task"])
            return tasks
        return {}

    def _fetch_files(self) -> None:
        """Download only the data and video files the selected episodes need."""
        patterns = set()
        for record in self._episodes:
            patterns.add(
                str(self._data_path(record).relative_to(self.root)).replace("\\", "/")
            )
            for key in (self.camera, *self.extra_cameras):
                if self.features[key].get("dtype") == "video":
                    patterns.add(self._video_path(key, record))
        assert self._repo_id is not None
        self.root = _download(
            self._repo_id, sorted(patterns), self._revision, self._cache_dir
        )

    @staticmethod
    def _read_table(path: Path) -> Any:
        import pyarrow.parquet as pq

        return pq.read_table(path)


def import_lerobot(
    source: str | Path,
    out_dir: str | Path,
    *,
    camera: Optional[str] = None,
    extra_cameras: str | Sequence[str] = (),
    episodes: Optional[Iterable[int]] = None,
    revision: Optional[str] = None,
) -> Path:
    """Convert a LeRobot dataset into an episode folder; returns ``out_dir``."""
    dataset = LeRobotEpisodes(
        source,
        camera=camera,
        extra_cameras=extra_cameras,
        episodes=episodes,
        revision=revision,
    )
    with EpisodeWriter(out_dir, meta=dataset.meta) as writer:
        for i in range(len(dataset)):
            writer.write(dataset[i])
    return Path(out_dir)


def _column_array(table: Any, name: str) -> np.ndarray:
    column = table.column(name)
    try:
        array = column.to_numpy()
    except Exception:
        array = np.asarray(column.to_pylist())
    if array.dtype == object:
        array = np.stack([np.asarray(v) for v in array])
    return array


def _match_key(name: str, keys: Sequence[str]) -> str:
    for candidate in (name, IMAGE_PREFIX + name):
        if candidate in keys:
            return candidate
    raise KeyError(f"No camera '{name}'; the dataset has: {', '.join(keys)}")


def _short(key: str) -> str:
    return key[len(IMAGE_PREFIX) :] if key.startswith(IMAGE_PREFIX) else key


def _download(
    repo_id: str,
    patterns: List[str],
    revision: Optional[str],
    cache_dir: Optional[str | Path],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - accelerate depends on it
        raise ImportError(
            "Reading LeRobot datasets from the Hub needs huggingface_hub."
        ) from exc
    path = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading LeRobot datasets needs PyArrow and PyAV: pip install simulusion[lerobot]"
        ) from exc


__all__ = ["LeRobotEpisodes", "import_lerobot"]
