"""PyTorch Dataset for the NuPlan autonomous driving dataset.

Requires ``nuplan-devkit`` and a local copy of the NuPlan dataset.
Download from https://www.nuplan.org/nuplan and set ``NUPLAN_DATA_ROOT``
to the extracted path (default: ``~/nuplan/dataset``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NuPlanSample:
    """A single training sample from the NuPlan dataset."""

    scenario_name: str
    map_raster: torch.Tensor  # (C_map, H_map, W_map)
    ego_past: torch.Tensor  # (past_horizon, 6)  x, y, yaw, vx, vy, yaw_rate
    ego_future: torch.Tensor  # (planning_horizon, 2)  relative x, y
    agents_past: torch.Tensor  # (max_agents, past_horizon, 6)
    agents_future: torch.Tensor  # (max_agents, planning_horizon, 6)
    agents_mask: torch.Tensor  # (max_agents,)  bool — valid vs padded
    agent_types: torch.Tensor  # (max_agents,)  integer type codes
    planning_target: torch.Tensor  # (planning_horizon, 2)


NUPLAN_DATA_ROOT = Path(
    os.environ.get("NUPLAN_DATA_ROOT", Path.home() / "nuplan" / "dataset")
)
NUPLAN_MAP_ROOT = Path(
    os.environ.get("NUPLAN_MAP_ROOT", Path.home() / "nuplan" / "maps")
)


class NuPlanDataset(Dataset[NuPlanSample]):
    """PyTorch Dataset over NuPlan scenarios for world model training.

    Each sample contains rasterised map tiles, ego and agent history, and
    future planning targets at 10 Hz.

    Parameters
    ----------
    data_root:
        Path to the NuPlan dataset root. Defaults to ``$NUPLAN_DATA_ROOT``.
    map_root:
        Path to NuPlan map data. Defaults to ``$NUPLAN_MAP_ROOT``.
    split:
        ``"train"``, ``"val"``, or ``"test"``. The mini split is used
        automatically when ``data_root / "mini"`` exists.
    db_files:
        Explicit list of ``.db`` files. When ``None`` the builder
        auto-discovers files under ``data_root / split``.
    map_version:
        Map version string, e.g. ``"nuplan-maps-v1.0"``.
    planning_horizon:
        Number of future steps at 10 Hz (default 80 = 8 s).
    past_horizon:
        Number of past steps at 10 Hz (default 20 = 2 s).
    map_extent:
        Raster crop half-extent in metres ``(width, height)``.
    map_resolution:
        Metres per pixel for the raster.
    max_agents:
        Maximum agents per sample; fewer are zero-padded.
    limit_scenarios:
        Cap on total scenarios (useful for prototyping).
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        map_root: str | Path | None = None,
        split: str = "train",
        db_files: list[str] | None = None,
        map_version: str = "nuplan-maps-v1.0",
        planning_horizon: int = 80,
        past_horizon: int = 20,
        map_extent: Tuple[float, float] = (100.0, 100.0),
        map_resolution: float = 0.1,
        max_agents: int = 32,
        limit_scenarios: int | None = None,
    ):
        super().__init__()

        self.data_root = Path(data_root) if data_root else NUPLAN_DATA_ROOT
        self.map_root = Path(map_root) if map_root else NUPLAN_MAP_ROOT
        self.split = split
        self.planning_horizon = planning_horizon
        self.past_horizon = past_horizon
        self.map_extent = map_extent
        self.map_resolution = map_resolution
        self.max_agents = max_agents

        log_path = self._resolve_log_path()
        self._scenarios = self._load_scenarios(
            log_path, db_files, map_version, limit_scenarios
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> NuPlanSample:
        scenario = self._scenarios[idx]

        map_raster = _rasterise_map(scenario, self.map_extent, self.map_resolution)
        ego_past, ego_future = _extract_ego(
            scenario, self.past_horizon, self.planning_horizon
        )
        agents_past, agents_future, agents_mask, agent_types = _extract_agents(
            scenario, self.past_horizon, self.planning_horizon, self.max_agents
        )

        return NuPlanSample(
            scenario_name=getattr(scenario, "token", str(idx)),
            map_raster=map_raster,
            ego_past=ego_past,
            ego_future=ego_future,
            agents_past=agents_past,
            agents_future=agents_future,
            agents_mask=agents_mask,
            agent_types=agent_types,
            planning_target=ego_future.clone(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_log_path(self) -> Path:
        """Pick the correct data directory (mini or full split)."""
        mini = self.data_root / "mini"
        if mini.is_dir():
            return mini
        split_dir = self.data_root / self.split
        if split_dir.is_dir():
            return split_dir
        if self.data_root.is_dir():
            return self.data_root
        raise FileNotFoundError(
            f"NuPlan data not found at {self.data_root}. "
            f"Set $NUPLAN_DATA_ROOT or pass a valid data_root."
        )

    def _load_scenarios(
        self,
        log_path: Path,
        db_files: list[str] | None,
        map_version: str,
        limit: int | None,
    ) -> list[Any]:
        """Build scenarios via the nuplan-devkit."""
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )

        builder = NuPlanScenarioBuilder(
            data_root=str(log_path),
            map_root=str(self.map_root),
            map_version=map_version,
            db_files=db_files,
        )

        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

        scenario_filter = ScenarioFilter(
            limit_total_scenarios=limit or 0,
        )
        scenarios = builder.get_scenarios(scenario_filter)
        return list(scenarios)


# ---------------------------------------------------------------------------
# Stateless helpers  (all accept a scenario object from nuplan-devkit)
# ---------------------------------------------------------------------------


def _rasterise_map(
    scenario: Any, extent: Tuple[float, float], resolution: float
) -> torch.Tensor:
    """Render a local map crop around the ego vehicle.

    Returns ``(3, H, W)`` float tensor with channels:
    0 — drivable area, 1 — lane centre-lines, 2 — crosswalks.
    """
    from nuplan.common.maps.nuplan_map.map_api import NuPlanMapAPI

    map_api = NuPlanMapAPI(
        map_root=str(scenario.map_root),
        map_version=scenario.map_version,
    )

    ego_pose = _get_ego_pose(scenario)
    ego_x, ego_y, ego_yaw = ego_pose

    w = int(extent[0] / resolution)
    h = int(extent[1] / resolution)
    canvas = np.zeros((3, h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2

    roadblocks = map_api.get_all_map_objects(ego_pose[:2], extent[0])
    _draw_geometries(canvas, roadblocks, ego_x, ego_y, ego_yaw, resolution, cx, cy)

    return torch.from_numpy(canvas)


def _draw_geometries(
    canvas: np.ndarray,
    roadblocks: list[Any],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    resolution: float,
    cx: int,
    cy: int,
) -> None:
    """Rasterise road-block polygons onto a fixed-size canvas."""
    from shapely import affinity

    for rb in roadblocks:
        geom = rb.polygon
        if geom is None or geom.is_empty:
            continue
        # transform into ego-relative coordinates then raster space
        geom = affinity.affine_transform(geom, [1, 0, 0, 1, -ego_x, -ego_y])
        geom = affinity.rotate(geom, -ego_yaw, origin=(0, 0), use_radians=True)
        geom = affinity.scale(
            geom, xfact=1 / resolution, yfact=1 / resolution, origin=(0, 0)
        )
        geom = affinity.translate(geom, xoff=cx, yoff=cy)

        coords = _polygon_to_mask(geom, canvas.shape[1], canvas.shape[2])
        if coords is None:
            continue
        if getattr(rb, "is_drivable", False):
            canvas[0, coords[0], coords[1]] = 1.0
        canvas[1, coords[0], coords[1]] = 1.0


def _polygon_to_mask(
    geom: Any, h: int, w: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert a Shapely (multi)polygon to ``(rows, cols)`` index arrays clipped to ``[0, h)``, ``[0, w)``."""
    from shapely import contains, prepare, points

    if geom.is_empty:
        return None
    if hasattr(geom, "geoms"):  # MultiPolygon
        row_parts, col_parts = [], []
        for sub in geom.geoms:
            result = _polygon_to_mask(sub, h, w)
            if result is not None:
                row_parts.append(result[0])
                col_parts.append(result[1])
        return (
            (np.concatenate(row_parts), np.concatenate(col_parts))
            if row_parts
            else None
        )

    min_x, min_y, max_x, max_y = map(int, geom.bounds)
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, w - 1)
    max_y = min(max_y, h - 1)
    if min_x >= max_x or min_y >= max_y:
        return None

    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    xx, yy = np.meshgrid(xs, ys)
    pts = points(np.stack([xx.ravel(), yy.ravel()], axis=1))
    prepare(geom)
    mask = contains(geom, pts).reshape(len(ys), len(xs))
    rows, cols = np.where(mask)
    return (rows + min_y, cols + min_x)


def _extract_ego(
    scenario: Any, past_horizon: int, future_horizon: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Past ego trajectory and relative future waypoints."""
    history = scenario.get_past_ego_states(
        num_steps=past_horizon, time_horizon=past_horizon * 0.1
    )
    future = scenario.get_future_ego_states(
        num_steps=future_horizon, time_horizon=future_horizon * 0.1
    )

    def _pack(states: Any, n: int) -> torch.Tensor:
        if not states:
            return torch.zeros(n, 6)
        arr = np.array(
            [[s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate] for s in states],
            dtype=np.float32,
        )
        return torch.from_numpy(arr)

    past = _pack(history, past_horizon)
    fwd = _pack(future, future_horizon)
    future_xy = fwd[:, :2] - past[-1:, :2]
    return past, future_xy


def _extract_agents(
    scenario: Any, past_horizon: int, future_horizon: int, max_agents: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agent trajectories with zero-padding to ``max_agents``."""
    tracks = scenario.get_tracked_agents(
        past_horizon=past_horizon, future_horizon=future_horizon
    )

    n = min(len(tracks), max_agents)
    past = torch.zeros(max_agents, past_horizon, 6)
    future = torch.zeros(max_agents, future_horizon, 6)
    mask = torch.zeros(max_agents, dtype=torch.bool)
    types = torch.zeros(max_agents, dtype=torch.long)

    for i, track in enumerate(tracks[:n]):
        mask[i] = True
        types[i] = int(getattr(track, "track_type", 0))
        if track.past_trajectory:
            arr = np.array(
                [
                    [s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate]
                    for s in track.past_trajectory
                ],
                dtype=np.float32,
            )
            past[i, : len(arr)] = torch.from_numpy(arr)
        if track.future_trajectory:
            arr = np.array(
                [
                    [s.x, s.y, s.yaw, s.vx, s.vy, s.yaw_rate]
                    for s in track.future_trajectory
                ],
                dtype=np.float32,
            )
            future[i, : len(arr)] = torch.from_numpy(arr)

    return past, future, mask, types


def _get_ego_pose(scenario: Any) -> Tuple[float, float, float]:
    """(x, y, yaw) of the ego vehicle at the current scenario iteration."""
    state = scenario.get_ego_state_at_iteration(scenario.iteration)
    return (float(state.x), float(state.y), float(state.yaw))


def make_nuplan_dataloader(
    data_root: str | Path | None = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs: Any,
) -> Tuple[NuPlanDataset, torch.utils.data.DataLoader]:
    """Create a NuPlan DataLoader.

    Parameters
    ----------
    data_root:
        Root of the NuPlan dataset (default: ``$NUPLAN_DATA_ROOT``).
    split:
        Dataset split.
    batch_size:
        Batch size.
    num_workers:
        Worker count for the DataLoader.
    **dataset_kwargs:
        Extra arguments forwarded to :class:`NuPlanDataset`.

    Returns
    -------
    (dataset, dataloader):
    """
    dataset = NuPlanDataset(data_root=data_root, split=split, **dataset_kwargs)

    def collate(batch: list[NuPlanSample]) -> dict[str, Any]:
        return {
            "scenario_name": [b.scenario_name for b in batch],
            "map_raster": torch.stack([b.map_raster for b in batch]),
            "ego_past": torch.stack([b.ego_past for b in batch]),
            "ego_future": torch.stack([b.ego_future for b in batch]),
            "agents_past": torch.stack([b.agents_past for b in batch]),
            "agents_future": torch.stack([b.agents_future for b in batch]),
            "agents_mask": torch.stack([b.agents_mask for b in batch]),
            "agent_types": torch.stack([b.agent_types for b in batch]),
            "planning_target": torch.stack([b.planning_target for b in batch]),
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return dataset, dataloader
