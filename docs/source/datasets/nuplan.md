# NuPlan Dataset

The NuPlan dataset backend provides a PyTorch interface to the [Motional NuPlan](https://www.nuplan.org/) autonomous driving dataset. It loads real-world driving scenarios, rasterises local map tiles, and returns structured samples with ego vehicle history, agent trajectories, and planning targets — suitable for world model training in the behavioural planning domain.

## Install

```bash
pip install nuplan-devkit
```

NuPlan is not part of Simulusion's minimal dependencies. You also need a local copy of the NuPlan dataset. Download it from [nuplan.org](https://www.nuplan.org/nuplan) and unpack it to `~/nuplan/dataset` (or set `$NUPLAN_DATA_ROOT`).

The dataset is ~1.8 TB for the full split. For prototyping, use the mini split (`~/nuplan/dataset/mini`, ~13 GB).

## Main API

```python
from simulusion.datasets.nuplan import NuPlanDataset, make_nuplan_dataloader

# Build a dataset over the mini split.
dataset = NuPlanDataset(
    split="train",
    planning_horizon=80,  # 8 seconds at 10 Hz
    past_horizon=20,  # 2 seconds at 10 Hz
    map_extent=(100.0, 100.0),
    map_resolution=0.1,
    max_agents=32,
    limit_scenarios=100,  # remove for full dataset
)

sample = dataset[0]
# sample.map_raster      -> (3, H, W)  float32
# sample.ego_past        -> (20, 6)    float32
# sample.ego_future      -> (80, 2)    float32
# sample.agents_past     -> (32, 20, 6) float32
# sample.agents_future   -> (32, 80, 6) float32
# sample.agents_mask     -> (32,)      bool
# sample.planning_target -> (80, 2)    float32

# Or create a DataLoader directly.
dataset, loader = make_nuplan_dataloader(
    split="train",
    batch_size=16,
    num_workers=4,
    limit_scenarios=100,
)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `$NUPLAN_DATA_ROOT` or `~/nuplan/dataset` | Dataset root directory |
| `map_root` | `$NUPLAN_MAP_ROOT` or `~/nuplan/maps` | Map data root |
| `split` | `train` | One of `train`, `val`, `test` |
| `planning_horizon` | `80` | Future steps at 10 Hz |
| `past_horizon` | `20` | Past steps at 10 Hz |
| `map_extent` | `(100.0, 100.0)` | Raster crop half-extent in metres |
| `map_resolution` | `0.1` | Metres per pixel |
| `max_agents` | `32` | Max agents per sample (zero-padded) |
| `limit_scenarios` | `None` | Cap on total scenarios |

## Sample structure

```{eval-rst}
.. autoclass:: simulusion.datasets.nuplan.NuPlanSample
   :members:
   :noindex:
```

## Dataset class

```{eval-rst}
.. autoclass:: simulusion.datasets.nuplan.NuPlanDataset
   :members:
   :noindex:
```

## DataLoader factory

```{eval-rst}
.. autofunction:: simulusion.datasets.nuplan.make_nuplan_dataloader
   :noindex:
```

## Observation contract

Each sample returned by `NuPlanDataset.__getitem__` is a `NuPlanSample` dataclass with the following fields:

- **`map_raster`** — `(3, H, W)` float32 tensor. Channels 0, 1, 2 are binary masks for drivable area, lane centre-lines, and crosswalks respectively. The crop is centred on the ego vehicle and oriented in the ego frame.

- **`ego_past`** — `(past_horizon, 6)` float32 tensor with columns `(x, y, yaw, vx, vy, yaw_rate)`.

- **`ego_future`** — `(planning_horizon, 2)` float32 tensor with relative `(x, y)` waypoints from the last known ego position.

- **`agents_past` / `agents_future`** — Zero-padded tensors of shape `(max_agents, past_horizon, 6)` and `(max_agents, planning_horizon, 6)`. Use `agents_mask` to distinguish real agents from padding.

- **`planning_target`** — Alias of `ego_future` for supervised trajectory prediction.

## Integration with world models

The NuPlan dataset can be used to train world models that operate on structured driving scenarios. The map raster and agent trajectories serve as context inputs, while the future ego trajectory provides a regression target for the planning head. A typical workflow:

```python
dataset, loader = make_nuplan_dataloader(split="train", batch_size=32)

for epoch in range(num_epochs):
    for batch in loader:
        # batch is a dict with keys matching the NuPlanSample fields.
        map_raster = batch["map_raster"]  # (B, 3, H, W)
        ego_past = batch["ego_past"]  # (B, T_past, 6)
        agents_past = batch["agents_past"]  # (B, N, T_past, 6)
        agents_mask = batch["agents_mask"]  # (B, N)
        planning_target = batch["planning_target"]  # (B, T_future, 2)

        # forward through a world model ...
        loss = model(map_raster, ego_past, agents_past, agents_mask, planning_target)
        loss.backward()
```

## Environment variables

- **`NUPLAN_DATA_ROOT`** — Path to the dataset root (default: `~/nuplan/dataset`).
- **`NUPLAN_MAP_ROOT`** — Path to the map data (default: `~/nuplan/maps`).
