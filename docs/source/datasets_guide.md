# Datasets

Simulusion provides dataset loaders for image, video, RL trajectory, and
curated benchmark data. All major datasets are accessible through the
top-level package or the `simulusion.datasets` module.

```{contents} Contents
:depth: 3
```

## Which dataset to use

| Dataset | Class / Factory | Pipe | Used by |
|---|---|---|---|
| **CIFAR-10** | `make_cifar10(root_path, transform, ...)` | Image classification | JEPA, DiT |
| **ImageNet-1K** | `make_imagenet1k(root_path, transform, ...)` | Image classification | JEPA |
| **ImageFolder** | `make_imagefolder(root_path, ...)` | Custom image folders | JEPA |
| **Video folder** | `VideoFolderDataset(path, num_frames, ...)` | Video files (.mp4, .avi) | Genie |
| **Image sequence** | `ImageFolderDataset(path, num_frames, ...)` | Per-frame folder sequences | Genie |
| **NumPy arrays** | `NumPyDataset(path, ...)` | Pre-encoded .npy/.npz | Genie |
| **HDF5** | `HDF5Dataset(path, key, ...)` | HDF5 video stores | Genie |
| **RL trajectories** | `RLEnvironmentDataset(path, ...)` | .npz episode files | World Models pipeline |
| **Rollout data** | `RolloutDataset(root, ...)` | Pre-collected .npz rollouts | World Models pipeline |
| **TinyWorlds** | `TinyWorldsDataset(name, ...)` | HuggingFace video datasets | Genie |
| **DIAMOND replay** | `ReplayBuffer(capacity, ...)` | Online Atari interaction | DIAMOND |

## Image datasets

### CIFAR-10

```python
from simulusion.datasets.cifar10 import make_cifar10

dataset, loader, sampler = make_cifar10(
    transform=transform,
    batch_size=256,
    root_path="./data",
    download=True,
    world_size=1,
    rank=0,
)
```

Returns a `torchvision.datasets.CIFAR10` wrapped in a `DistributedSampler`
dataloader. Used for JEPA and DiT prototyping.

| Arg | Default | Description |
|---|---|---|
| `transform` | required | Torchvision transforms |
| `batch_size` | required | Samples per batch |
| `root_path` | `None` | Data directory |
| `download` | `False` | Download if missing |
| `train` | `True` | Train or test split |
| `world_size` / `rank` | `1` / `0` | Distributed training |

### ImageNet-1K

```python
from simulusion.datasets.imagenet1k import make_imagenet1k

dataset, loader, sampler = make_imagenet1k(
    transform=transform,
    batch_size=256,
    root_path="/data/imagenet",
    image_folder="imagenet_full_size/061417/",
    training=True,
    copy_data=False,  # set True for SLURM with network storage
)
```

The `ImageNet` class extends `torchvision.datasets.ImageFolder` with:

| Feature | Description |
|---|---|
| **Local data staging** | Copies tar archives from network storage to `/scratch/slurm_tmpdir/{job_id}/` and extracts once per SLURM job |
| **Subset filtering** | `ImageNetSubset` restricts to a text-file list of allowed image IDs |

```python
# Custom image folder (any directory structure)
from simulusion.datasets.imagenet1k import make_imagefolder

dataset, loader, sampler = make_imagefolder(
    transform=transform,
    batch_size=64,
    root_path="./my_dataset",
    image_folder="train",
    val_split=0.1,  # hold out 10% for validation
)
```

## Video datasets

All video datasets inherit from `VideoDatasetBase` and share the same
interface:

```python
dataset = VideoFolderDataset(
    data_source="/path/to/videos",
    num_frames=16,
    image_size=64,
    transform=None,
    normalize=True,
)
video = dataset[0]  # (16, 3, 64, 64) float
```

### `VideoFolderDataset` — raw video files

```python
from simulusion.datasets.video_datasets import VideoFolderDataset

dataset = VideoFolderDataset(
    data_source="/data/videos",
    num_frames=16,
    image_size=64,
    extensions=(".mp4", ".avi", ".mkv", ".mov"),
    recursive=True,
)
```

Scans a directory for video files, loads them with OpenCV, samples
`num_frames` uniformly, and resizes to `image_size`.

### `ImageFolderDataset` — per-frame sequences

```python
from simulusion.datasets.video_datasets import ImageFolderDataset

dataset = ImageFolderDataset(
    data_source="/data/sequences",
    num_frames=16,
    image_size=64,
)

# Structure:
# /data/sequences/sequence_001/
#   frame_0001.jpg
#   frame_0002.jpg
#   ...
```

Each subfolder is a video sequence. Images are sorted by filename (numeric
stems first, then lexicographic). Shorter sequences pad with the last frame.

### `NumPyDataset` — pre-encoded numpy arrays

```python
from simulusion.datasets.video_datasets import NumPyDataset

# .npy file with shape (N, T, H, W, C)
dataset = NumPyDataset(
    data_source="/data/videos.npy",
    num_frames=16,
    image_size=64,
)

# .npz file
dataset = NumPyDataset(
    data_source="/data/videos.npz",
    key="videos",
)
```

Supports both `.npy` and `.npz` files. For `.npz`, specify the array key.

### `HDF5Dataset` — HDF5 video stores

```python
from simulusion.datasets.video_datasets import HDF5Dataset

dataset = HDF5Dataset(
    data_source="/data/videos.h5",
    key="videos",
    num_frames=16,
    image_size=64,
    memmap=False,  # set True for large files
)
```

Supports layouts `(N, T, H, W, C)`, `(N, T, C, H, W)`, and `(N, T, H, W)`.
The `memmap=True` option reads on demand instead of loading into RAM.

### Factory function

```python
from simulusion.datasets.video_datasets import create_video_dataloader

dataset, loader = create_video_dataloader(
    dataset_type="video_folder",  # "video_folder" | "image_folder" | "numpy" | "rl"
    data_source="/path/to/data",
    num_frames=16,
    image_size=64,
    batch_size=4,
)
```

## RL trajectory datasets

### `RLEnvironmentDataset` — episode recordings

```python
from simulusion.datasets.video_datasets import RLEnvironmentDataset

dataset = RLEnvironmentDataset(
    data_source="/data/episodes",
    num_frames=16,
    image_size=64,
    obs_key="observations",
)
```

Loads `.npz` files containing RL episodes. Each `.npz` should have an
`obs_key` entry with shape `(T, ...)`. Supports dict observations
(preferring `"image"` or `"pixels"` keys), single `.npz` files, or
directories of `.npz` files.

### `RolloutDataset` — World Models pipeline

```python
from simulusion.datasets.wm_dataset import RolloutDataset

dataset = RolloutDataset(
    root="data/carracing",
    transform=transform,
    train=True,
    buffer_size=100,
    num_test_files=10,
)
obs, action, reward, terminal = dataset[0]
```

Loads pre-collected rollout `.npz` files for the classic World Models
pipeline (VAE → MDNRNN → Controller). Each `.npz` contains observation
sequences, actions, rewards, and terminal flags.

## TinyWorlds

Curated game-video datasets from HuggingFace for training Genie-style
world models.

```python
from simulusion.datasets.tinyworlds import (
    TinyWorldsDataset,
    TinyWorldsDataLoader,
    create_tinyworlds_dataloader,
    download_all_datasets,
)
```

### Available datasets

| Name | Description | Size |
|---|---|---|
| `PICO_DOOM` | Minimal Doom gameplay | ~50K videos |
| `PONG` | Classic Pong | ~50K videos |
| `ZELDA` | Zelda Ocarina of Time (2D) | ~50K videos |
| `POLE_POSITION` | Racing game | ~50K videos |
| `SONIC` | Sonic the Hedgehog | ~50K videos |

### Usage

```python
# Single dataset
dataset, loader = create_tinyworlds_dataloader(
    dataset_name="SONIC",
    num_frames=16,
    image_size=64,
    batch_size=4,
    download=True,
)

# List available datasets
from simulusion.datasets.tinyworlds import TinyWorldsDataLoader

print(TinyWorldsDataLoader.list_available_datasets())

# Get metadata without downloading
info = TinyWorldsDataLoader.get_dataset_info("ZELDA")

# Download all datasets at once
paths = download_all_datasets()  # returns dict of name → local path

# Direct usage with cache
dataset = TinyWorldsDataset(
    dataset_name="PONG",
    num_frames=16,
    image_size=64,
    download=True,
    cache_dir="~/.cache/tinyworlds",
)
```

Data is downloaded from HuggingFace (`AlmondGod/tinyworlds`) and cached
locally. Requires `h5py` and `huggingface_hub`.

## DIAMOND replay buffer

```python
from simulusion.datasets.diamond_dataset import ReplayBuffer, SequenceDataset

buffer = ReplayBuffer(capacity=100000, obs_shape=(64, 64, 3), action_dim=4)
buffer.add(obs, action, reward, done, next_obs)

dataset = SequenceDataset(buffer, sequence_length=5, burn_in=4)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

See the [Memory guide](memory_guide.md) for full details on replay buffers.

## See Also

- {doc}`jepa` — uses CIFAR-10 and ImageNet-1K datasets
- {doc}`genie` — uses TinyWorlds datasets and video datasets
- {doc}`diamond` — uses DIAMOND replay buffer (Atari)
- {doc}`dreamer` — uses environment interaction data (not static datasets)
- {doc}`memory_guide` — replay buffer details for Dreamer, IRIS, PlaNet
