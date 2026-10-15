import os
import torch
import pickle
import pathlib
import numpy as np
import glob
import warnings
from typing import Any, Dict, List


try:
    import plotly
    from plotly.graph_objs import Scatter, Line

    from sklearn.manifold import TSNE
    import umap

    HAS_VIZ = True
except ImportError:
    plotly = None
    HAS_VIZ = False

from collections import defaultdict
from simulusion.memory.planet_memory import Memory

from torchvision.utils import make_grid, save_image

import torch.nn.functional as F

import yaml


class _RestrictedReplayUnpickler(pickle.Unpickler):
    """Unpickler that only resolves classes needed by replay buffers."""

    _ALLOWED_GLOBALS = {
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "slice"),
        ("builtins", "tuple"),
        ("collections", "deque"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("simulusion.memory.planet_memory", "Episode"),
        ("simulusion.memory.planet_memory", "Memory"),
        ("simulusion.memory.planet_memory", "_identity"),
    }

    def find_class(self, module: str, name: str) -> type:
        if (module, name) in self._ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"global '{module}.{name}' is not allowed in replay buffers"
        )


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def load_yml_config(path: str) -> AttrDict | None:
    with open(path) as fileStream:
        loaded = yaml.safe_load(fileStream)
        dictionary = AttrDict(loaded)
        return dictionary


def to_tensor_obs(image: np.ndarray) -> torch.Tensor:
    """
    Converts the input np img to channel first 64x64 dim torch img.
    """
    import cv2

    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    tensor: torch.Tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    return tensor


def postprocess_img(image: np.ndarray, depth: int) -> np.ndarray:
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2**depth)
    return np.clip(image * 2 ** (8 - depth), 0, 2**8 - 1).astype(np.uint8)


def preprocess_img(image: torch.Tensor, depth: int) -> None:
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    Also adds some noise to the observations !!
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2**depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2**depth)).clamp_(-0.5, 0.5)


def bottle(func: Any, *tensors: torch.Tensor) -> torch.Tensor:
    """
    Evaluates a func that operates in N x D with inputs of shape N x T x D
    """
    n, t = tensors[0].shape[:2]
    inputs = [x.reshape(n * t, *x.shape[2:]) for x in tensors]
    out = func(*inputs)
    return out.view(n, t, *out.shape[1:])


def get_combined_params(*models: Any) -> list:
    """
    Returns the combine parameter list of all the models given as input.
    """
    params = []
    for model in models:
        params.extend(list(model.parameters()))
    return params


class Mp4Writer:
    """Write BGR uint8 frames to a browser-playable MP4.

    Browsers only decode H.264 in MP4, but OpenCV's ``mp4v`` fourcc produces
    MPEG-4 Part 2, which no browser plays. Backends are tried in order:

    1. ``imageio-ffmpeg`` (bundled ffmpeg, libx264, yuv420p, faststart) -
       H.264 on every platform. Installed by the ``viz`` extra.
    2. OpenCV ``avc1`` - H.264 where the OpenCV build has an encoder
       (typically Windows/macOS; usually not pip wheels on Linux).
    3. OpenCV ``mp4v`` - always available, but not browser-playable; a
       warning is emitted.

    Args:
        path: output ``.mp4`` path
        fps: frames per second
        size: ``(width, height)`` of every frame
    """

    def __init__(self, path: str, fps: float, size: tuple[int, int]) -> None:
        self.path = str(path)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        self.codec: str
        self._ffmpeg: Any = None
        self._cv2: Any = None
        try:
            import imageio_ffmpeg

            self._ffmpeg = imageio_ffmpeg.write_frames(
                self.path,
                self.size,
                fps=self.fps,
                codec="libx264",
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                # yuv420p needs even dimensions; pad rather than rescale.
                macro_block_size=1,
                output_params=[
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags",
                    "+faststart",
                ],
                ffmpeg_log_level="error",
            )
            self._ffmpeg.send(None)
            self.codec = "h264"
            return
        except ImportError:
            pass

        import cv2

        for fourcc, codec in (("avc1", "h264"), ("mp4v", "mpeg4")):
            writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter.fourcc(*fourcc), self.fps, self.size, True
            )
            if writer.isOpened():
                self._cv2 = writer
                self.codec = codec
                break
            writer.release()
        else:
            raise RuntimeError(f"Could not open a video writer for {self.path}")
        if self.codec != "h264":
            warnings.warn(
                f"{self.path} is encoded as MPEG-4 Part 2 (mp4v), which browsers "
                "cannot play. Install imageio-ffmpeg (`pip install simulusion[viz]`) "
                "for H.264 output.",
                stacklevel=2,
            )

    def write(self, frame: np.ndarray) -> None:
        """Append one HWC (or HW) uint8 BGR frame of size ``self.size``."""
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(
                f"Frame size {(frame.shape[1], frame.shape[0])} does not match "
                f"writer size {self.size}"
            )
        if self._ffmpeg is not None:
            self._ffmpeg.send(frame)
        else:
            self._cv2.write(frame)

    def release(self) -> None:
        if self._ffmpeg is not None:
            self._ffmpeg.close()
            self._ffmpeg = None
        if self._cv2 is not None:
            self._cv2.release()
            self._cv2 = None

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def save_video(frames: Any, path: str, name: str) -> str:
    """
    Saves a video containing frames.

    Accepts frames in either:
      - (T, C, H, W) float in [0,1]
      - (T, H, W, C) float in [0,1]

    Produces {path}/{name}.mp4 and a debug PNG {path}/{name}_debug_frame.png
    with per-channel statistics printed to stdout.
    """
    # use top-level numpy (np) to keep imports top-level

    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(
            f"Expected frames with 4 dims (T, C, H, W) or (T, H, W, C), got shape {frames.shape}"
        )

    # detect layout
    if frames.shape[1] in (1, 3, 4):
        # (T, C, H, W)
        is_chw = True
    elif frames.shape[-1] in (1, 3, 4):
        # (T, H, W, C)
        is_chw = False
    else:
        raise ValueError(f"Can't infer channel axis from frames.shape={frames.shape}")

    # convert floats -> uint8 and to HWC format for OpenCV
    if is_chw:
        # (T, C, H, W) -> (T, H, W, C)
        frames_u8 = (frames * 255.0).clip(0, 255).astype("uint8").transpose(0, 2, 3, 1)
    else:
        frames_u8 = (frames * 255.0).clip(0, 255).astype("uint8")

    # Basic per-frame / per-channel sanity checks on first frame
    if frames_u8.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unexpected channel count: {frames_u8.shape[-1]}")

    first = frames_u8[0]
    ch = first.shape[-1]
    stats: Dict[str, List[Any]] = {"min": [], "max": [], "mean": []}
    for c in range(ch):
        stats["min"].append(int(first[..., c].min()))
        stats["max"].append(int(first[..., c].max()))
        stats["mean"].append(float(first[..., c].mean()))
    equal_ch: bool = False
    if ch >= 3:
        equal_ch = bool(
            np.all(first[..., 0] == first[..., 1])
            and np.all(first[..., 1] == first[..., 2])
        )

    import cv2

    out_dir = pathlib.Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = out_dir / f"{name}_debug_frame.png"
    try:
        to_write = first
        if first.ndim == 2 or first.shape[-1] == 1:
            to_write = np.repeat(first[..., None], 3, axis=-1)
        cv2.imwrite(str(debug_path), to_write[..., ::-1])
    except Exception as e:
        print(f"Failed to write debug frame PNG: {e}")

    print(
        f"[save_video] frames.shape={frames.shape} inferred_chw={is_chw} -> written_frame_shape={frames_u8.shape}"
    )
    print(
        f"[save_video] first_frame stats min={stats['min']} max={stats['max']} mean={stats['mean']} equal_rgb_channels={equal_ch}"
    )
    print(f"[save_video] debug PNG saved to: {debug_path}")

    # Write video with OpenCV (expecting HWC uint8 BGR)
    H, W = (None, None)
    if frames_u8.ndim == 4:
        H, W = frames_u8.shape[1], frames_u8.shape[2]
    else:
        raise RuntimeError("Unexpected frames_u8 shape after conversion")

    writer = Mp4Writer(str(out_dir / f"{name}.mp4"), 25.0, (W, H))
    try:
        for frame in frames_u8:
            # ensure contiguous HWC uint8
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            # OpenCV expects BGR
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
    return str(out_dir / f"{name}.mp4")


def combine_videos(
    video_dir: str,
    output_name: str = "combined.mp4",
    pattern: str = "vid_*.mp4",
    fps: int = 25,
    resize: bool = True,
) -> str:
    """
    Combine all videos matching `pattern` in `video_dir` into a single MP4 file.
    Returns the output filepath (string).

    Example:
      combine_videos("results/planet", output_name="all_training.mp4")
    """
    import cv2

    files = sorted(glob.glob(os.path.join(video_dir, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No videos found in {video_dir} matching {pattern}")

    # probe first video for size
    cap0 = cv2.VideoCapture(files[0])
    if not cap0.isOpened():
        cap0.release()
        raise RuntimeError(f"Failed to open video {files[0]}")
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    out_path = str(pathlib.Path(video_dir) / output_name)
    writer = Mp4Writer(out_path, float(fps), (width, height))

    try:
        for f in files:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                cap.release()
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # frame is BGR height x width x channels
                if resize and (frame.shape[1] != width or frame.shape[0] != height):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
            cap.release()
    finally:
        writer.release()
    return out_path


def ensure_results_dir_exists(results_dir: str) -> None:
    """
    Simple helper to validate a results directory exists.
    Raises FileNotFoundError if not present.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")


def save_frames(
    target: torch.Tensor,
    pred_prior: torch.Tensor,
    pred_posterior: torch.Tensor,
    name: str,
    n_rows: int = 5,
) -> None:
    """
    Save side-by-side target, prior-prediction, and posterior-prediction frames.

    The function accepts tensors with optional time dimension and writes a PNG
    grid to ``{name}.png``. Spatial sizes are aligned per timestep before
    concatenation and values are normalized to ``[0, 1]`` when needed.
    """

    def ensure_time_dim(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() == 3:
            return x.unsqueeze(0)
        return x

    tgt = ensure_time_dim(target).float()
    pp = ensure_time_dim(pred_prior).float()
    ppp = ensure_time_dim(pred_posterior).float()

    T_pred = pp.shape[0]
    T_tgt = max(0, tgt.shape[0] - 1)
    n = min(T_pred, T_tgt if T_tgt > 0 else T_pred)
    if n == 0:
        n = min(T_pred, tgt.shape[0])

    frames = []
    for t in range(n):
        if tgt.shape[0] > 1 and t + 1 < tgt.shape[0]:
            tf = tgt[t + 1]
        else:
            tf = tgt[min(t, tgt.shape[0] - 1)]
        pr = pp[min(t, pp.shape[0] - 1)]
        ppr = ppp[min(t, ppp.shape[0] - 1)]

        H, W = tf.shape[1], tf.shape[2]

        def match_size(img: torch.Tensor, H: int, W: int) -> torch.Tensor:
            img4 = img.unsqueeze(0)
            if img4.shape[2] != H or img4.shape[3] != W:
                img4 = F.interpolate(
                    img4, size=(H, W), mode="bilinear", align_corners=False
                )
            return img4.squeeze(0)

        tf = match_size(tf, H, W)
        pr = match_size(pr, H, W)
        ppr = match_size(ppr, H, W)

        def clamp01(x: torch.Tensor) -> torch.Tensor:
            if x.min() < 0 or x.max() > 1:
                return (x - x.min()) / (x.max() - x.min() + 1e-8)
            return x

        tf = clamp01(tf)
        pr = clamp01(pr)
        ppr = clamp01(ppr)

        # concatenate side-by-side along width (dim=2 because [C,H,W])
        row = torch.cat([tf, pr, ppr], dim=2)
        frames.append(row)

    if len(frames) == 0:
        raise RuntimeError(f"No frames to save for {name}")

    grid = make_grid(torch.stack(frames), nrow=n_rows, normalize=False)
    out_dir = os.path.dirname(name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image(grid, f"{name}.png")


def get_mask(tensor: Any, lengths: Any) -> torch.Tensor:
    """
    Build a batch-first validity mask from sequence lengths.

    ``tensor`` may be a tensor/array with shape ``(N, T, ...)`` or ``(N,)``.
    The returned mask marks valid timesteps with ones up to each element in
    ``lengths`` and preserves device/dtype conventions from the input.
    """
    # convert numpy -> torch if needed
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    # If tensor has no time axis (e.g. shape [N]), create mask with T = max(lengths)
    if tensor.dim() == 1:
        N = tensor.shape[0]
        T = int(max(lengths))
        mask = torch.zeros((N, T), dtype=tensor.dtype, device=tensor.device)
        for i in range(N):
            mask[i, : int(lengths[i])] = 1.0
        return mask

    # If tensor has time axis as second dim (batch-first: [N, T, ...])
    mask = torch.zeros_like(tensor)
    N = tensor.shape[0]
    for i in range(N):
        mask[i, : int(lengths[i])] = 1.0
    return mask


def load_memory(path: str, device: torch.device, *, trusted: bool = False) -> Any:
    """
    Loads an experience replay buffer.

    Pickle can execute arbitrary code during unrestricted deserialization, so
    user-supplied replay buffers are always loaded with a restricted unpickler
    that only allows the replay buffer classes and numpy containers required by
    historical buffers. The ``trusted`` argument is retained for backwards
    compatibility, but it no longer enables unrestricted pickle loading.

    Converts legacy list/.data formats into the current Memory(episodes) object.
    """
    if trusted:
        warnings.warn(
            "load_memory(trusted=True) is deprecated and no longer enables "
            "unrestricted pickle loading.",
            DeprecationWarning,
            stacklevel=2,
        )

    with open(path, "rb") as f:
        memory = _RestrictedReplayUnpickler(f).load()

    # If file contains a plain list of Episode objects -> wrap into Memory
    if isinstance(memory, list):
        mem = Memory(len(memory))
        mem.append(memory)
        memory = mem

    # If old object had `.data` attribute (legacy) -> convert to Memory
    if hasattr(memory, "data"):
        try:
            mem = Memory(len(memory.data))
            mem.append(memory.data)
            memory = mem
        except Exception:
            # fallback: just attach device to elements
            for e in memory.data:
                setattr(e, "device", device)

    # If object already has `.episodes`, ensure devices are set
    if hasattr(memory, "episodes"):
        for e in memory.episodes:
            setattr(e, "device", device)

    # Final attach device for the container itself
    setattr(memory, "device", device)
    return memory


def flatten_dict(data: dict, sep: str = ".", prefix: str = "") -> dict:
    """Flattens a nested dict into a single-level dict.

    Example:
      {'a': 2, 'b': {'c': 20}} -> {'a': 2, 'b.c': 20}
    """

    def build_key(parent: str, child: str) -> str:
        return f"{parent}{sep}{child}" if parent else str(child)

    flattened = {}

    for key, val in data.items():
        flat_key = build_key(prefix, key)
        if isinstance(val, dict):
            flattened.update(flatten_dict(val, sep=sep, prefix=key))
        else:
            flattened[flat_key] = val
    return flattened


def normalize_frames_for_saving(frames: Any) -> np.ndarray:
    """
    Ensure frames are in shape (T, H, W, 3) with float values in [0,1].
    Handles inputs in (T, C, H, W) or (T, H, W, C), repeats single-channel -> RGB,
    drops alpha if present, and maps [-0.5,0.5] -> [0,1] when detected.
    """
    # use top-level numpy (np) to keep imports at module scope

    frames = np.asarray(frames).astype(np.float32)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames array, got shape {frames.shape}")

    if frames.shape[1] in (1, 3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    elif frames.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Can't infer channel axis from frames.shape={frames.shape}")

    ch = frames.shape[-1]
    if ch == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif ch == 4:
        frames = frames[..., :3]

    mn, mx = float(frames.min()), float(frames.max())
    if mn >= -0.6 and mx <= 0.6:
        frames = (frames + 0.5).clip(0.0, 1.0)
    else:
        frames = frames.clip(0.0, 1.0)

    return frames


class TensorBoardMetrics:
    """Plots and (optionally) stores metrics for an experiment."""

    def __init__(self, path: str) -> None:
        self.steps: defaultdict[str, int] = defaultdict(lambda: 0)
        self.summary: dict[str, Any] = {}

    def assign_type(self, key: str, val: Any) -> None:
        pass

    def update(self, metrics: dict) -> None:
        metrics = flatten_dict(metrics)
        for key_dots, val in metrics.items():
            key = key_dots.replace(".", "/")
            if self.summary.get(key, None) is None:
                self.assign_type(key, val)
            self.steps[key] += 1


def apply_model(model: Any, inputs: Any, ignore_dim: Any = None) -> None:
    """Placeholder helper for generic model application across input structures.

    Currently not implemented; kept as an extension hook for future utility code.
    """
    pass


def plot_metrics(metrics: dict, path: str, prefix: str) -> None:
    """Render and save line plots for each metric series in a dictionary."""
    os.makedirs(path, exist_ok=True)
    for key, val in metrics.items():
        lineplot(np.arange(len(val)), val, f"{prefix}{key}", path)


def lineplot(
    xs: np.ndarray | list,
    ys: Any,
    title: str,
    path: str = "",
    xaxis: str = "episode",
) -> None:
    """Create a Plotly line plot for scalar, dict, or ensemble-series data.

    Supports uncertainty-band plotting when `ys` is a 2D array.
    """
    MAX_LINE = Line(color="rgb(0, 132, 180)", dash="dash")
    MIN_LINE = Line(color="rgb(0, 132, 180)", dash="dash")
    NO_LINE = Line(color="rgba(0, 0, 0, 0)")
    MEAN_LINE = Line(color="rgb(0, 172, 237)")
    std_colour = "rgba(29, 202, 255, 0.2)"
    if isinstance(ys, dict):
        data = []
        for key, val in ys.items():
            xs = np.arange(len(val))
            data.append(Scatter(x=xs, y=np.array(val), name=key))
    elif np.asarray(ys, dtype=np.float32).ndim == 2:
        ys = np.asarray(ys, dtype=np.float32)
        ys_mean, ys_std = ys.mean(-1), ys.std(-1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std
        l_max = Scatter(x=xs, y=ys.max(-1), line=MAX_LINE, name="Max")
        l_min = Scatter(x=xs, y=ys.min(-1), line=MIN_LINE, name="Min")
        l_stu = Scatter(x=xs, y=ys_upper, line=NO_LINE, showlegend=False)
        l_mean = Scatter(
            x=xs,
            y=ys_mean,
            fill="tonexty",
            fillcolor=std_colour,
            line=MEAN_LINE,
            name="Mean",
        )
        l_stl = Scatter(
            x=xs,
            y=ys_lower,
            fill="tonexty",
            fillcolor=std_colour,
            line=NO_LINE,
            name="-1 Std. Dev.",
            showlegend=False,
        )
        data = [l_stu, l_mean, l_stl, l_min, l_max]
    else:
        data = [Scatter(x=xs, y=ys, line=MEAN_LINE)]
    plotly.offline.plot(
        {
            "data": data,
            "layout": dict(title=title, xaxis={"title": xaxis}, yaxis={"title": title}),
        },
        filename=os.path.join(path, title + ".html"),
        auto_open=False,
    )


class TorchImageEnvWrapper:
    """
    Torch Env Wrapper that wraps a gym env and makes interactions using Tensors.
    Also returns observations in image form.
    """

    def __init__(
        self, env: Any, bit_depth: int, observation_shape: Any = None, act_rep: int = 2
    ) -> None:
        self.env: Any
        if isinstance(env, str):
            # gymnasium first. The legacy `gym` package is what this used to
            # import unconditionally, and its passive env checker still reads
            # np.bool8, which NumPy 2.0 removed -- so PlaNet and RSSM died with
            # "module 'numpy' has no attribute 'bool8'" on their first env step.
            # step() below already accepts both the 4- and 5-tuple returns.
            try:
                import gymnasium as gym
            except ImportError:
                import gym  # type: ignore[no-redef]

            try:
                self.env = gym.make(env, render_mode="rgb_array")
                self._render_mode_supported = True
            except ImportError as exc:
                from simulusion.envs.robotics_env import (
                    is_moved_mujoco_error,
                    make_robotics_env,
                )

                if not is_moved_mujoco_error(exc):
                    raise
                self.env = make_robotics_env(env, render_mode="rgb_array")
                self._render_mode_supported = True
            except TypeError:
                self.env = gym.make(env)
                self._render_mode_supported = False
        else:
            self.env = env
            self._render_mode_supported = True
        self.bit_depth = bit_depth
        self.action_repeats = act_rep

    def _get_frame(self, last_obs: Any = None) -> np.ndarray | None:
        frame = None
        try:
            out: Any = self.env.render()
        except Exception:
            out = None
        # gym may return (frame, info)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, np.ndarray):
            frame = out
        else:
            # try alternative render signature
            try:
                out = self.env.render(mode="rgb_array")
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, np.ndarray):
                    frame = out
            except Exception:
                frame = None

        # If frame is still not ndarray, and last_obs is image-like, use that
        if frame is None and last_obs is not None:
            frame = self._obs_to_frame(
                last_obs if not isinstance(last_obs, tuple) else last_obs[0]
            )

        # If still none, synthesize a simple visualization from a 1D state vector
        if frame is None and last_obs is not None:
            try:
                obs = last_obs if not isinstance(last_obs, tuple) else last_obs[0]
                arr = np.asarray(obs)
                # if simple vector -> make a 64x64 RGB gradient from values
                if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
                    vals = (
                        (arr.flatten() - arr.min())
                        if arr.max() != arr.min()
                        else arr.flatten()
                    )
                    if vals.size == 0:
                        vals = np.zeros(1)
                    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
                    for i, v in enumerate(vals[:8]):  # encode first few dims as bands
                        band = int(255 * v)
                        canvas[:, i * 8 : (i + 1) * 8, :] = band
                    frame = canvas
            except Exception:
                frame = None

        return frame

    def _obs_to_frame(self, obs: Any) -> np.ndarray | None:
        """Convert common observation formats to an HWC image when possible."""
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, dict):
            for key in ("image", "pixels", "rgb", "observation"):
                if key in obs:
                    frame = self._obs_to_frame(obs[key])
                    if frame is not None:
                        return frame
            for value in obs.values():
                frame = self._obs_to_frame(value)
                if frame is not None:
                    return frame
            return None

        if not isinstance(obs, np.ndarray):
            try:
                obs = np.asarray(obs)
            except Exception:
                return None

        if obs.ndim == 3:
            frame = obs
            if frame.shape[-1] not in (1, 3, 4) and frame.shape[0] in (1, 3, 4):
                frame = frame.transpose(1, 2, 0)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            return frame

        if obs.ndim == 2:
            return np.repeat(obs[..., None], 3, axis=-1)

        if obs.ndim == 1:
            vals = obs.astype(np.float32).reshape(-1)
            if vals.size == 0:
                return None
            if vals.max() != vals.min():
                vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            else:
                vals = np.zeros_like(vals)
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            for i, v in enumerate(vals[:8]):
                canvas[:, i * 8 : (i + 1) * 8, :] = int(255 * float(v))
            return canvas

        return None

    def reset(self) -> torch.Tensor:
        ret = self.env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        frame = self._obs_to_frame(obs)
        if frame is None:
            frame = self._get_frame(last_obs=obs)
        if frame is None:
            raise RuntimeError(
                "Environment did not provide an RGB frame on reset. "
                "Use an env that supports image rendering or instantiate with render_mode='rgb_array'."
            )
        x = to_tensor_obs(frame)
        preprocess_img(x, self.bit_depth)
        return x

    def step(self, u: torch.Tensor | np.ndarray | list | float | int) -> tuple:
        if isinstance(u, torch.Tensor):
            u_t: Any = u.cpu().detach()
        else:
            u_t = u
        if getattr(self.env.action_space, "n", None) is not None:
            n = int(getattr(self.env.action_space, "n", 1))
            arr = u_t.numpy() if isinstance(u_t, torch.Tensor) else np.asarray(u_t)
            arr = arr.reshape(-1)
            if arr.size > 1:
                action: Any = int(np.argmax(arr))
            else:
                val = float(arr[0]) if arr.size else 0.0
                action = int(np.clip(int(round(val)), 0, n - 1))
        else:
            action = u_t.numpy() if isinstance(u_t, torch.Tensor) else np.asarray(u_t)

        rwds: float = 0.0
        last_d = False
        last_i = {}
        last_obs = None
        for _ in range(self.action_repeats):
            ret = self.env.step(action)
            if len(ret) == 4:
                obs, r, d, i = ret
            else:
                obs, r, term, trunc, i = ret
                d = term or trunc
            rwds += r
            last_d = d
            last_i = i
            last_obs = obs
        frame = self._obs_to_frame(
            last_obs if not isinstance(last_obs, tuple) else last_obs[0]
        )
        if frame is None:
            frame = self._get_frame(last_obs=last_obs)
        if frame is None:
            raise RuntimeError(
                "Environment did not provide an RGB frame on step. "
                "Use an env that supports image rendering or instantiate with render_mode='rgb_array'."
            )
        x = to_tensor_obs(frame)
        preprocess_img(x, self.bit_depth)
        return x, rwds, last_d, last_i

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def observation_size(self) -> tuple[int, int, int]:
        return (3, 64, 64)

    @property
    def action_size(self) -> int:
        space = getattr(self.env, "action_space", None)
        if space is None:
            return 1

        shp = getattr(space, "shape", None)
        if shp and len(shp) > 0:
            try:
                prod = int(np.prod(shp))
                return prod if prod > 0 else 1
            except Exception:
                pass

        if getattr(space, "n", None) is not None:
            return 1

        try:
            sample = space.sample()
            arr = np.asarray(sample)
            if arr.ndim == 0:
                return 1
            return int(arr.size)
        except Exception:
            return 1

    def sample_random_action(self) -> torch.Tensor:
        return torch.tensor(self.env.action_space.sample())

    @property
    def max_episode_steps(self) -> int:
        """Return environment max episode steps (compatible with TimeLimit/spec)."""
        if (
            hasattr(self.env, "_max_episode_steps")
            and self.env._max_episode_steps is not None
        ):
            return self.env._max_episode_steps
        if (
            getattr(self.env, "spec", None) is not None
            and getattr(self.env.spec, "max_episode_steps", None) is not None
        ):
            steps = self.env.spec.max_episode_steps
            assert steps is not None
            return int(steps)
        return 1000


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather token subsets from patch sequences using index masks.

    Each mask selects token positions from `x`; selected groups are concatenated
    along the batch dimension.
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        all_x.append(torch.gather(x, 1, mask_keep))
    return torch.cat(all_x, dim=0)


def visualize_latent_tsne(
    latents: torch.Tensor | np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
    perplexity: int = 30,
) -> "plotly.graph_objs.Figure":
    """
    Visualize latent representations using t-SNE.

    Args:
        latents: torch.Tensor of shape (N, D) or numpy array
        labels: optional list or array of labels for coloring
        save_path: path to save the plot (HTML for plotly)
        perplexity: t-SNE perplexity parameter
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    fig = plotly.graph_objs.Figure()
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            fig.add_trace(
                Scatter(
                    x=latents_2d[idx, 0],
                    y=latents_2d[idx, 1],
                    mode="markers",
                    name=str(label),
                    marker=dict(size=5),
                )
            )
    else:
        fig.add_trace(
            Scatter(
                x=latents_2d[:, 0],
                y=latents_2d[:, 1],
                mode="markers",
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title="Latent Space t-SNE", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2"
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def visualize_latent_umap(
    latents: torch.Tensor | np.ndarray,
    labels: np.ndarray | None = None,
    save_path: str | None = None,
    n_neighbors: int = 15,
) -> "plotly.graph_objs.Figure":
    """
    Visualize latent representations using UMAP.

    Args:
        latents: torch.Tensor of shape (N, D) or numpy array
        labels: optional list or array of labels for coloring
        save_path: path to save the plot (HTML for plotly)
        n_neighbors: UMAP n_neighbors parameter
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()

    reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
    latents_2d = reducer.fit_transform(latents)

    fig = plotly.graph_objs.Figure()
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            fig.add_trace(
                Scatter(
                    x=latents_2d[idx, 0],
                    y=latents_2d[idx, 1],
                    mode="markers",
                    name=str(label),
                    marker=dict(size=5),
                )
            )
    else:
        fig.add_trace(
            Scatter(
                x=latents_2d[:, 0],
                y=latents_2d[:, 1],
                mode="markers",
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title="Latent Space UMAP", xaxis_title="UMAP 1", yaxis_title="UMAP 2"
    )

    if save_path:
        fig.write_html(save_path)
    return fig


class StreamingVideoWriter:
    """
    A class for streaming video writing to save frames in real-time.

    Args:
        path: output video file path
        fps: frames per second
        frame_shape: (width, height) of frames; inferred from the first frame if None
        format: 'mp4' or 'avi'
    """

    def __init__(
        self, path: str, fps: int = 20, frame_shape: Any = None, format: str = "mp4"
    ) -> None:
        self.path = path
        self.fps = fps
        self.frame_shape = frame_shape
        self.format = format.lower()
        self.writer: Any = None

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.

        Args:
            frame: numpy array of shape (H, W, C) or (H, W), uint8 or float in [0,1]
        """
        import cv2

        if self.writer is None:
            if self.frame_shape is None:
                self.frame_shape = frame.shape[:2][::-1]  # (W, H)
            if self.format == "mp4":
                self.writer = Mp4Writer(self.path, self.fps, self.frame_shape)
            elif self.format == "avi":
                self.writer = cv2.VideoWriter(
                    self.path,
                    cv2.VideoWriter.fourcc(*"XVID"),
                    self.fps,
                    self.frame_shape,
                )
            else:
                raise ValueError("Unsupported format")

        # Convert frame to uint8 HWC BGR
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 3:  # RGB to BGR
            frame = frame[..., ::-1]
        assert self.writer is not None
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
