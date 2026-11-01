"""Frame and video helpers shared by the demo scripts.

Every demo produces the same three kinds of artifact -- labelled side-by-side
clips, image grids and a learning curve -- so they are written here once.
Frames are HWC float arrays in [0, 1] or uint8 arrays; everything is converted
to uint8 RGB on the way out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PANEL_BG = (17, 24, 39)  # slate-900, matches the docs' dark theme
LABEL_FG = (241, 245, 249)
ACCENT = (96, 165, 250)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """HWC/CHW, float [0, 1] or uint8 -> HWC uint8 RGB."""
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = frame.transpose(1, 2, 0)
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])


def upscale(frame: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upscale, so 64x64 pixels stay crisp."""
    frame = to_uint8(frame)
    if factor <= 1:
        return frame
    return frame.repeat(factor, axis=0).repeat(factor, axis=1)


def label(frame: np.ndarray, text: str, accent: bool = False) -> np.ndarray:
    """Add a caption bar above a frame."""
    import cv2

    frame = to_uint8(frame)
    height = max(18, frame.shape[0] // 12)
    bar = np.empty((height, frame.shape[1], 3), dtype=np.uint8)
    bar[:] = PANEL_BG
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1 if height < 30 else 2
    scale = height / 30.0
    # Shrink long captions until they fit the panel width.
    width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    if width > frame.shape[1] - 12:
        scale *= (frame.shape[1] - 12) / width
    cv2.putText(
        bar,
        text,
        (6, int(height * 0.72)),
        font,
        scale,
        ACCENT if accent else LABEL_FG,
        thickness,
        cv2.LINE_AA,
    )
    return np.concatenate([bar, frame], axis=0)


def hstack(frames: Sequence[np.ndarray], gap: int = 4) -> np.ndarray:
    """Place frames side by side with a thin separator."""
    frames = [to_uint8(f) for f in frames]
    height = max(f.shape[0] for f in frames)
    parts: list[np.ndarray] = []
    for i, f in enumerate(frames):
        if f.shape[0] < height:
            pad = np.empty((height - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            pad[:] = PANEL_BG
            f = np.concatenate([f, pad], axis=0)
        if i:
            sep = np.empty((height, gap, 3), dtype=np.uint8)
            sep[:] = PANEL_BG
            parts.append(sep)
        parts.append(f)
    return np.concatenate(parts, axis=1)


def vstack(frames: Sequence[np.ndarray], gap: int = 4) -> np.ndarray:
    frames = [to_uint8(f) for f in frames]
    width = max(f.shape[1] for f in frames)
    parts: list[np.ndarray] = []
    for i, f in enumerate(frames):
        if f.shape[1] < width:
            pad = np.empty((f.shape[0], width - f.shape[1], 3), dtype=np.uint8)
            pad[:] = PANEL_BG
            f = np.concatenate([f, pad], axis=1)
        if i:
            sep = np.empty((gap, width, 3), dtype=np.uint8)
            sep[:] = PANEL_BG
            parts.append(sep)
        parts.append(f)
    return np.concatenate(parts, axis=0)


def grid(images: Sequence[np.ndarray], cols: int, gap: int = 2) -> np.ndarray:
    """Tile equally sized images into rows of ``cols``."""
    rows = [hstack(images[i : i + cols], gap=gap) for i in range(0, len(images), cols)]
    return vstack(rows, gap=gap)


def _even(frame: np.ndarray) -> np.ndarray:
    """H.264 with yuv420p needs even dimensions."""
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def write_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 15) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-crf", "23", "-preset", "slow", "-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(_even(to_uint8(frame)))
    finally:
        writer.close()
    return path


def write_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 15,
    max_width: int | None = 480,
) -> Path:
    """Palette-optimised looping GIF, for places that cannot play MP4 (README)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        img = Image.fromarray(to_uint8(frame))
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            img = img.resize(
                (max_width, max(2, round(img.height * ratio))), Image.Resampling.NEAREST
            )
        images.append(img)
    # One shared palette keeps the file small and stops colours flickering.
    palette = images[len(images) // 2].quantize(
        colors=128, method=Image.Quantize.MEDIANCUT
    )
    quantized = [
        im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images
    ]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=max(20, round(1000 / fps)),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


def write_png(image: np.ndarray, path: str | Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path, optimize=True)
    return path


def read_metrics(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def plot_curves(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """One small line chart in the docs' palette, readable on light and dark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for (name, (xs, ys)), color in zip(series.items(), colors):
        ax.plot(xs, ys, label=name, color=color, linewidth=2)
    ax.set_title(title, color="#64748b", fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color="#64748b")
    ax.set_ylabel(ylabel, color="#64748b")
    ax.tick_params(colors="#64748b")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#94a3b8")
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(frameon=False, labelcolor="#64748b")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path
