"""Copy the recorded demo media into the docs, compressed for the web.

    python demos/build_gallery.py

Reads ``demos/runs/<demo>/media`` and writes ``docs/source/_static/gallery``:
MP4s are re-encoded small (H.264, CRF 28, no audio), PNGs are copied, and a
short looping GIF is made from each headline clip for the README, which
GitHub cannot play MP4s in. Missing demos are skipped with a note.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import write_gif  # noqa: E402

ROOT = _HERE.parent
RUNS = _HERE / "runs"
GALLERY = ROOT / "docs" / "source" / "_static" / "gallery"

# (run dir, media file, gallery name). The first clip of each model also gets
# a README GIF.
MEDIA = [
    ("dreamer", "dream.mp4", "dreamer_dream"),
    ("dreamer", "policy.mp4", "dreamer_policy"),
    ("dreamer", "curve.png", "dreamer_curve"),
    ("dreamer_v2", "dream.mp4", "dreamer_v2_dream"),
    ("dreamer_v2", "policy.mp4", "dreamer_v2_policy"),
    ("dreamer_v2", "curve.png", "dreamer_v2_curve"),
    ("planet", "dream.mp4", "planet_dream"),
    ("planet", "policy.mp4", "planet_policy"),
    ("planet", "curve.png", "planet_curve"),
    ("modular_rssm", "dream.mp4", "modular_rssm_dream"),
    ("modular_rssm", "curve.png", "modular_rssm_curve"),
    ("diamond", "dream.mp4", "diamond_dream"),
    ("diamond", "play_in_dream.mp4", "diamond_play_in_dream"),
    ("diamond", "curve.png", "diamond_curve"),
    ("iris_boxing", "tokens.mp4", "iris_tokens"),
    ("iris_boxing", "dream.mp4", "iris_dream"),
    ("iris_boxing", "policy.mp4", "iris_policy"),
    ("iris_boxing", "curve.png", "iris_curve"),
    ("genie", "replay.mp4", "genie_replay"),
    ("genie", "tokens.mp4", "genie_tokens"),
    ("genie", "curve.png", "genie_curve"),
    ("dit", "denoising.mp4", "dit_denoising"),
    ("dit", "samples.png", "dit_samples"),
    ("jepa", "neighbours.png", "jepa_neighbours"),
    ("jepa", "knn.json", "jepa_knn"),
]
README_GIFS = {
    "dreamer_dream",
    "planet_dream",
    "diamond_dream",
    "iris_dream",
    "genie_replay",
}


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def compress_mp4(src: Path, dst: Path, max_width: int = 720) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale='min({max_width},iw)':-2:flags=neighbor",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ],
        check=True,
    )


def gif_from_mp4(
    src: Path, dst: Path, seconds: float = 6.0, max_width: int = 420
) -> None:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(src))
    fps = reader.get_meta_data().get("fps", 10)
    frames = []
    for i, frame in enumerate(reader.iter_data()):
        if i >= seconds * fps:
            break
        frames.append(np.asarray(frame))
    reader.close()
    write_gif(frames, dst, fps=int(fps), max_width=max_width)


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    missing = []
    for run, filename, name in MEDIA:
        src = RUNS / run / "media" / filename
        if not src.exists():
            missing.append(str(src.relative_to(ROOT)))
            continue
        dst = GALLERY / f"{name}{src.suffix}"
        if src.suffix == ".mp4":
            compress_mp4(src, dst)
            if name in README_GIFS:
                gif_from_mp4(src, GALLERY / f"{name}.gif")
        else:
            shutil.copyfile(src, dst)
        print(f"{dst.relative_to(ROOT)}  {dst.stat().st_size / 1024:.0f} KB")
    total = sum(p.stat().st_size for p in GALLERY.iterdir()) / 1024 / 1024
    print(f"gallery total: {total:.1f} MB")
    if missing:
        print("not recorded yet:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
