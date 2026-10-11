"""I-JEPA: self-supervised image representations learned by predicting in latent space.

    python demos/jepa_demo.py train  --epochs 30
    python demos/jepa_demo.py record --run demos/runs/jepa

I-JEPA never reconstructs pixels and never sees a label: it learns by
predicting the *representations* of masked image blocks from a visible
context block. The demo trains a ViT-Tiny on CIFAR-10 and then probes what the
encoder learned, comparing against the same network at random initialisation.

``record`` writes into ``<run>/media``:

* ``neighbours.png`` -- test images and their nearest training images in the
  encoder's feature space (no labels involved).
* ``patches.png`` -- the top three principal components of the patch features,
  shown as colour: patches the encoder considers similar get similar colours.
* ``knn.json`` -- k-nearest-neighbour classification accuracy, trained vs random.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]
from _media import grid, hstack, label, upscale, vstack, write_png  # noqa: E402

MEAN = (0.485, 0.456, 0.406)  # the normalisation train_jepa applies
STD = (0.229, 0.224, 0.225)


def train(args: argparse.Namespace) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "simulusion.training.train_jepa",
        "data.dataset=cifar10",
        f"data.root_path={args.data}",
        "data.download=false",
        "data.crop_size=32",
        f"data.batch_size={args.batch_size}",
        # Spawned DataLoader workers on Windows stalled at start-up here
        # (each re-imports torch and pickles the mask collator).
        f"data.num_workers={args.workers}",
        "mask.patch_size=4",
        "mask.min_keep=4",
        "meta.model_name=vit_tiny",
        "meta.use_bfloat16=true",
        f"optimization.epochs={args.epochs}",
        "optimization.warmup=2",
        # Without this the learning rate is scaled by batch/2048 (to ~3e-5).
        "optimization.lr_reference_batch_size=null",
        "optimization.lr=1e-3",
        "optimization.start_lr=2e-4",
        f"logging.folder={run.as_posix()}",
        "logging.write_tag=jepa",
    ]
    print(" ".join(cmd), flush=True)
    with (run / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=_HERE.parent
        )


def record(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    from simulusion.helpers.jepa_helper import init_model

    run = Path(args.run)
    out = run / "media"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encoder(trained: bool):
        torch.manual_seed(0)
        enc, _ = init_model(device, patch_size=4, model_name="vit_tiny", crop_size=32)
        if trained:
            ckpt = torch.load(
                run / "jepa-latest.pth.tar", map_location=device, weights_only=True
            )
            # The EMA target encoder is the one the paper evaluates.
            enc.load_state_dict(
                {
                    k.removeprefix("module."): v
                    for k, v in ckpt["target_encoder"].items()
                }
            )
        return enc.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_set = datasets.CIFAR10(args.data, train=True, download=False, transform=tf)
    test_set = datasets.CIFAR10(args.data, train=False, download=False, transform=tf)

    def features(enc, dataset, limit: int):
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, range(limit)), batch_size=500
        )
        pooled, patches, labels = [], [], []
        with torch.no_grad():
            for x, y in loader:
                tokens = enc(x.to(device))  # (B, N, D)
                pooled.append(F.normalize(tokens.mean(1), dim=-1).cpu())
                if not patches:  # patch tokens of the first batch, for the PCA view
                    patches.append(tokens.cpu())
                labels.append(y)
        return torch.cat(pooled), patches[0], torch.cat(labels)

    def knn_accuracy(bank, bank_y, query, query_y, k: int = 20) -> float:
        sims = query @ bank.T
        top = sims.topk(k, dim=1).indices
        votes = torch.zeros(len(query), 10)
        votes.scatter_add_(1, bank_y[top], torch.ones_like(top, dtype=torch.float))
        return float((votes.argmax(1) == query_y).float().mean())

    results = {}
    cache = {}
    for name, trained in (("random init", False), ("I-JEPA", True)):
        enc = encoder(trained)
        bank, _, bank_y = features(enc, train_set, args.bank)
        query, query_patches, query_y = features(enc, test_set, args.queries)
        results[name] = knn_accuracy(bank, bank_y, query, query_y)
        cache[name] = (bank, query, query_patches)
        print(f"kNN@20 accuracy ({name}): {results[name]:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "knn.json").write_text(json.dumps(results, indent=2))

    raw_train = datasets.CIFAR10(args.data, train=True, download=False)
    raw_test = datasets.CIFAR10(args.data, train=False, download=False)
    bank, query, query_patches = cache["I-JEPA"]
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(args.queries, size=args.rows, replace=False)
    rows = []
    for q in picks:
        nn_idx = (query[q] @ bank.T).topk(args.neighbours).indices.tolist()
        tiles = [
            label(upscale(np.asarray(raw_test[int(q)][0]), 3), "query", accent=True)
        ]
        tiles += [label(upscale(np.asarray(raw_train[i][0]), 3), "") for i in nn_idx]
        rows.append(hstack(tiles, gap=3))
    print("wrote", write_png(vstack(rows, gap=3), out / "neighbours.png"))

    # PCA of patch tokens -> RGB, fitted over the first batch of test images.
    tokens = query_patches  # (B, 64, D)
    flat = tokens.reshape(-1, tokens.shape[-1])
    flat = flat - flat.mean(0)
    _, _, v = torch.pca_lowrank(flat, q=3)
    proj = (flat @ v).reshape(tokens.shape[0], 8, 8, 3)
    lo, hi = proj.amin((0, 1, 2)), proj.amax((0, 1, 2))
    proj = ((proj - lo) / (hi - lo)).numpy()
    tiles = []
    for i in range(args.pca_images):
        tiles.append(
            hstack(
                [upscale(np.asarray(raw_test[i][0]), 4), upscale(proj[i], 16)],
                gap=2,
            )
        )
    print("wrote", write_png(grid(tiles, cols=4, gap=6), out / "patches.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--data", default="data")
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--run", default="demos/runs/jepa")

    r = sub.add_parser("record")
    r.add_argument("--run", default="demos/runs/jepa")
    r.add_argument("--data", default="data")
    r.add_argument("--bank", type=int, default=50000)
    r.add_argument("--queries", type=int, default=10000)
    r.add_argument("--rows", type=int, default=6)
    r.add_argument("--neighbours", type=int, default=7)
    r.add_argument("--pca-images", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    {"train": train, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
