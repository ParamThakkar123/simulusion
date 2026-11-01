"""Frozen-encoder linear evaluation for I-JEPA checkpoints.

Implements the protocol of Assran et al., "Self-Supervised Learning from Images
with a Joint-Embedding Predictive Architecture" (CVPR 2023), Appendix A.2:

* the **target-encoder** is frozen and its patch tokens are average-pooled to
  form a global image representation (I-JEPA trains no ``[cls]`` token);
* the reported number is the better of the average-pooled last layer and the
  concatenation of the average-pooled last four layers;
* a linear head is trained on those features with LARS, a batch size of 16384
  and 50 epochs, decaying the learning rate by 10x every 15 epochs, sweeping
  reference learning rates ``[0.01, 0.05, 0.001]`` and weight decays
  ``[0.0005, 0.0]`` and keeping the best.

Usage::

    python -m simulusion.training.eval_jepa \\
        --checkpoint results/jepa/jepa_run-latest.pth.tar \\
        --root-path /data/imagenet

    from simulusion.training.eval_jepa import jepa_linear_probe
    results = jepa_linear_probe(checkpoint="...", root_path="/data/imagenet")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Iterable, Sequence

import torch
from simulusion.utils.device import default_device_name
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder

from simulusion.helpers.jepa_helper import init_model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Appendix A.2: the ImageNet linear probe sweeps these and reports the best.
REFERENCE_LRS = (0.01, 0.05, 0.001)
WEIGHT_DECAYS = (0.0005, 0.0)


class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (You et al., 2017).

    The optimizer used for the linear probe in Appendix A.2, following MAE.
    Biases and 1-D parameters are excluded from both adaptation and weight
    decay by passing them in a group with ``lars_exclude=True``.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if not group.get("lars_exclude", False):
                    grad = grad.add(p, alpha=group["weight_decay"])
                    param_norm = torch.norm(p)
                    grad_norm = torch.norm(grad)
                    # Only adapt where both norms are non-zero, as in the paper.
                    trust = torch.where(
                        param_norm > 0,
                        torch.where(
                            grad_norm > 0,
                            group["eta"] * param_norm / grad_norm,
                            torch.ones_like(param_norm),
                        ),
                        torch.ones_like(param_norm),
                    )
                    grad = grad.mul(trust)

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(group["momentum"]).add_(grad)
                p.add_(mu, alpha=-group["lr"])

        return loss


def load_jepa_encoder(
    checkpoint: str,
    device: torch.device,
    model_name: str = "vit_base",
    patch_size: int = 16,
    crop_size: int = 224,
    weights: str = "target_encoder",
) -> nn.Module:
    """Load a frozen I-JEPA encoder from a training checkpoint.

    ``weights`` selects which set of encoder weights to evaluate; the paper
    uses the EMA ``target_encoder`` ("We use the target-encoder for evaluation
    and average pool its output").
    """
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
        pred_depth=None,
        pred_emb_dim=384,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights not in state:
        raise KeyError(
            f"checkpoint {checkpoint!r} has no {weights!r} weights "
            f"(found: {sorted(k for k in state if isinstance(state[k], dict))})"
        )
    # Checkpoints written under DistributedDataParallel carry a "module." prefix.
    encoder_state = {
        key.replace("module.", "", 1): value for key, value in state[weights].items()
    }
    msg = encoder.load_state_dict(encoder_state)
    logger.info(f"loaded {weights} from {checkpoint} with msg: {msg}")

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def make_eval_transforms(crop_size: int = 224, training: bool = False) -> Any:
    """Build the VISSL-style probe transforms used in Appendix A.2.

    Random resized crop plus horizontal flip while training the head, and a
    resize/center-crop at evaluation time. These augment the *probe*, not the
    pretraining, so they do not affect the paper's no-augmentation claim.
    """
    if training:
        pipeline = [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        pipeline = [
            transforms.Resize(int(crop_size * 256 / 224)),
            transforms.CenterCrop(crop_size),
        ]
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_NORMALIZATION),
    ]
    return transforms.Compose(pipeline)


def _make_dataset(
    dataset: str,
    root_path: str,
    training: bool,
    crop_size: int,
    train_folder: str,
    val_folder: str,
    download: bool,
) -> torch.utils.data.Dataset:
    transform = make_eval_transforms(crop_size=crop_size, training=training)
    if dataset.lower() == "cifar10":
        return CIFAR10(
            root=root_path, train=training, download=download, transform=transform
        )
    folder = train_folder if training else val_folder
    return ImageFolder(root=f"{root_path.rstrip('/')}/{folder}", transform=transform)


@torch.no_grad()
def extract_features(
    encoder: Any,
    loader: Iterable[Any],
    device: torch.device,
    last_n_blocks: int = 1,
    use_bfloat16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average-pool frozen encoder tokens into one feature vector per image.

    With ``last_n_blocks > 1`` the average-pooled outputs of the last ``n``
    transformer blocks are concatenated, which is the second representation the
    paper's protocol considers.
    """
    features, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            if last_n_blocks > 1:
                tokens = encoder.get_intermediate_layers(images, n=last_n_blocks)
                pooled = torch.cat([t.mean(dim=1) for t in tokens], dim=-1)
            else:
                pooled = encoder(images).mean(dim=1)
        features.append(pooled.float().cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def train_linear_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    reference_lr: float = 0.01,
    weight_decay: float = 0.0005,
    epochs: int = 50,
    batch_size: int = 16384,
    lr_decay_every: int = 15,
    lr_decay_factor: float = 10.0,
    batch_norm: bool = False,
) -> tuple[nn.Module, float]:
    """Train one linear head on frozen features and return it with its top-1.

    Follows Appendix A.2: LARS, batch size 16384, 50 epochs, and a step-wise
    decay dividing the learning rate by 10 every 15 epochs. ``batch_norm``
    adds the batch-normalized variant of the head that the protocol also tries.
    """
    layers: list[nn.Module] = []
    if batch_norm:
        layers.append(nn.BatchNorm1d(train_features.shape[1], affine=False))
    layers.append(nn.Linear(train_features.shape[1], num_classes))
    head = nn.Sequential(*layers).to(device)

    weights = [p for _, p in head.named_parameters() if p.ndim > 1]
    biases = [p for _, p in head.named_parameters() if p.ndim <= 1]
    optimizer = LARS(
        [
            {"params": weights, "weight_decay": weight_decay},
            {"params": biases, "weight_decay": 0.0, "lars_exclude": True},
        ],
        lr=reference_lr,
        momentum=0.9,
    )

    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False
    )

    for epoch in range(epochs):
        lr = reference_lr / (lr_decay_factor ** (epoch // lr_decay_every))
        for group in optimizer.param_groups:
            group["lr"] = lr
        head.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            loss = F.cross_entropy(head(batch_features), batch_labels)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(val_features.to(device))
        top1 = (logits.argmax(dim=-1).cpu() == val_labels).float().mean().item() * 100
    return head, top1


def jepa_linear_probe(
    checkpoint: str,
    root_path: str,
    dataset: str = "imagenet",
    model_name: str = "vit_base",
    patch_size: int = 16,
    crop_size: int = 224,
    weights: str = "target_encoder",
    train_folder: str = "train",
    val_folder: str = "val",
    download: bool = False,
    batch_size: int = 256,
    num_workers: int = 8,
    epochs: int = 50,
    head_batch_size: int = 16384,
    reference_lrs: Sequence[float] = REFERENCE_LRS,
    weight_decays: Sequence[float] = WEIGHT_DECAYS,
    representations: Sequence[int] = (1, 4),
    device: str | None = None,
    use_bfloat16: bool = False,
) -> dict[str, Any]:
    """Run the paper's linear evaluation and return every swept result.

    Returns a dict with the best top-1 accuracy under ``"top1"`` and the full
    sweep under ``"sweep"``. ``representations`` lists how many trailing blocks
    to average-pool and concatenate -- the paper tries 1 and 4.
    """
    torch_device = torch.device(device or default_device_name())
    encoder = load_jepa_encoder(
        checkpoint=checkpoint,
        device=torch_device,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=crop_size,
        weights=weights,
    )

    loaders = {}
    for split, training in (("train", True), ("val", False)):
        data = _make_dataset(
            dataset=dataset,
            root_path=root_path,
            training=training,
            crop_size=crop_size,
            train_folder=train_folder,
            val_folder=val_folder,
            download=download,
        )
        loaders[split] = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    num_classes = len(getattr(loaders["train"].dataset, "classes", []))
    if num_classes == 0:
        raise ValueError(f"could not infer class count for dataset {dataset!r}")

    sweep = []
    best = {"top1": -1.0}
    for last_n_blocks in representations:
        logger.info(f"extracting features from the last {last_n_blocks} block(s)")
        train_features, train_labels = extract_features(
            encoder, loaders["train"], torch_device, last_n_blocks, use_bfloat16
        )
        val_features, val_labels = extract_features(
            encoder, loaders["val"], torch_device, last_n_blocks, use_bfloat16
        )
        for batch_norm in (False, True):
            for reference_lr in reference_lrs:
                for weight_decay in weight_decays:
                    _, top1 = train_linear_head(
                        train_features,
                        train_labels,
                        val_features,
                        val_labels,
                        num_classes=num_classes,
                        device=torch_device,
                        reference_lr=reference_lr,
                        weight_decay=weight_decay,
                        epochs=epochs,
                        batch_size=head_batch_size,
                        batch_norm=batch_norm,
                    )
                    result = {
                        "last_n_blocks": last_n_blocks,
                        "batch_norm": batch_norm,
                        "reference_lr": reference_lr,
                        "weight_decay": weight_decay,
                        "top1": top1,
                    }
                    logger.info(f"linear probe: {result}")
                    sweep.append(result)
                    if top1 > best["top1"]:
                        best = result
    return {**best, "sweep": sweep}


def main_from_cli(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments and run the I-JEPA linear evaluation."""
    parser = argparse.ArgumentParser(description="I-JEPA linear evaluation")
    parser.add_argument("--checkpoint", required=True, help="JEPA training checkpoint")
    parser.add_argument("--root-path", required=True, help="Dataset root directory")
    parser.add_argument("--dataset", default="imagenet", help="imagenet|cifar10|folder")
    parser.add_argument("--model-name", default="vit_base")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--weights", default="target_encoder", help="or 'encoder'")
    parser.add_argument("--train-folder", default="train")
    parser.add_argument("--val-folder", default="val")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head-batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-bfloat16", action="store_true")
    parser.add_argument("--output", default=None, help="Write results JSON here")
    parsed = parser.parse_args(argv)

    results = jepa_linear_probe(
        checkpoint=parsed.checkpoint,
        root_path=parsed.root_path,
        dataset=parsed.dataset,
        model_name=parsed.model_name,
        patch_size=parsed.patch_size,
        crop_size=parsed.crop_size,
        weights=parsed.weights,
        train_folder=parsed.train_folder,
        val_folder=parsed.val_folder,
        download=parsed.download,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        epochs=parsed.epochs,
        head_batch_size=parsed.head_batch_size,
        device=parsed.device,
        use_bfloat16=parsed.use_bfloat16,
    )
    logger.info(f"best linear-probe top-1: {results['top1']:.2f}")
    if parsed.output:
        with open(parsed.output, "w") as handle:
            json.dump(results, handle, indent=2)
    return results


if __name__ == "__main__":
    main_from_cli()
