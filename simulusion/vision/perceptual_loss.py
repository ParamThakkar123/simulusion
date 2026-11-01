"""Perceptual loss for the IRIS discrete autoencoder.

The IRIS training objective (Micheli et al., ICLR 2023, Appendix A.1) is an
equally-weighted combination of an L1 reconstruction loss, a commitment loss and
a perceptual loss::

    L(E, D, E) = ||x - D(z)||_1
               + ||sg(E(x)) - E(z)||_2^2
               + ||sg(E(z)) - E(x)||_2^2
               + L_perceptual(x, D(z))

The perceptual term matters more here than in a typical autoencoder: the policy
is trained *entirely* on reconstructions, so anything the decoder averages away
is invisible to it. An L1-only objective happily erases a two-pixel ball in Pong
because the pixel cost of doing so is negligible.

IRIS inherits this term from VQGAN (Esser et al., 2021), which uses LPIPS
(Zhang et al., CVPR 2018): VGG16 features, unit-normalised per channel, squared
differences weighted by a set of *learned* 1x1 linear layers, averaged
spatially and summed across layers. This module reproduces that structure. The
learned linear weights are a small (~7KB) external artifact; when they are not
available the layers fall back to uniform weights, which reduces to the
"unweighted VGG feature distance" variant. That fallback is a real
approximation, not a silent equivalence -- see :class:`LPIPSPerceptualLoss`.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from simulusion.utils.logging_utils import setup_logging
from simulusion.export import ExportableAgentMixin

_LOGGER = setup_logging("PerceptualLoss")

# Reference LPIPS v0.1 linear weights for the VGG backbone (~7KB). These are the
# calibrated per-channel weights from Zhang et al.; without them the loss
# degrades to an unweighted VGG feature distance.
LPIPS_VGG_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
    "master/lpips/weights/v0.1/vgg.pth"
)
LPIPS_VGG_WEIGHTS_FILENAME = "lpips_vgg_v0.1.pth"


def find_lpips_weights(download: bool = True) -> Optional[Path]:
    """Locate the LPIPS linear weights, downloading them into the hub cache.

    Looks in torch's hub checkpoint directory first so a previously fetched copy
    is reused offline. Returns None if the file is absent and cannot be
    retrieved, in which case the caller falls back to uniform weights.
    """
    try:
        import torch.hub

        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    except Exception:  # noqa: BLE001
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

    path = cache_dir / LPIPS_VGG_WEIGHTS_FILENAME
    if path.is_file():
        return path
    if not download:
        return None

    try:
        import torch.hub

        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(LPIPS_VGG_WEIGHTS_URL, str(path), progress=False)
        return path if path.is_file() else None
    except Exception as exc:  # noqa: BLE001 - offline is a normal condition
        _LOGGER.warning("Could not fetch LPIPS linear weights: %s", exc)
        return None


# Indices of the ReLU ending each VGG16 conv block, and that block's channel
# count. LPIPS compares all five.
_VGG16_BLOCKS: Sequence[tuple[int, int]] = (
    (3, 64),
    (8, 128),
    (15, 256),
    (22, 512),
    (29, 512),
)
_VGG16_BN_BLOCKS: Sequence[tuple[int, int]] = (
    (5, 64),
    (12, 128),
    (22, 256),
    (32, 512),
    (42, 512),
)


class LPIPSPerceptualLoss(ExportableAgentMixin, nn.Module):
    """LPIPS-structured perceptual distance over VGG16 features.

    Pipeline, following Zhang et al. (2018) as used by VQGAN:

    1. Map inputs from [0, 1] to [-1, 1] and apply LPIPS's fixed per-channel
       shift/scale (this is *not* ImageNet mean/std -- LPIPS uses its own
       calibration).
    2. Extract features at the end of each VGG16 conv block.
    3. Unit-normalise each feature vector across channels, so no single block's
       activation magnitude dominates the sum.
    4. Square the difference, weight per channel with a 1x1 linear layer, then
       average spatially and sum over blocks.

    Args:
        num_blocks: How many VGG16 conv blocks to compare (1-5). LPIPS uses 5.
        use_batch_norm: Load ``vgg16_bn`` instead of ``vgg16``.
        linear_weights: Path to LPIPS learned linear weights (the ``lin*.model``
            /``vgg.pth`` state dict from the reference implementation). When
            None or unloadable, uniform weights are used and
            :attr:`has_learned_weights` stays False.

    Attributes:
        has_learned_weights: Whether the calibrated LPIPS linear weights were
            loaded. False means this is the unweighted VGG-feature variant.
    """

    # LPIPS's own input calibration, applied after mapping [0,1] -> [-1,1].
    _SHIFT = (-0.030, -0.088, -0.188)
    _SCALE = (0.458, 0.448, 0.450)

    def __init__(
        self,
        num_blocks: int = 5,
        use_batch_norm: bool = False,
        linear_weights: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        from torchvision import models  # imported lazily; optional dependency

        if use_batch_norm:
            vgg = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.IMAGENET1K_V1
            ).features
            block_spec = _VGG16_BN_BLOCKS
        else:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            block_spec = _VGG16_BLOCKS

        num_blocks = max(1, min(num_blocks, len(block_spec)))
        block_spec = block_spec[:num_blocks]

        layer_list = list(vgg.children())
        blocks: List[nn.Module] = []
        start = 0
        for end, _channels in block_spec:
            blocks.append(nn.Sequential(*layer_list[start : end + 1]))
            start = end + 1
        self.blocks = nn.ModuleList(blocks)

        # Per-channel linear weights, one 1x1 convolution per block. Initialised
        # uniform so that, absent the learned weights, each block contributes its
        # mean squared normalised-feature difference.
        self.lins = nn.ModuleList(
            [
                nn.Conv2d(channels, 1, kernel_size=1, bias=False)
                for _, channels in block_spec
            ]
        )
        for lin, (_, channels) in zip(self.lins, block_spec):
            assert isinstance(lin, nn.Conv2d)
            nn.init.constant_(lin.weight, 1.0 / channels)

        self.has_learned_weights = False
        if linear_weights is not None:
            if use_batch_norm:
                # The reference weights are calibrated against plain vgg16's
                # feature statistics. vgg16_bn's normalised activations differ,
                # so applying them there would be worse than uniform.
                _LOGGER.warning(
                    "Ignoring LPIPS linear weights: they are calibrated for "
                    "vgg16, not vgg16_bn."
                )
            else:
                self.has_learned_weights = self._load_linear_weights(linear_weights)

        self.register_buffer("shift", torch.tensor(self._SHIFT).view(1, 3, 1, 1))
        self.shift: torch.Tensor = self.shift
        self.register_buffer("scale", torch.tensor(self._SCALE).view(1, 3, 1, 1))
        self.scale: torch.Tensor = self.scale

        self.eval()
        self.requires_grad_(False)

    def _load_linear_weights(self, path: Union[str, Path]) -> bool:
        """Load LPIPS's calibrated linear weights; return whether it succeeded."""
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            loaded = 0
            for i, lin in enumerate(self.lins):
                assert isinstance(lin, nn.Conv2d)
                # Reference checkpoints key these as "lin{i}.model.1.weight".
                key = next(
                    (
                        k
                        for k in state
                        if k.startswith(f"lin{i}.") and k.endswith("weight")
                    ),
                    None,
                )
                if key is None:
                    continue
                weight = state[key].reshape(lin.weight.shape)
                lin.weight.data.copy_(weight)
                loaded += 1
            if loaded == len(self.lins):
                return True
            _LOGGER.warning(
                "LPIPS weights at %s covered %d/%d blocks; using uniform weights "
                "for the rest.",
                path,
                loaded,
                len(self.lins),
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            _LOGGER.warning(
                "Could not load LPIPS linear weights from %s: %s", path, exc
            )
        return False

    def train(self, mode: bool = True) -> "LPIPSPerceptualLoss":
        """Keep the frozen backbone in eval mode regardless of the parent's mode."""
        return super().train(False)

    @staticmethod
    def _unit_normalize(features: torch.Tensor) -> torch.Tensor:
        norm = features.pow(2).sum(dim=1, keepdim=True).sqrt()
        return features / (norm + 1e-10)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perceptual distance between two batches of images in [0, 1].

        Args:
            x: Images (B, 3, H, W), typically the ground-truth frames.
            y: Images (B, 3, H, W), typically the reconstructions.

        Returns:
            Scalar loss, averaged over the batch and summed over blocks.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        # [0, 1] -> [-1, 1], then LPIPS's own calibration.
        x = (x * 2.0 - 1.0 - self.shift) / self.scale
        y = (y * 2.0 - 1.0 - self.shift) / self.scale

        loss = x.new_zeros(())
        for block, lin in zip(self.blocks, self.lins):
            x = block(x)
            y = block(y)
            diff = (self._unit_normalize(x) - self._unit_normalize(y)).pow(2)
            # 1x1 conv weights the channels, then average spatially.
            loss = loss + lin(diff).mean()
        return loss


def build_perceptual_loss(
    enabled: bool = True,
    num_blocks: int = 5,
    linear_weights: Optional[Union[str, Path]] = None,
    download_weights: bool = True,
) -> Optional[nn.Module]:
    """Build the perceptual loss, returning ``None`` if it cannot be constructed.

    Loading VGG16 requires torchvision and, on first use, a weight download. When
    either is unavailable this returns ``None`` and logs a warning rather than
    failing training -- the autoencoder then falls back to the L1 + commitment
    objective, which trains but reconstructs small sprites poorly.

    ``vgg16`` is tried first (the variant LPIPS is calibrated on), then
    ``vgg16_bn`` as a fallback since it is more often cached locally.

    When ``linear_weights`` is not given, the reference LPIPS weights are looked
    up in the torch hub cache and fetched if missing, so the loss is calibrated
    LPIPS by default. Set ``download_weights=False`` to stay strictly offline.
    """
    if not enabled:
        return None

    if linear_weights is None:
        linear_weights = find_lpips_weights(download=download_weights)

    for use_batch_norm in (False, True):
        try:
            loss = LPIPSPerceptualLoss(
                num_blocks=num_blocks,
                use_batch_norm=use_batch_norm,
                linear_weights=linear_weights,
            )
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            variant = "vgg16_bn" if use_batch_norm else "vgg16"
            _LOGGER.warning("Could not build perceptual loss from %s: %s", variant, exc)
            continue

        if not loss.has_learned_weights:
            _LOGGER.info(
                "Perceptual loss is using uniform channel weights. This is the "
                "unweighted VGG-feature variant, not calibrated LPIPS; pass "
                "`perceptual_linear_weights` pointing at the reference LPIPS "
                "weights for an exact match."
            )
        return loss

    _LOGGER.warning(
        "Perceptual loss disabled. The autoencoder will train with L1 + commitment "
        "only, which tends to erase small objects (e.g. the ball in Pong)."
    )
    return None
