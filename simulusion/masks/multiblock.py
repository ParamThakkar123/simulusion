import math

from multiprocessing import Value

from logging import getLogger
from typing import Any

import torch

logger = getLogger()


class MaskCollator(object):
    """Generate multi-block encoder and predictor masks for JEPA training.

    For each sample, this collator samples predictor target blocks and
    context encoder blocks (optionally non-overlapping), then returns masked
    patch indices aligned across the batch.

    Defaults follow the I-JEPA paper (Sec. 3 / Appendix A): 4 target blocks
    with scale in (0.15, 0.2) and aspect ratio in (0.75, 1.5), and 1 context
    block with scale in (0.85, 1.0) and unit aspect ratio, from which any
    region overlapping a target block is removed.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        enc_mask_scale: tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ) -> None:
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self, generator: torch.Generator, scale: tuple, aspect_ratio_scale: tuple
    ) -> tuple:
        # I-JEPA (Sec. 3) samples the block scale and the block aspect ratio
        # independently, so draw a fresh uniform for each. (The reference
        # implementation reuses a single draw for both, which correlates the
        # two perfectly and never yields e.g. a large block with a small
        # aspect ratio.)
        min_s, max_s = scale
        mask_scale = min_s + (max_s - min_s) * torch.rand(1, generator=generator).item()
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = (
            min_ar + (max_ar - min_ar) * torch.rand(1, generator=generator).item()
        )
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        # ensure at least 1 and not larger than grid
        h = max(1, min(h, self.height))
        w = max(1, min(w, self.width))

        return (h, w)

    def _sample_block_mask(
        self, b_size: tuple, acceptable_regions: Any = None
    ) -> tuple:
        h, w = b_size
        if h * w <= self.min_keep:
            # The retry loop below can never satisfy min_keep for a block this
            # small, so say so instead of spinning forever. This bites when the
            # paper's min_keep of 10 meets a crop small enough that a
            # (0.15, 0.2)-scale target block is only a few patches.
            raise ValueError(
                f"a {h}x{w} block holds at most {h * w} patches, which cannot "
                f"exceed min_keep={self.min_keep}; lower min_keep, raise the "
                f"mask scale, or use a larger crop / smaller patch size"
            )

        def constrain_mask(mask: torch.Tensor, tries: int = 0) -> None:
            # Restrict the candidate block to the intersection of the
            # acceptable regions (the complements of the already-sampled
            # target blocks), so the context block does not overlap the
            # targets. As failed attempts accumulate, progressively drop the
            # last acceptable regions to avoid an infinite loop.
            n = max(int(len(acceptable_regions) - tries), 0)
            for k in range(n):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # allow placement anywhere such that top+h <= height
            top = int(torch.randint(0, max(1, self.height - h + 1), (1,)).item())
            left = int(torch.randint(0, max(1, self.width - w + 1), (1,)).item())
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze(1)
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        return mask, mask_complement

    def __call__(self, batch: Any) -> tuple:
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0)
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions: Any = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f"Encountered exception in mask-generator {e}")

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
