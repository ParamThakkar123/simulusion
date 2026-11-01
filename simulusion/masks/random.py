from multiprocessing import Value

from logging import getLogger
from typing import Any

import torch

logger = getLogger()


class MaskCollator(object):
    """Generate random context/prediction patch splits for masked training.

    A random permutation of patch indices is sampled per image; a configurable
    fraction is assigned to context and the remainder to prediction targets.
    """

    def __init__(
        self,
        ratio: tuple = (0.4, 0.6),
        input_size: tuple = (224, 224),
        patch_size: int = 16,
    ) -> None:
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.ratio = ratio
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch: Any) -> tuple:
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio
        ratio = ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - ratio[0])
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1 - ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(num_patches)
            collated_masks_enc.append(m[:num_keep])
            collated_masks_pred.append(m[num_keep:])

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
