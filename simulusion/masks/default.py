from logging import getLogger
from typing import Any

import torch

logger = getLogger()


class DefaultCollator(object):
    """Simple collator that returns batch data and no masking metadata.

    This is used when training code expects the JEPA-style collator return
    shape `(batch, masks_enc, masks_pred)` but masking is disabled.
    """

    def __call__(self, batch: Any) -> tuple:
        collated_batch = torch.utils.data.default_collate(batch)
        return collated_batch, None, None
