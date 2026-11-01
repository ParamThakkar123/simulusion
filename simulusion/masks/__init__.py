"""
Masks sub-module - Masking strategies for JEPA and masked training.

This package provides various masking collator classes for generating
encoder/predictor masks during masked representation learning.

Usage:
    from simulusion.masks import MaskCollator, DefaultCollator
    collator = MaskCollator(input_size=(64, 64), patch_size=8)
"""

from .multiblock import MaskCollator as MultiblockMaskCollator
from .random import MaskCollator as RandomMaskCollator
from .default import DefaultCollator

__all__ = [
    "MultiblockMaskCollator",
    "RandomMaskCollator",
    "DefaultCollator",
]
