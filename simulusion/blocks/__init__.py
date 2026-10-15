"""
Blocks sub-module - Transformer blocks and attention mechanisms.

Exported Components:
    Transformers:
        - STTransformer: Spatiotemporal Transformer for video processing
        - STSpatialAttention: Spatial attention layer
        - STTemporalAttention: Temporal attention layer
        - STTransformerBlock: Combined spatiotemporal transformer block
        - STBlock: Backwards-compatible alias for STTransformerBlock

    Attention:
        - MultiHeadSelfAttention: Multi-head self-attention
        - MultiHeadAttention: Backwards-compatible alias for MultiHeadSelfAttention
        - Attention: Attention mechanism

    Normalization:
        - RMSNorm: Root Mean Square Layer Normalization
        - AdaLNNormalization: Adaptive Layer Normalization
"""

from typing import Any

__all__ = [
    # Transformers
    "STTransformer",
    "STSpatialAttention",
    "STTemporalAttention",
    "STTransformerBlock",
    "STBlock",
    # Attention
    "MultiHeadSelfAttention",
    "MultiHeadAttention",
    # Normalization
    "RMSNorm",
    "AdaLNNormalization",
]


def __getattr__(name: str) -> Any:
    # Transformers
    if name == "STTransformer":
        from .st_transformer import STTransformer

        return STTransformer
    if name == "STSpatialAttention":
        from .st_transformer import STSpatialAttention

        return STSpatialAttention
    if name == "STTemporalAttention":
        from .st_transformer import STTemporalAttention

        return STTemporalAttention
    if name == "STTransformerBlock":
        from .st_transformer import STTransformerBlock

        return STTransformerBlock
    if name == "STBlock":
        from .st_transformer import STTransformerBlock

        return STTransformerBlock

    # Attention
    if name == "MultiHeadSelfAttention":
        from .mhsa import MultiHeadSelfAttention

        return MultiHeadSelfAttention
    if name == "MultiHeadAttention":
        from .mhsa import MultiHeadSelfAttention

        return MultiHeadSelfAttention

    # Normalization
    if name == "RMSNorm":
        from simulusion.layers.rms_norm import RMSNorm

        return RMSNorm
    if name == "AdaLNNormalization":
        from simulusion.layers.ada_ln_norm import AdaLNNormalization

        return AdaLNNormalization

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
