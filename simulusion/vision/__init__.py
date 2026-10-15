"""
Vision sub-module - Encoders, decoders, and visual processing components.

Exported Components:
    Encoders:
        - ConvEncoder: Dreamer convolutional encoder
        - CNNEncoder: PlaNet CNN encoder
        - IRISEncoder: IRIS-specific encoder

    Decoders:
        - ConvDecoder: Dreamer convolutional decoder
        - CNNDecoder: PlaNet CNN decoder
        - DenseDecoder: MLP decoder for rewards/values
        - ActionDecoder: Dreamer policy head
        - IRISDecoder: IRIS-specific decoder

    Video Processing:
        - VideoTokenizer: VQ-VAE video tokenizer for Genie
        - create_video_tokenizer: Factory for VideoTokenizer

    Quantization:
        - VectorQuantizer: Basic vector quantization
        - VectorQuantizerEMA: EMA-based vector quantization

    Distribution Utilities:
        - TanhBijector: Action squashing transformation
        - SampleDist: Distribution wrapper for sampling
"""

from typing import Any

__all__ = [
    # Encoders
    "ConvEncoder",
    "CNNEncoder",
    "IRISEncoder",
    # Decoders
    "ConvDecoder",
    "CNNDecoder",
    "DenseDecoder",
    "ActionDecoder",
    "IRISDecoder",
    # Video Processing
    "VideoTokenizer",
    "create_video_tokenizer",
    # Quantization
    "VectorQuantizer",
    "VectorQuantizerEMA",
    # Perceptual loss (IRIS / VQGAN autoencoder objective)
    "LPIPSPerceptualLoss",
    "build_perceptual_loss",
    # Distribution Utilities
    "TanhBijector",
    "SampleDist",
]


def __getattr__(name: str) -> Any:
    # Encoders
    if name == "ConvEncoder":
        from .dreamer_encoder import ConvEncoder

        return ConvEncoder
    if name == "CNNEncoder":
        from .planet_encoder import CNNEncoder

        return CNNEncoder
    if name == "IRISEncoder":
        from .iris_encoder import IRISEncoder

        return IRISEncoder

    # Perceptual loss
    if name == "LPIPSPerceptualLoss":
        from .perceptual_loss import LPIPSPerceptualLoss

        return LPIPSPerceptualLoss
    if name == "build_perceptual_loss":
        from .perceptual_loss import build_perceptual_loss

        return build_perceptual_loss

    # Decoders
    if name == "ConvDecoder":
        from .dreamer_decoder import ConvDecoder

        return ConvDecoder
    if name == "CNNDecoder":
        from .planet_decoder import CNNDecoder

        return CNNDecoder
    if name == "DenseDecoder":
        from .dreamer_decoder import DenseDecoder

        return DenseDecoder
    if name == "ActionDecoder":
        from .dreamer_decoder import ActionDecoder

        return ActionDecoder
    if name == "IRISDecoder":
        from .iris_decoder import IRISDecoder

        return IRISDecoder

    # Video Processing
    if name == "VideoTokenizer":
        from .video_tokenizer import VideoTokenizer

        return VideoTokenizer
    if name == "create_video_tokenizer":
        from .video_tokenizer import create_video_tokenizer

        return create_video_tokenizer

    # Quantization
    if name == "VectorQuantizer":
        from .vq_layer import VectorQuantizer

        return VectorQuantizer
    if name == "VectorQuantizerEMA":
        from .vq_layer import VectorQuantizerEMA

        return VectorQuantizerEMA

    # Distribution Utilities
    if name == "TanhBijector":
        from .dreamer_decoder import TanhBijector

        return TanhBijector
    if name == "SampleDist":
        from .dreamer_decoder import SampleDist

        return SampleDist

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
