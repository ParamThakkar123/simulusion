import pytest

torch = pytest.importorskip("torch")

import simulusion.blocks as blocks  # noqa: E402  (guarded by importorskip above)


def test_blocks_all_exports_are_resolvable():
    for name in blocks.__all__:
        assert getattr(blocks, name) is not None


def test_spatiotemporal_attention_exports_preserve_shapes():
    x = torch.randn(2, 3, 4, 8)

    spatial = blocks.STSpatialAttention(dim=8, num_heads=2)
    temporal = blocks.STTemporalAttention(dim=8, num_heads=2)

    assert spatial(x).shape == x.shape
    assert temporal(x).shape == x.shape


def test_backwards_compatible_aliases_point_to_real_classes():
    assert blocks.MultiHeadAttention is blocks.MultiHeadSelfAttention
    assert blocks.STBlock is blocks.STTransformerBlock
