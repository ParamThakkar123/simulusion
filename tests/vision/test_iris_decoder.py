import torch
import torch.nn as nn

from simulusion.vision.iris_decoder import IRISDecoder


def _decoder_and_codebook(vocab_size: int = 10, embedding_dim: int = 8):
    dec = IRISDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        base_channels=4,
        out_channels=3,
        frame_shape=(3, 64, 64),
    )
    # Stands in for IRISEncoder.quantizer.codebook -- the table the commitment
    # and reconstruction losses actually train. The decoder deliberately owns no
    # codebook of its own; a private one would never receive a gradient.
    codebook = nn.Embedding(vocab_size, embedding_dim)
    return dec, codebook


def test_decode_from_indices_matches_embeddings():
    torch.manual_seed(0)

    vocab_size = 10
    H = W = 4
    B = 2

    dec, codebook = _decoder_and_codebook(vocab_size=vocab_size)

    # Create deterministic indices within vocab range
    flat_indices = (torch.arange(H * W) % vocab_size).unsqueeze(0).repeat(B, 1)
    indices_hw = flat_indices.view(B, H, W)

    out_from_indices = dec.decode_from_indices(indices_hw, codebook)
    assert out_from_indices.shape == (B, 3, 64, 64)

    # Looking the indices up in the same codebook must give the same image.
    emb = codebook(flat_indices)
    out_from_emb = dec.decode_from_embeddings(emb)

    assert torch.allclose(out_from_indices, out_from_emb, atol=1e-6)

    # Ensure changing an index changes the output
    indices_hw2 = indices_hw.clone()
    indices_hw2[0, 0, 0] = (indices_hw2[0, 0, 0] + 1) % vocab_size
    out_changed = dec.decode_from_indices(indices_hw2, codebook)
    # outputs for the modified sample should differ
    assert not torch.allclose(out_from_indices[0], out_changed[0])

    # Also support flat (B, HW) input
    out_from_flat = dec.decode_from_indices(flat_indices, codebook)
    assert torch.allclose(out_from_flat, out_from_indices)


def test_decoder_owns_no_codebook():
    """A decoder-private token table would be trained by nothing.

    Paper A.1 has a single embedding table E, shared by the quantizer and the
    decode path. A second copy inside the decoder receives no gradient from the
    L1, commitment or perceptual terms, so decoding through it returns noise.
    """
    dec, _ = _decoder_and_codebook()
    assert not any(isinstance(module, nn.Embedding) for module in dec.modules()), (
        "IRISDecoder must not carry its own token embedding table"
    )
