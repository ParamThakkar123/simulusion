"""Tests for codebook-collapse handling in the vector quantizers.

A collapsed codebook is the quietest way for IRIS to fail: reconstruction loss
still falls (the decoder learns a constant), the world model still trains, and
only the perplexity metric reveals that the "language" the Transformer models
has a vocabulary of one.
"""

import pytest
import torch

from simulusion.vision.vq_layer import (
    DEFAULT_DEAD_CODE_THRESHOLD,
    VectorQuantizer,
    VectorQuantizerEMA,
)

QUANTIZERS = [VectorQuantizer, VectorQuantizerEMA]


def _quantizer(cls, vocab_size=64, embedding_dim=16):
    return cls(vocab_size=vocab_size, embedding_dim=embedding_dim)


@pytest.mark.parametrize("cls", QUANTIZERS)
class TestDeadCodeRevival:
    def test_usage_buffer_tracks_selection_counts(self, cls):
        """Usage must rise for selected codes and decay for unselected ones."""
        torch.manual_seed(0)
        q = cls(vocab_size=32, embedding_dim=8, restart_dead_codes_after=0.0).train()
        # Put one code exactly on the data so it wins every assignment.
        z = torch.zeros(4, 8, 2, 2)
        q.codebook.weight.data.normal_(0.0, 10.0)
        q.codebook.weight.data[7] = 0.0

        for _ in range(5):
            _, indices, _ = q(z)
        assert torch.all(indices == 7)

        usage = q.code_usage if isinstance(q, VectorQuantizer) else q.ema_cluster_size
        assert float(usage[7]) > 0.0, "winning code did not accumulate usage"
        others = torch.cat([usage[:7], usage[8:]])
        assert float(others.max()) == 0.0, "unselected codes accumulated usage"

    def test_usage_buffer_starts_at_zero(self, cls):
        """Dead codes must be detectable immediately, not after ~460 steps.

        With an EMA decay of 0.99 and a threshold of 0.01, a usage buffer
        initialised to ones keeps every unused entry above the threshold for
        hundreds of steps -- long past the point where collapse is entrenched.
        """
        q = _quantizer(cls)
        usage = q.code_usage if isinstance(q, VectorQuantizer) else q.ema_cluster_size
        assert float(usage.max()) == 0.0

    def test_no_revival_in_eval_mode(self, cls):
        """Evaluation must not mutate the codebook."""
        torch.manual_seed(0)
        q = _quantizer(cls).eval()
        before = q.codebook.weight.detach().clone()

        for _ in range(10):
            q(torch.randn(8, 16, 4, 4))

        torch.testing.assert_close(q.codebook.weight, before)

    def test_revival_can_be_disabled(self, cls):
        q = cls(vocab_size=64, embedding_dim=16, restart_dead_codes_after=0.0).train()
        before = q.codebook.weight.detach().clone()
        _, _, losses = q(torch.randn(8, 16, 4, 4))
        assert float(losses["dead_codes_restarted"]) == 0.0
        if isinstance(q, VectorQuantizer):
            # The gradient quantizer only changes its codebook via the optimizer,
            # so with revival off a forward pass must leave it untouched.
            torch.testing.assert_close(q.codebook.weight, before)

    def test_decode_indices_round_trip(self, cls):
        """decode_indices must accept the (B, H, W) grids forward() returns."""
        q = _quantizer(cls).train()
        z = torch.randn(4, 16, 4, 4)
        z_q, indices, _ = q(z)

        assert indices.shape == (4, 4, 4)
        decoded = q.decode_indices(indices)
        assert decoded.shape == z_q.shape


@pytest.mark.parametrize("quantizer", ["gradient", "ema"])
def test_revival_rescues_a_collapsed_encoder_codebook(quantizer):
    """Revival separates a usable codebook from a collapsed one.

    This runs the quantizer where collapse actually happens: downstream of a
    freshly initialised CNN encoder, whose outputs are tightly clustered. In
    that regime one or two entries win nearly every assignment, and without
    revival the state is absorbing -- unused entries receive no gradient and no
    EMA mass, so they can never win again. (Random Gaussian latents do *not*
    reproduce this, which is why the test pays for a real encoder.)
    """
    from simulusion.vision.iris_encoder import IRISEncoder

    def final_perplexity(revive: bool) -> float:
        torch.manual_seed(0)
        encoder = IRISEncoder(
            vocab_size=128, embedding_dim=32, base_channels=16, quantizer=quantizer
        ).train()
        encoder.quantizer.restart_dead_codes_after = (
            DEFAULT_DEAD_CODE_THRESHOLD if revive else 0.0
        )
        perplexity = 0.0
        for step in range(40):
            torch.manual_seed(100 + step)  # identical data for both runs
            _, _, losses = encoder(torch.rand(8, 3, 64, 64))
            perplexity = float(losses["perplexity"])
        return perplexity

    without_revival = final_perplexity(False)
    with_revival = final_perplexity(True)

    assert without_revival < 8.0, (
        f"expected collapse without revival, got perplexity {without_revival:.2f}/128"
    )
    assert with_revival > 24.0, (
        f"codebook stayed collapsed despite revival: {with_revival:.2f}/128"
    )


class TestThresholdSemantics:
    def test_default_threshold_is_conservative(self):
        """The threshold is in units of mean assignments per step.

        With V codes and roughly V assignments per step the mean usage is ~1, so
        a threshold anywhere near 1.0 would restart a large fraction of a healthy
        codebook on every step.
        """
        assert 0.0 < DEFAULT_DEAD_CODE_THRESHOLD < 0.1

    def test_healthy_codebook_is_left_alone(self):
        """A well-spread codebook should stop triggering restarts."""
        torch.manual_seed(0)
        q = VectorQuantizer(vocab_size=16, embedding_dim=8).train()
        # Spread the codebook over the data distribution up front.
        q.codebook.weight.data.normal_(0.0, 1.0)
        q.code_usage.fill_(1.0)

        restarts = [
            float(q(torch.randn(32, 8, 4, 4))[2]["dead_codes_restarted"])
            for _ in range(20)
        ]
        assert sum(restarts[-5:]) == 0.0, (
            "a healthy, fully-used codebook is still being restarted"
        )
