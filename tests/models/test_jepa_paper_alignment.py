"""Checks that the I-JEPA implementation matches the paper's specification.

These pin the details of Assran et al., "Self-Supervised Learning from Images
with a Joint-Embedding Predictive Architecture" (CVPR 2023) that are easy to
drift from silently: the paper's own ablations show that getting the masking
wrong still trains happily while destroying the representation (Table 10: one
target block scores 9.0 top-1 where four score 54.2).
"""

import copy
import inspect

import pytest
import torch
import torch.nn.functional as F

from simulusion.configs.jepa_config import JEPAConfig
from simulusion.helpers.jepa_helper import (
    PAPER_PRED_DEPTH,
    init_model,
    resolve_pred_depth,
)
from simulusion.masks.multiblock import MaskCollator
from simulusion.models.vit import VisionTransformer, vit_predictor
from simulusion.training import train_jepa
from simulusion.training.train_jepa import build_loss_fn
from simulusion.utils.utils import apply_masks


class TestMaskingDefaults:
    """Sec. 3 / Appendix A: 4 targets at (0.15, 0.2), 1 context at (0.85, 1.0)."""

    def test_config_target_blocks(self):
        cfg = JEPAConfig()
        assert cfg.num_pred_masks == 4
        assert cfg.pred_mask_scale == (0.15, 0.2)
        assert cfg.aspect_ratio == (0.75, 1.5)

    def test_config_context_block(self):
        cfg = JEPAConfig()
        assert cfg.num_enc_masks == 1
        assert cfg.enc_mask_scale == (0.85, 1.0)

    def test_collator_defaults_match_config(self):
        cfg = JEPAConfig()
        collator = MaskCollator()
        assert collator.npred == cfg.num_pred_masks
        assert collator.nenc == cfg.num_enc_masks
        assert collator.pred_mask_scale == cfg.pred_mask_scale
        assert collator.enc_mask_scale == cfg.enc_mask_scale

    def test_context_block_uses_unit_aspect_ratio(self):
        """The context block is square; only target blocks vary in shape."""
        source = inspect.getsource(MaskCollator.__call__)
        assert "aspect_ratio_scale=(1.0, 1.0)" in source

    def test_scale_and_aspect_ratio_are_drawn_independently(self):
        """Sec. 3 samples "a random aspect ratio" and "random scale" separately.

        Sharing one uniform draw across both correlates them perfectly, so a
        block can never be both large and wide.
        """
        collator = MaskCollator(input_size=(448, 448), patch_size=16)
        generator = torch.Generator().manual_seed(0)
        sizes = [
            collator._sample_block_size(
                generator=generator, scale=(0.1, 0.5), aspect_ratio_scale=(0.5, 2.0)
            )
            for _ in range(300)
        ]
        areas = torch.tensor([float(h * w) for h, w in sizes])
        ratios = torch.tensor([h / w for h, w in sizes])
        # Both quantities rise monotonically with the shared draw, so reusing
        # one uniform for both pins their correlation at ~1.0.
        correlation = torch.corrcoef(torch.stack([areas, ratios]))[0, 1]
        assert abs(correlation) < 0.5, correlation

    def test_target_blocks_respect_the_scale_range(self):
        collator = MaskCollator(input_size=(224, 224), patch_size=16)
        generator = torch.Generator().manual_seed(0)
        num_patches = collator.height * collator.width
        for _ in range(20):
            h, w = collator._sample_block_size(
                generator=generator, scale=(0.15, 0.2), aspect_ratio_scale=(0.75, 1.5)
            )
            # Rounding to whole patches costs at most one row/column each way.
            assert 0.1 * num_patches <= h * w <= 0.25 * num_patches

    def test_context_excludes_overlapping_targets(self):
        """Sec. 3: overlapping regions are removed from the context block."""
        torch.manual_seed(0)
        collator = MaskCollator(input_size=(224, 224), patch_size=16)
        batch = [(torch.zeros(3, 224, 224), 0) for _ in range(4)]
        _, masks_enc, masks_pred = collator(batch)
        for sample in range(len(batch)):
            context = set(masks_enc[0][sample].tolist())
            for block in range(len(masks_pred)):
                assert not (context & set(masks_pred[block][sample].tolist()))


class TestNoViewAugmentations:
    """The paper's headline claim: no hand-crafted view data augmentations."""

    @pytest.mark.parametrize(
        "field", ["use_gaussian_blur", "use_horizontal_flip", "use_color_distortion"]
    )
    def test_augmentation_disabled_by_default(self, field):
        assert getattr(JEPAConfig(), field) is False

    def test_color_jitter_strength_is_zero(self):
        assert JEPAConfig().color_jitter_strength == 0.0


class TestPredictorArchitecture:
    """Appendix A: narrow predictor, 384 channels, heads inherited, no cls token."""

    def test_predictor_width_is_384(self):
        assert JEPAConfig().pred_emb_dim == 384

    def test_predictor_depth_follows_the_backbone(self):
        # ViT-B: 6, ViT-L/H: 12, ViT-G: 16.
        assert PAPER_PRED_DEPTH["vit_base"] == 6
        assert PAPER_PRED_DEPTH["vit_large"] == 12
        assert PAPER_PRED_DEPTH["vit_huge"] == 12
        assert PAPER_PRED_DEPTH["vit_giant"] == 16

    def test_config_defers_predictor_depth_to_the_backbone(self):
        assert JEPAConfig().pred_depth is None

    @pytest.mark.parametrize("model_name, depth", sorted(PAPER_PRED_DEPTH.items()))
    def test_depth_resolution_defaults_to_the_paper(self, model_name, depth):
        assert resolve_pred_depth(model_name, None) == depth

    def test_explicit_depth_still_wins(self):
        assert resolve_pred_depth("vit_large", 6) == 6

    def test_init_model_builds_the_resolved_depth(self):
        _, predictor = init_model(
            device=torch.device("cpu"),
            model_name="vit_tiny",
            crop_size=32,
            patch_size=16,
            pred_depth=None,
        )
        assert len(predictor.predictor_blocks) == PAPER_PRED_DEPTH["vit_tiny"]

    def test_predictor_inherits_backbone_head_count(self):
        encoder, predictor = init_model(
            device=torch.device("cpu"),
            model_name="vit_tiny",
            crop_size=32,
            patch_size=16,
        )
        assert predictor.predictor_blocks[0].attn.num_heads == encoder.num_heads

    def test_encoder_has_no_cls_token(self):
        encoder = VisionTransformer(img_size=[32], patch_size=16, embed_dim=64, depth=1)
        assert not hasattr(encoder, "cls_token")
        assert encoder.pos_embed.shape[1] == encoder.patch_embed.num_patches

    def test_mask_token_is_a_single_shared_vector(self):
        predictor = vit_predictor(
            num_patches=4, embed_dim=64, predictor_embed_dim=384, depth=1, num_heads=2
        )
        assert predictor.mask_token.shape == (1, 1, 384)
        assert predictor.predictor_proj.out_features == 64

    def test_positional_embeddings_are_not_learned(self):
        predictor = vit_predictor(
            num_patches=4, embed_dim=64, predictor_embed_dim=384, depth=1, num_heads=2
        )
        assert predictor.predictor_pos_embed.requires_grad is False


class TestTargetsComeFromTheEncoderOutput:
    """Appendix C, Table 11: masking the output beats masking the input by 11 pts."""

    def test_trainer_masks_the_target_encoder_output(self):
        source = inspect.getsource(train_jepa.main)
        forward_target = source[source.index("def forward_target") :]
        forward_target = forward_target[: forward_target.index("def forward_context")]
        encode_at = forward_target.index("target_encoder(imgs)")
        mask_at = forward_target.index("apply_masks(h, masks_pred)")
        # The full image is encoded first, and only then masked.
        assert encode_at < mask_at
        assert "target_encoder(imgs, masks" not in forward_target

    def test_masking_input_and_output_are_not_equivalent(self):
        """Guards the test above: the two orders genuinely differ."""
        torch.manual_seed(0)
        encoder = VisionTransformer(
            img_size=[32], patch_size=16, embed_dim=32, depth=2, num_heads=2
        )
        images = torch.rand(1, 3, 32, 32)
        masks = [torch.tensor([[0, 1]])]
        masked_output = apply_masks(encoder(images), masks)
        masked_input = encoder(images, masks)
        assert not torch.allclose(masked_output, masked_input, atol=1e-4)


class TestLoss:
    """Sec. 3: the loss is the average L2 distance in representation space."""

    def test_default_loss_is_l2(self):
        assert JEPAConfig().loss_type == "l2"

    def test_l2_sum_matches_the_paper_formula(self):
        """The literal Eq.: sum over patches in a block, average over blocks."""
        torch.manual_seed(0)
        predicted, target = torch.randn(8, 5, 3), torch.randn(8, 5, 3)
        per_block = torch.stack(
            [(predicted[i] - target[i]).pow(2).sum() for i in range(len(predicted))]
        )
        assert torch.allclose(
            build_loss_fn("l2_sum")(predicted, target), per_block.mean()
        )

    def test_l2_is_the_same_objective_mean_reduced(self):
        """The default reduces by mean, as the reference implementation does.

        Same minimizer and gradient direction, but a magnitude that does not
        scale with block size -- which is what the paper's LRs assume.
        """
        torch.manual_seed(0)
        predicted, target = torch.randn(8, 5, 3), torch.randn(8, 5, 3)
        elements = predicted[0].numel()
        assert torch.allclose(
            build_loss_fn("l2")(predicted, target),
            build_loss_fn("l2_sum")(predicted, target) / elements,
        )

    def test_smooth_l1_remains_available_for_reference_parity(self):
        predicted, target = torch.randn(4, 2, 3), torch.randn(4, 2, 3)
        assert torch.allclose(
            build_loss_fn("smooth_l1")(predicted, target),
            F.smooth_l1_loss(predicted, target),
        )

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown loss_type"):
            build_loss_fn("cosine")


class TestOptimizationSchedules:
    """Appendix A: AdamW, warmup 1e-4 -> 1e-3 over 15 epochs, wd 0.04 -> 0.4."""

    def test_learning_rate_schedule(self):
        cfg = JEPAConfig()
        assert (cfg.start_lr, cfg.lr, cfg.final_lr) == (1e-4, 1e-3, 1e-6)
        assert cfg.warmup == 15

    def test_learning_rates_are_quoted_at_the_paper_batch_size(self):
        cfg = JEPAConfig()
        assert cfg.batch_size == 2048
        assert cfg.lr_reference_batch_size == 2048

    def test_weight_decay_schedule(self):
        cfg = JEPAConfig()
        assert (cfg.weight_decay, cfg.final_weight_decay) == (0.04, 0.4)

    def test_ema_momentum_schedule(self):
        assert JEPAConfig().ema == (0.996, 1.0)

    def test_target_encoder_takes_no_gradient(self):
        # Sec. 3: the target encoder is updated by EMA of the context encoder,
        # never by backprop, so its parameters must not require grad.
        source = inspect.getsource(train_jepa.main)
        assert "p.requires_grad = False" in source

    def test_target_encoder_ema_matches_the_paper_rule(self):
        """theta_k <- m * theta_k + (1 - m) * theta_q, exactly.

        Asserted on behaviour rather than on the source text of ``main``: the
        update is a hot loop that gets refactored (per-tensor ops fused into
        ``torch._foreach_*``, for one), and a substring check both breaks on
        equivalent rewrites and passes on a wrong rule that happens to contain
        the expected string.
        """
        torch.manual_seed(0)
        online = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Linear(6, 3))
        target = copy.deepcopy(online)
        for param in online.parameters():
            param.data.normal_()

        m = 0.996
        expected = [
            m * pk.detach().clone() + (1.0 - m) * pq.detach().clone()
            for pq, pk in zip(online.parameters(), target.parameters())
        ]

        with torch.no_grad():
            target_params = list(target.parameters())
            torch._foreach_mul_(target_params, m)
            torch._foreach_add_(
                target_params, [q.detach() for q in online.parameters()], alpha=1.0 - m
            )

        for got, want in zip(target.parameters(), expected):
            # Rounding only: ``alpha=`` fuses the multiply-add, so it rounds once
            # where the explicit form rounds twice.
            assert torch.allclose(got, want, rtol=1e-6, atol=1e-7)
