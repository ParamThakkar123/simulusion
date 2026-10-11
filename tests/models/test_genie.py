import pytest
import numpy as np
import torch
from simulusion.models import create_genie_small
from simulusion.vision import create_video_tokenizer
from simulusion.models.latent_action_model import create_latent_action_model


class TestVideoTokenizer:
    def test_initialization(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        assert tokenizer is not None

    def test_forward_pass(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        tokenizer.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            recon, indices, loss_dict = tokenizer(x)

        assert recon.shape == (B, C, T, H, W)
        assert indices.shape == (B, T, H // 4, W // 4)
        assert "recon_loss" in loss_dict
        assert "vq_loss" in loss_dict

    def test_encode_decode(self):
        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            decoder_dim=512,
            encoder_depth=4,
            decoder_depth=8,
            vocab_size=1024,
            embedding_dim=32,
        )
        tokenizer.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            z_q, indices, vq_loss = tokenizer.encode(x)
            recon = tokenizer.decode(z_q)

        assert z_q.shape[0] == B
        assert indices.shape[0] == B
        assert recon.shape == (B, C, T, H, W)

    @pytest.mark.parametrize("use_ema", [False, True])
    def test_encoded_embeddings_match_a_codebook_lookup_of_their_indices(self, use_ema):
        """Genie decodes generated frames with `decode_indices` (a codebook
        lookup), so `encode` must hand the decoder that same layout. It used to
        reshape the quantizer's channel-first output instead of permuting it,
        and every frame generated from tokens decoded to noise."""
        torch.manual_seed(0)
        tokenizer = create_video_tokenizer(
            num_frames=2,
            image_size=16,
            encoder_dim=32,
            decoder_dim=32,
            encoder_depth=1,
            decoder_depth=1,
            num_heads=2,
            vocab_size=16,
            embedding_dim=8,
            use_ema=use_ema,
        ).eval()

        with torch.no_grad():
            z_q, indices, _ = tokenizer.encode(torch.rand(2, 3, 2, 16, 16))
            looked_up = tokenizer.decode_indices(indices)

        assert torch.allclose(z_q, looked_up, atol=1e-5)


class TestLatentActionModel:
    def test_initialization(self):
        lam = create_latent_action_model(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            encoder_depth=4,
            num_heads=8,
            vocab_size=8,
            embedding_dim=32,
        )
        assert lam is not None

    def test_encode(self):
        lam = create_latent_action_model(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            encoder_depth=4,
            num_heads=8,
            vocab_size=8,
            embedding_dim=32,
        )
        lam.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x_prev = torch.randn(B, C, T, H, W)
        x_next = torch.randn(B, C, H, W)

        with torch.no_grad():
            latent_actions, z_q = lam.encode(x_prev, x_next)

        assert latent_actions.shape == (B, T)
        assert z_q.shape[0] == B

    def test_encode_with_windowed_attention_pooling(self):
        lam = create_latent_action_model(
            num_frames=8,
            image_size=32,
            encoder_dim=256,
            encoder_depth=4,
            num_heads=8,
            vocab_size=8,
            embedding_dim=32,
            action_pooling="windowed_attention",
            window_attention_heads=4,
        )
        lam.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x_prev = torch.randn(B, C, T, H, W)
        x_next = torch.randn(B, C, H, W)

        with torch.no_grad():
            latent_actions, z_q = lam.encode(x_prev, x_next)

        assert lam.action_pooling == "windowed_attention"
        assert latent_actions.shape == (B, T)
        assert z_q.shape == (B, T, 32)


class TestGenie:
    def test_initialization(self):
        model = create_genie_small(num_frames=8, image_size=32)
        assert model is not None

    def test_get_num_parameters(self):
        model = create_genie_small(num_frames=8, image_size=32)
        num_params = model.get_num_parameters()
        assert num_params > 0
        print(f"Model parameters: {num_params:,}")

    def test_forward_pass(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            outputs = model(x, mask_prob=0.5)

        assert "reconstructed_video" in outputs
        assert "video_indices" in outputs
        assert "latent_actions" in outputs
        assert "tokenizer_loss" in outputs

        assert outputs["reconstructed_video"].shape == (B, C, T, H, W)
        assert outputs["latent_actions"].shape[0] == B

    def test_different_batch_sizes(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 8, 32, 32)
            with torch.no_grad():
                outputs = model(x, mask_prob=0.5)
            assert outputs["reconstructed_video"].shape[0] == batch_size

    def test_different_frame_counts(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        x = torch.randn(2, 3, 4, 32, 32)
        with torch.no_grad():
            outputs = model(x, mask_prob=0.5)
        assert outputs["reconstructed_video"].shape == (2, 3, 4, 32, 32)


class TestGenieTraining:
    def test_training_step(self):
        from simulusion.training.train_genie import create_genie_trainer, GenieConfig

        config = GenieConfig()
        config.max_steps = 1
        config.num_frames = 8
        config.image_size = 32
        config.tokenizer_encoder_depth = 4
        config.tokenizer_decoder_depth = 8
        config.action_encoder_depth = 4
        config.dynamics_dim = 128
        config.dynamics_depth = 2

        trainer, model = create_genie_trainer(config)

        B, C, T, H, W = 2, 3, 8, 32, 32
        batch = torch.randn(B, C, T, H, W)

        losses = trainer.train_step(batch)
        assert "total_loss" in losses
        assert losses["total_loss"] > 0

    @staticmethod
    def _tiny_small_config():
        from simulusion.configs.genie_config import GenieSmallConfig

        config = GenieSmallConfig()
        config.num_frames = 4
        config.image_size = 16
        config.tokenizer_vocab_size = 16
        config.tokenizer_encoder_dim = 32
        config.tokenizer_decoder_dim = 32
        config.tokenizer_encoder_depth = 1
        config.tokenizer_decoder_depth = 1
        config.action_encoder_dim = 32
        config.action_decoder_dim = 32
        config.action_encoder_depth = 1
        config.dynamics_dim = 32
        config.dynamics_depth = 1
        config.dynamics_num_heads = 2
        return config

    def test_trainer_builds_the_configured_head_counts(self):
        from simulusion.training.train_genie import create_genie_trainer

        config = self._tiny_small_config()
        config.tokenizer_num_heads = 2
        config.action_num_heads = 4

        _, model = create_genie_trainer(config, torch.device("cpu"))

        def heads(module):
            return {m.num_heads for m in module.modules() if hasattr(m, "num_heads")}

        assert heads(model.video_tokenizer) == {2}
        assert heads(model.latent_action_model) == {4}

    def test_checkpoint_after_warmup_loads_with_weights_only(self, tmp_path):
        """Past warmup the cosine schedule must not leave numpy scalars in the
        scheduler state, which `torch.load(weights_only=True)` rejects."""
        from simulusion.training.train_genie import create_genie_trainer

        config = self._tiny_small_config()
        config.tokenizer_num_heads = 2
        config.action_num_heads = 2
        config.warmup_steps = 1
        config.max_steps = 10

        trainer, _ = create_genie_trainer(config, torch.device("cpu"))
        batch = torch.rand(
            2, 3, config.num_frames, config.image_size, config.image_size
        )
        for _ in range(3):
            trainer.train_step(batch)

        path = tmp_path / "genie.pt"
        trainer.save_checkpoint(str(path))
        trainer.load_checkpoint(str(path))
        assert trainer.global_step == 3

    def test_video_dataset_loads_npy_clips(self, tmp_path):
        from simulusion.training.train_genie import VideoDataset

        clip = np.random.rand(10, 48, 48, 3).astype(np.float32)
        path = tmp_path / "clip.npy"
        np.save(path, clip)
        item = VideoDataset([path], num_frames=8, image_size=32)[0]
        assert item.shape == (3, 8, 32, 32)
        assert item.dtype == torch.float32


class TestGenieGeneration:
    def test_generate_output_shape(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        prompt_frame = torch.randn(B, C, H, W)

        with torch.no_grad():
            generated = model.generate(prompt_frame, num_frames=8)

        assert generated.shape == (B, C, 8, H, W)

    def test_generate_with_actions(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        prompt_frame = torch.randn(B, C, H, W)
        actions = torch.randint(0, model.action_vocab_size, (B, 7))

        with torch.no_grad():
            generated = model.generate(prompt_frame, num_frames=8, actions=actions)

        assert generated.shape == (B, C, 8, H, W)

    def test_generate_autoregressive(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        prompt_frame = torch.randn(B, C, H, W)

        with torch.no_grad():
            generated = model.generate(prompt_frame, num_frames=8, use_maskgit=False)

        assert generated.shape == (B, C, 8, H, W)


class TestGeniePlay:
    def test_play_single_frame(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        current_frame = torch.randn(B, C, H, W)
        action = torch.tensor([1, 2])

        with torch.no_grad():
            next_frame = model.play(current_frame, action)

        assert next_frame.shape == (B, C, H, W)

    def test_play_with_history(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 4, 32, 32
        current_frame = torch.randn(B, C, H, W)
        current_frames = torch.randn(B, C, T, H, W)
        action = torch.tensor([1, 2])

        with torch.no_grad():
            next_frame = model.play(current_frame, action, current_frames)

        assert next_frame.shape == (B, C, H, W)

    def test_play_with_int_action(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        current_frame = torch.randn(B, C, H, W)
        action = 1

        with torch.no_grad():
            next_frame = model.play(current_frame, action)

        assert next_frame.shape == (B, C, H, W)


class TestGenieInferActions:
    def test_infer_actions_output_shape(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        frames = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            actions = model.infer_actions(frames)

        assert actions.shape == (B, T - 1)

    def test_infer_actions_different_frame_counts(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        for t in [4, 8, 16]:
            B, C, H, W = 2, 3, 32, 32
            frames = torch.randn(B, C, t, H, W)

            with torch.no_grad():
                actions = model.infer_actions(frames)

            assert actions.shape == (B, t - 1)


class TestGenieVideoTokenizer:
    def test_decode_indices(self):
        from simulusion.vision import create_video_tokenizer

        tokenizer = create_video_tokenizer(
            num_frames=8,
            image_size=32,
            encoder_dim=64,
            decoder_dim=128,
            encoder_depth=2,
            decoder_depth=2,
            vocab_size=256,
            embedding_dim=16,
        )
        tokenizer.eval()

        B, T = 2, 8
        H_idx = W_idx = 8
        indices = torch.randint(0, 256, (B, T, H_idx, W_idx))

        with torch.no_grad():
            z_q = tokenizer.decode_indices(indices)

        assert z_q.shape == (B, T, H_idx, W_idx, 16)


class TestGenieEdgeCases:
    def test_minimum_frames(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            outputs = model(x, mask_prob=0.0)

        assert "total_loss" in outputs

    def test_single_batch(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, T, H, W = 1, 3, 8, 32, 32
        x = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            outputs = model(x, mask_prob=0.5)

        assert outputs["reconstructed_video"].shape == (B, C, T, H, W)

    def test_different_image_sizes(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        for size in [32]:
            x = torch.randn(2, 3, 4, size, size)
            with torch.no_grad():
                outputs = model(x, mask_prob=0.0)
            assert outputs["reconstructed_video"].shape == (2, 3, 4, size, size)


class TestGenieGradients:
    def test_gradient_flow(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.train()

        x = torch.randn(1, 3, 4, 32, 32, requires_grad=True)

        outputs = model(x, mask_prob=0.5)

        loss = outputs["total_loss"]
        loss.backward()

        assert x.grad is not None

    def test_loss_components_backward(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.train()

        x = torch.randn(1, 3, 4, 32, 32)

        outputs = model(x, mask_prob=0.5)

        outputs["tokenizer_loss"]["recon_loss"].backward(retain_graph=True)
        outputs["dynamics_loss"].backward(retain_graph=True)


class TestGenieModelVariants:
    def test_create_genie(self):
        model = create_genie_small()
        assert model is not None

    def test_create_genie_small(self):
        model = create_genie_small(num_frames=8, image_size=32)
        assert model is not None
        num_params = model.get_num_parameters()
        assert num_params > 0


class TestGenieDevice:
    def test_cpu_device(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        x = torch.randn(2, 3, 4, 32, 32)

        with torch.no_grad():
            outputs = model(x)

        assert outputs["reconstructed_video"].device.type == "cpu"


class TestGenieEvalMode:
    def test_eval_mode(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        assert not model.training
        assert not model.video_tokenizer.training

        x = torch.randn(1, 3, 4, 32, 32)
        with torch.no_grad():
            _ = model(x)

    def test_train_mode(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.train()

        assert model.training

        x = torch.randn(1, 3, 4, 32, 32)
        _ = model(x)


class TestGenieLatentActions:
    def test_minimum_frames_for_infer(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32
        frames = torch.randn(B, C, 3, H, W)

        with torch.no_grad():
            actions = model.infer_actions(frames)

        assert actions.shape == (B, 2)

    def test_consistent_inference(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        B, C, T, H, W = 2, 3, 8, 32, 32
        frames1 = torch.randn(B, C, T, H, W)
        frames2 = torch.randn(B, C, T, H, W)

        with torch.no_grad():
            actions1 = model.infer_actions(frames1)
            actions2 = model.infer_actions(frames2)

        assert actions1.shape == actions2.shape


class TestGenieEndToEnd:
    def test_full_pipeline(self):
        model = create_genie_small(num_frames=4, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32

        prompt = torch.randn(B, C, H, W)

        with torch.no_grad():
            generated = model.generate(prompt, num_frames=4)

        assert generated.shape == (B, C, 4, H, W)

        action = torch.randint(0, model.action_vocab_size, (B,))

        with torch.no_grad():
            next_frame = model.play(prompt, action)

        assert next_frame.shape == (B, C, H, W)

    def test_play_after_generate(self):
        model = create_genie_small(num_frames=8, image_size=32)
        model.eval()

        B, C, H, W = 2, 3, 32, 32

        with torch.no_grad():
            generated = model.generate(torch.randn(B, C, H, W), num_frames=8)
            last_frame = generated[:, :, -1, :, :]

            action = torch.randint(0, model.action_vocab_size, (B,))
            next_frame = model.play(last_frame.squeeze(2), action)

        assert next_frame.shape == (B, C, H, W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
