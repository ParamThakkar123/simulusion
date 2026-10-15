import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Optional, Dict, Literal

from simulusion.vision.video_tokenizer import VideoTokenizer
from simulusion.models.latent_action_model import LatentActionModel
from simulusion.models.dynamics_model import DynamicsModel, MaskGITSampler
from simulusion.configs.genie_config import GenieConfig, GenieSmallConfig
from simulusion.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from simulusion.export import ExportableAgentMixin


class Genie(ExportableAgentMixin, nn.Module):
    """Genie: Generative Interactive Environment.

    A generative model trained from video-only data that can be used as an
    interactive environment. Contains three key components:
    1. Video Tokenizer: Converts raw video frames into discrete tokens
    2. Latent Action Model (LAM): Infers latent actions between frames
    3. Dynamics Model: Predicts future frames given past frames and latent actions

    Based on "Genie: Generative Interactive Environments" paper (arXiv:2402.15391).

    Training follows two phases as per paper:
    1. Train video tokenizer first (on video tokens)
    2. Co-train LAM (from pixels) and dynamics model (on video tokens)

    The LAM uses VQ-VAE training with:
    - Encoder: Takes x1:t and x_{t+1} → outputs latent actions
    - Decoder: Takes x1:t-1 (masked) + actions → reconstructs x_t
    - Auxiliary variance loss to prevent action collapse

    At inference, latent actions are stopgrad'd when passed to dynamics model.
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_size: int = 64,
        in_channels: int = 3,
        tokenizer_vocab_size: int = 1024,
        tokenizer_embedding_dim: int = 32,
        tokenizer_encoder_dim: int = 512,
        tokenizer_decoder_dim: int = 1024,
        action_vocab_size: int = 8,
        action_embedding_dim: int = 32,
        action_encoder_dim: int = 1024,
        action_decoder_dim: int = 1024,
        dynamics_dim: int = 5120,
        dynamics_depth: int = 48,
        dynamics_num_heads: int = 40,
        encoder_depth: int = 12,
        decoder_depth: int = 20,
        latent_action_depth: int = 20,
        use_bfloat16: bool = False,
        action_pooling: Literal["mean", "windowed_attention"] = "mean",
        window_attention_heads: int = 1,
        tokenizer_num_heads: int = 16,
        action_num_heads: int = 16,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.action_vocab_size = action_vocab_size
        self.use_bfloat16 = use_bfloat16
        self.config = GenieConfig(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            tokenizer_vocab_size=tokenizer_vocab_size,
            tokenizer_embedding_dim=tokenizer_embedding_dim,
            tokenizer_encoder_dim=tokenizer_encoder_dim,
            tokenizer_decoder_dim=tokenizer_decoder_dim,
            tokenizer_encoder_depth=encoder_depth,
            tokenizer_decoder_depth=decoder_depth,
            tokenizer_num_heads=tokenizer_num_heads,
            action_vocab_size=action_vocab_size,
            action_embedding_dim=action_embedding_dim,
            action_encoder_dim=action_encoder_dim,
            action_decoder_dim=action_decoder_dim,
            action_encoder_depth=latent_action_depth,
            action_num_heads=action_num_heads,
            dynamics_dim=dynamics_dim,
            dynamics_depth=dynamics_depth,
            dynamics_num_heads=dynamics_num_heads,
            action_pooling=action_pooling,
            window_attention_heads=window_attention_heads,
        )

        # Video Tokenizer (VQ-VAE with ST-Transformer)
        self.video_tokenizer = VideoTokenizer(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            encoder_dim=tokenizer_encoder_dim,
            decoder_dim=tokenizer_decoder_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=tokenizer_num_heads,
            patch_size=4,
            vocab_size=tokenizer_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            use_ema=False,
        )

        # Latent Action Model (VQ-VAE with ST-Transformer decoder)
        self.latent_action_model = LatentActionModel(
            num_frames=num_frames,
            image_size=image_size,
            in_channels=in_channels,
            encoder_dim=action_encoder_dim,
            decoder_dim=action_decoder_dim,
            encoder_depth=latent_action_depth,
            decoder_depth=latent_action_depth,
            num_heads=action_num_heads,
            patch_size=16,
            vocab_size=action_vocab_size,
            embedding_dim=action_embedding_dim,
            commitment_weight=1.0,
            action_pooling=action_pooling,
            window_attention_heads=window_attention_heads,
        )

        # Dynamics Model (MaskGIT transformer)
        self.dynamics_model = DynamicsModel(
            num_frames=num_frames,
            image_size=image_size,
            vocab_size=tokenizer_vocab_size,
            embedding_dim=tokenizer_embedding_dim,
            action_vocab_size=action_vocab_size,
            dim=dynamics_dim,
            depth=dynamics_depth,
            num_heads=dynamics_num_heads,
            patch_size=4,
        )

        # MaskGIT sampler for inference
        self.sampler = MaskGITSampler(num_steps=25, temperature=2.0)

    @classmethod
    def from_config(
        cls,
        config: GenieConfig
        | GenieSmallConfig
        | dict[str, Any]
        | str
        | Path
        | None = None,
        **overrides: Any,
    ) -> "Genie":
        """Build Genie from a config object, dict, YAML file, or YAML string."""

        if isinstance(config, GenieSmallConfig):
            args = apply_config_overrides(config, overrides)
        else:
            args = apply_config_overrides(coerce_config(GenieConfig, config), overrides)
        return cls(**genie_kwargs_from_config(args))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: GenieConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        map_location: str | torch.device | None = None,
        **overrides: Any,
    ) -> "Genie":
        """Load Genie weights from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "genie.pt", "checkpoint.pt", "pytorch_model.bin")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a Genie checkpoint for {pretrained_model_name_or_path!r}."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or "cpu", weights_only=True
        )
        checkpoint_config = (
            checkpoint.get("config") if isinstance(checkpoint, dict) else None
        )
        if config is None and isinstance(checkpoint_config, dict):
            args = GenieConfig.from_dict(checkpoint_config)
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "genie_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = GenieConfig.from_yaml(config_path)
        else:
            args = coerce_config(GenieConfig, config)
        model = cls.from_config(apply_config_overrides(args, overrides))
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
        model.load_state_dict(state_dict)
        return model

    # Bumped when a change invalidates existing weights.
    #   v2: the ST-transformer's spatial attention merged its heads with a
    #       transpose that interleaved the head, spatial and temporal axes. The
    #       shapes matched, so nothing raised -- but every component (tokenizer,
    #       latent action model, dynamics) was non-causal, and the latent action
    #       model additionally read its actions from positions that had not seen
    #       the future frame. Weights from before this are meaningless, not
    #       merely mismatched. The dynamics model also gained a [MASK] token row.
    CHECKPOINT_FORMAT = 2

    def save_pretrained(self, path: str | Path) -> None:
        """Save Genie weights and config in a from_pretrained-compatible format."""

        checkpoint_path = Path(path)
        if checkpoint_path.suffix == "":
            checkpoint_path = checkpoint_path / "model.pt"
        save_config_next_to_checkpoint(self.config, checkpoint_path)
        torch.save(
            {
                "checkpoint_format": self.CHECKPOINT_FORMAT,
                "config": self.config.to_dict(),
                "model_state_dict": self.state_dict(),
            },
            checkpoint_path,
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            param.numel()
            for param in self.parameters()
            if not trainable_only or param.requires_grad
        )

    def summary(self) -> dict[str, Any]:
        return module_summary(
            {
                "video_tokenizer": self.video_tokenizer,
                "latent_action_model": self.latent_action_model,
                "dynamics_model": self.dynamics_model,
            }
        )

    def forward(
        self,
        video: torch.Tensor,
        mask_prob: float | None = None,
        training_phase: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through all components.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Dynamics masking rate. None (default) lets the
                dynamics model draw it from U(0.5, 1) per step, per paper 2.1.
            training_phase: "all", "tokenizer", or "lam_dynamics"

        Returns:
            Dictionary containing losses and predictions
        """
        B, _, T, _, _ = video.shape

        if training_phase == "tokenizer":
            # Phase 1: Train only video tokenizer
            recon_video, video_indices, tokenizer_loss_dict = self.video_tokenizer(
                video
            )
            return {
                "reconstructed_video": recon_video,
                "video_indices": video_indices,
                "tokenizer_loss": tokenizer_loss_dict,
                "vq_loss": tokenizer_loss_dict["vq_loss"],
                "total_loss": tokenizer_loss_dict["recon_loss"]
                + tokenizer_loss_dict["vq_loss"],
            }

        # Phase 2 or 3: Get video tokens first (frozen or training)
        with (
            torch.no_grad() if training_phase == "lam_dynamics" else torch.enable_grad()
        ):
            recon_video, video_indices, tokenizer_loss_dict = self.video_tokenizer(
                video
            )
            video_tokens = video_indices.reshape(B, T, -1)  # (B, T, H*W)

        # ===== LATENT ACTION MODEL =====
        # Train LAM from pixels - includes encoder + decoder losses
        lam_output = self.latent_action_model(
            video[:, :, :-1],  # x1:T-1
            video[:, :, -1],  # x_T
        )

        # Get latent actions - apply stopgrad for dynamics (as per paper)
        latent_actions = lam_output["latent_actions"]  # (B, T-1)
        z_q = lam_output["z_q"]  # (B, T-1, embedding_dim)

        # stopgrad on latent actions when passing to dynamics (per paper Section 2.1)
        z_q_for_dynamics = z_q.detach()

        # Map z_q to action indices for dynamics model
        # z_q is (B, T-1, embedding_dim), we need (B, T-1) indices
        # Use the latent_actions directly
        actions_for_dynamics = latent_actions[:, : T - 1]

        # ===== DYNAMICS MODEL =====
        # Predict next frame tokens given past tokens and latent actions
        # Input: video_tokens[:, :-1] (past frames), actions_for_dynamics
        # Target: video_tokens[:, 1:] (next frames)
        target_tokens = video_tokens[:, 1:, :]  # (B, T-1, H*W)

        dynamics_logits = self.dynamics_model(
            video_tokens[:, :-1, :],
            actions_for_dynamics,
            mask_prob=mask_prob,
        )

        # Compute dynamics loss
        B_pred, T_pred, N, V = dynamics_logits.shape
        target_flat = target_tokens.reshape(B_pred * T_pred * N)
        logits_flat = dynamics_logits.reshape(B_pred * T_pred * N, V)
        dynamics_loss = F.cross_entropy(logits_flat, target_flat)

        # ===== TOTAL LOSS =====
        # Paper 2.1 trains in two phases: "we train the video tokenizer first,
        # which is used for the dynamics model. We then co-train the latent
        # action model (directly from pixels) and the dynamics model (on video
        # tokens)." So in the co-training phase the objective is LAM + dynamics
        # only -- the tokenizer is frozen and its losses are reported for
        # monitoring, not optimised. Including them would both fight the
        # tokenizer's own converged solution and (once frozen) add a constant
        # that makes the training curve unreadable.
        lam_recon_loss = lam_output["recon_loss"]
        lam_vq_loss = lam_output["vq_loss"]

        total_loss = lam_recon_loss + lam_vq_loss + dynamics_loss
        if training_phase == "all":
            # Explicit opt-in to single-phase joint training of all three
            # components. Not the paper's protocol; useful for small-scale runs
            # where a separate tokenizer stage is not worth the wall-clock.
            total_loss = (
                total_loss
                + tokenizer_loss_dict["recon_loss"]
                + tokenizer_loss_dict["vq_loss"]
            )

        return {
            "reconstructed_video": recon_video,
            "video_indices": video_indices,
            "latent_actions": latent_actions,
            "lam_reconstructed": lam_output["reconstructed"],
            "dynamics_logits": dynamics_logits,
            "tokenizer_loss": tokenizer_loss_dict,
            "vq_loss": tokenizer_loss_dict["vq_loss"],
            "recon_loss": tokenizer_loss_dict["recon_loss"],
            "lam_recon_loss": lam_recon_loss,
            "lam_vq_loss": lam_vq_loss,
            "dynamics_loss": dynamics_loss,
            "z_q_for_dynamics": z_q_for_dynamics,
            "total_loss": total_loss,
        }

    def training_step(
        self,
        video: torch.Tensor,
        mask_prob: float | None = None,
        training_phase: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """Single training step computing all losses.

        Args:
            video: (B, C, T, H, W) input video
            mask_prob: Dynamics masking rate; None draws it per step.
            training_phase: "all", "tokenizer", or "lam_dynamics"

        Returns:
            Dictionary containing all losses for backpropagation
        """
        if self.use_bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.forward(video, mask_prob, training_phase)
        return self.forward(video, mask_prob, training_phase)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to discrete tokens.

        Args:
            video: (B, C, T, H, W)

        Returns:
            video_tokens: (B, T, H*W)
        """
        _, video_indices, _ = self.video_tokenizer(video)
        return video_indices.reshape(video_indices.shape[0], video_indices.shape[1], -1)

    def infer_actions(self, frames: torch.Tensor) -> torch.Tensor:
        """Infer latent actions from a sequence of frames.

        Args:
            frames: (B, C, T, H, W) video frames

        Returns:
            latent_actions: (B, T-1) inferred latent action indices
        """
        lam_output = self.latent_action_model(
            frames[:, :, :-1],
            frames[:, :, -1],
        )
        return lam_output["latent_actions"]

    def generate(
        self,
        prompt_frame: torch.Tensor,
        num_frames: int = 16,
        actions: Optional[torch.Tensor] = None,
        use_maskgit: bool = True,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Generate video frames given a prompt frame and actions.

        Args:
            prompt_frame: (B, C, H, W) initial frame
            num_frames: Total number of frames to generate
            actions: (B, num_frames-1) latent action indices, or None for random
            use_maskgit: Whether to use MaskGIT sampling
            use_cache: Generate each frame from a temporal KV cache instead of
                re-running the dynamics model over the whole prefix; O(T)
                rather than O(T^2) in the number of frames. Same logits up to
                float rounding.

        Returns:
            generated_video: (B, C, num_frames, H, W)
        """
        B, _, _, _ = prompt_frame.shape

        # Tokenize prompt frame
        prompt_frame_expanded = prompt_frame.unsqueeze(2).expand(
            -1, -1, num_frames, -1, -1
        )
        _, prompt_indices, _ = self.video_tokenizer(prompt_frame_expanded)

        # Use first frame tokens as prompt
        prompt_tokens = prompt_indices[:, 0, :, :].reshape(B, -1).unsqueeze(1)

        # Sample random actions if not provided
        if actions is None:
            actions = torch.randint(
                0,
                self.action_vocab_size,
                (B, num_frames - 1),
                device=prompt_frame.device,
            )

        # Generate
        if use_maskgit and hasattr(self, "sampler"):
            generated_tokens = self._generate_maskgit(
                prompt_tokens, actions, num_frames, use_cache=use_cache
            )
        else:
            generated_tokens = self.dynamics_model.autoregressive_sample(
                prompt_tokens[:, :1, :],
                actions,
                num_frames,
                temperature=2.0,
                use_cache=use_cache,
            )

        # Decode tokens to video
        z_generated = self.video_tokenizer.decode_indices(generated_tokens)
        generated_video = self.video_tokenizer.decode(z_generated)

        return generated_video

    def _generate_maskgit(
        self,
        prompt_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_frames: int,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate frame tokens, honoring the given actions.

        The Genie dynamics model is a decoder-only ST-transformer that predicts
        the *next* frame's tokens from past frame tokens ``z_{1:t}`` and latent
        actions ``a_{1:t}`` (paper Section 2.2). We generate one frame per outer
        step, feeding the user-provided action for each transition and sampling
        the predicted frame with the sampler's temperature.

        Note: a faithful within-frame MaskGIT reveal (25 iterative steps per
        frame) requires a dedicated ``[MASK]`` token in the tokenizer vocabulary
        so the dynamics model can re-attend to a partially-filled current frame.
        This module has no mask token and conditions only on completed past
        frames, so iterative refinement of a single frame is a no-op here; we
        therefore sample each frame in a single forward pass. Adding a mask
        token is the remaining architectural gap for true MaskGIT sampling.

        With ``use_cache`` the prompt is encoded once into a temporal KV cache
        and each new frame is one single-frame forward.
        """
        if use_cache:
            B, T_prompt, _ = prompt_tokens.shape
            cache = self.dynamics_model.init_cache(B, device=prompt_tokens.device)
            logits = self.dynamics_model.forward_cached(
                prompt_tokens, actions[:, :T_prompt], cache
            )
            frames = [prompt_tokens]
            t = T_prompt
            while t < num_frames:
                next_tokens = self.sampler.sample_frame(logits[:, -1]).unsqueeze(1)
                frames.append(next_tokens)
                t += 1
                if t < num_frames:
                    logits = self.dynamics_model.forward_cached(
                        next_tokens, actions[:, t - 1 : t], cache
                    )
            return torch.cat(frames, dim=1)

        current_tokens = prompt_tokens  # (B, T_prompt, N)

        while current_tokens.shape[1] < num_frames:
            t = current_tokens.shape[1]  # number of frames generated so far
            # Actions a_{1:t} aligned with the t input frames; the last one
            # (a_t) drives the t -> t+1 transition being predicted.
            acts = actions[:, :t]

            logits = self.dynamics_model(current_tokens, acts, mask_prob=0.0)
            next_frame_logits = logits[:, -1, :, :]  # (B, N, V) for frame t+1

            next_tokens = self.sampler.sample_frame(next_frame_logits)  # (B, N)
            current_tokens = torch.cat(
                [current_tokens, next_tokens.unsqueeze(1)], dim=1
            )

        return current_tokens

    def play(
        self,
        current_frame: torch.Tensor,
        action: torch.Tensor,
        current_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Play step - generate next frame given current frame and action.

        Args:
            current_frame: (B, C, H, W) current frame
            action: (B,) latent action indices
            current_frames: (B, C, T, H, W) history frames, or None for first frame

        Returns:
            next_frame: (B, C, H, W)
        """
        B, _, _, _ = current_frame.shape

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=current_frame.device)
        action = action.to(current_frame.device)

        if current_frames is None:
            current_frames = current_frame.unsqueeze(2)

        T_history = current_frames.shape[2]

        # Tokenize current frames
        _, prompt_indices, _ = self.video_tokenizer(current_frames)
        prompt_tokens = prompt_indices.reshape(B, T_history, -1)

        if action.dim() == 0:
            action = action.unsqueeze(0)
        action_expanded = action.unsqueeze(1).expand(-1, T_history)

        # Predict next frame
        next_frame_logits = self.dynamics_model(
            prompt_tokens,
            action_expanded,
            mask_prob=0.0,
        )

        next_frame_logits = next_frame_logits[:, -1, :, :]
        next_token_ids = torch.argmax(next_frame_logits, dim=-1)

        num_patches_per_side = int(next_token_ids.shape[1] ** 0.5)
        next_token_ids_reshaped = next_token_ids.reshape(
            B, num_patches_per_side, num_patches_per_side
        )

        z_next = self.video_tokenizer.decode_indices(
            next_token_ids_reshaped.unsqueeze(1)
        )
        next_frame = self.video_tokenizer.decode(z_next)

        return next_frame.squeeze(2)

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def genie_kwargs_from_config(
    config: GenieConfig | GenieSmallConfig | dict[str, Any],
) -> dict[str, Any]:
    """Map a Genie config onto :class:`Genie` constructor keyword arguments.

    The config and constructor name several fields differently (for example
    ``tokenizer_encoder_depth`` -> ``encoder_depth``), so filtering the config
    by parameter name silently drops them and builds those components at the
    constructor defaults. Every construction path goes through this mapping.
    """
    if not isinstance(config, (GenieConfig, GenieSmallConfig)):
        config = coerce_config(GenieConfig, config)
    return dict(
        num_frames=config.num_frames,
        image_size=config.image_size,
        in_channels=config.in_channels,
        tokenizer_vocab_size=config.tokenizer_vocab_size,
        tokenizer_embedding_dim=config.tokenizer_embedding_dim,
        tokenizer_encoder_dim=config.tokenizer_encoder_dim,
        tokenizer_decoder_dim=config.tokenizer_decoder_dim,
        action_vocab_size=config.action_vocab_size,
        action_embedding_dim=config.action_embedding_dim,
        action_encoder_dim=config.action_encoder_dim,
        action_decoder_dim=config.action_decoder_dim,
        encoder_depth=config.tokenizer_encoder_depth,
        decoder_depth=config.tokenizer_decoder_depth,
        latent_action_depth=config.action_encoder_depth,
        dynamics_dim=config.dynamics_dim,
        dynamics_depth=config.dynamics_depth,
        dynamics_num_heads=config.dynamics_num_heads,
        action_pooling=config.action_pooling,
        window_attention_heads=config.window_attention_heads,
        tokenizer_num_heads=config.tokenizer_num_heads,
        action_num_heads=config.action_num_heads,
    )


def create_genie(
    num_frames: int = 16,
    image_size: int = 64,
    in_channels: int = 3,
    tokenizer_vocab_size: int = 1024,
    tokenizer_embedding_dim: int = 32,
    action_vocab_size: int = 8,
    action_embedding_dim: int = 32,
    dynamics_dim: int = 5120,
    dynamics_depth: int = 48,
    dynamics_num_heads: int = 40,
    use_bfloat16: bool = False,
    action_pooling: Literal["mean", "windowed_attention"] = "mean",
    window_attention_heads: int = 1,
) -> Genie:
    """Factory function to create a Genie model."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        in_channels=in_channels,
        tokenizer_vocab_size=tokenizer_vocab_size,
        tokenizer_embedding_dim=tokenizer_embedding_dim,
        action_vocab_size=action_vocab_size,
        action_embedding_dim=action_embedding_dim,
        dynamics_dim=dynamics_dim,
        dynamics_depth=dynamics_depth,
        dynamics_num_heads=dynamics_num_heads,
        use_bfloat16=use_bfloat16,
        action_pooling=action_pooling,
        window_attention_heads=window_attention_heads,
    )


def create_genie_small(
    num_frames: int = 16,
    image_size: int = 64,
    use_bfloat16: bool = False,
    action_pooling: Literal["mean", "windowed_attention"] = "mean",
    window_attention_heads: int = 1,
) -> Genie:
    """Create a smaller Genie model for development/testing."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        tokenizer_encoder_dim=256,
        tokenizer_decoder_dim=512,
        action_vocab_size=8,
        action_embedding_dim=32,
        action_encoder_dim=512,
        action_decoder_dim=512,
        dynamics_dim=512,
        dynamics_depth=8,
        dynamics_num_heads=8,
        encoder_depth=4,
        decoder_depth=8,
        latent_action_depth=8,
        use_bfloat16=use_bfloat16,
        action_pooling=action_pooling,
        window_attention_heads=window_attention_heads,
    )


def create_genie_large(
    num_frames: int = 16,
    image_size: int = 64,
    use_bfloat16: bool = True,
    action_pooling: Literal["mean", "windowed_attention"] = "mean",
    window_attention_heads: int = 1,
) -> Genie:
    """Create the full 11B parameter Genie model (approximate)."""
    return Genie(
        num_frames=num_frames,
        image_size=image_size,
        tokenizer_vocab_size=1024,
        tokenizer_embedding_dim=32,
        tokenizer_encoder_dim=512,
        tokenizer_decoder_dim=1024,
        action_vocab_size=8,
        action_embedding_dim=32,
        action_encoder_dim=1024,
        action_decoder_dim=1024,
        dynamics_dim=5120,
        dynamics_depth=48,
        dynamics_num_heads=40,
        encoder_depth=12,
        decoder_depth=20,
        latent_action_depth=20,
        use_bfloat16=use_bfloat16,
        action_pooling=action_pooling,
        window_attention_heads=window_attention_heads,
    )
