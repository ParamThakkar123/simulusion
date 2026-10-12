import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Any, Tuple, Optional
import torch.nn.functional as F
from simulusion.utils.logging_utils import setup_logging

from simulusion.configs.iris_config import IRISConfig
from simulusion.models.model_io import (
    apply_config_overrides,
    coerce_config,
    module_summary,
    parameter_count as count_parameters,
    resolve_pretrained_file,
    save_config_next_to_checkpoint,
)
from simulusion.vision.iris_encoder import IRISEncoder
from simulusion.vision.iris_decoder import IRISDecoder
from simulusion.vision.perceptual_loss import build_perceptual_loss
from simulusion.models.iris_transformer import GraphedDecodeStep, IRISTransformer
from simulusion.controller.iris_policy import (
    CNNFeatureExtractor,
)
from simulusion.export import ExportableAgentMixin


def compute_lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    lambda_coef: float = 0.95,
) -> torch.Tensor:
    """Compute λ-return target for value function training.

    Args:
        rewards: Rewards (B, T)
        values: Value estimates (B, T+1)
        discounts: Discount factors (B, T)
        lambda_coef: Lambda parameter for bootstrapping

    Returns:
        lambda_returns: λ-return targets (B, T)
    """
    T = rewards.shape[1]
    lambda_returns = torch.zeros_like(rewards)

    # Start with the last bootstrapped value
    lambda_returns[:, T - 1] = rewards[:, T - 1] + discounts[:, T - 1] * values[:, T]

    # Compute λ-returns backwards
    for t in range(T - 2, -1, -1):
        lambda_returns[:, t] = rewards[:, t] + discounts[:, t] * (
            (1 - lambda_coef) * values[:, t + 1]
            + lambda_coef * lambda_returns[:, t + 1]
        )

    return lambda_returns


class IRISAgent(ExportableAgentMixin, nn.Module):
    """Complete IRIS Agent with world model and policy.

    Combines:
    - Discrete autoencoder (encoder + decoder)
    - Transformer world model
    - Actor-Critic for policy and value learning
    """

    def __init__(
        self,
        config: IRISConfig,
        action_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.config = coerce_config(IRISConfig, config)
        config = self.config
        self.action_size = action_size
        #: Replay imagination's single-position decode steps as a CUDA graph.
        #: Same computation, far fewer kernel launches; only used on CUDA.
        self.use_cuda_graphs = True
        #: Frames per autoencoder pass in :meth:`reconstruct`.
        self.reconstruct_chunk_size = 128
        self.device = device
        self.logger = setup_logging("IRISAgent")
        self.use_amp = bool(
            getattr(config, "use_amp", True)
            and getattr(device, "type", str(device)) == "cuda"
        )

        # === Discrete Autoencoder ===
        self.encoder = IRISEncoder(
            vocab_size=config.vocab_size,
            tokens_per_frame=config.tokens_per_frame,
            embedding_dim=config.token_embedding_dim,
            in_channels=config.frame_channels,
            base_channels=config.encoder_channels,
            num_layers=config.encoder_layers,
            num_residual_blocks=config.encoder_residual_blocks,
            frame_shape=config.get_frame_shape(),
            commitment_weight=config.commitment_weight,
            quantizer=config.quantizer,
        ).to(device)

        self.decoder = IRISDecoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.token_embedding_dim,
            base_channels=config.decoder_depth,
            out_channels=config.frame_channels,
            frame_shape=config.get_frame_shape(),
            num_residual_blocks=config.encoder_residual_blocks,
        ).to(device)

        # Perceptual loss (paper A.1). Frozen VGG features; excluded from the
        # autoencoder optimiser because it exposes no trainable parameters.
        self.perceptual_loss = build_perceptual_loss(
            enabled=config.perceptual_weight > 0.0,
            num_blocks=config.perceptual_blocks,
            linear_weights=config.perceptual_linear_weights or None,
        )
        if self.perceptual_loss is not None:
            self.perceptual_loss = self.perceptual_loss.to(device)

        # === Transformer World Model ===
        self.transformer = IRISTransformer(
            vocab_size=config.vocab_size,
            tokens_per_frame=config.tokens_per_frame,
            action_size=action_size,
            embed_dim=config.transformer_embed_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            dropout=config.transformer_dropout,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", True),
            # sign-transformed rewards are categorical over {-1, 0, +1}
            reward_classes=3 if config.reward_loss == "cross_entropy" else 1,
        ).to(device)

        # === Actor-Critic ===
        # Combine actor and critic with shared CNN features
        self.cnn = CNNFeatureExtractor(
            frame_shape=config.get_frame_shape(),
            output_size=config.actor_hidden_size,
        ).to(device)

        self.lstm = nn.LSTM(
            input_size=config.actor_hidden_size,
            hidden_size=config.actor_hidden_size,
            num_layers=config.actor_layers,
            batch_first=True,
        ).to(device)

        self.actor_head = nn.Linear(config.actor_hidden_size, action_size).to(device)
        self.critic_head = nn.Linear(config.actor_hidden_size, 1).to(device)

        # === Optimizers ===
        self._setup_optimizers()

        self.autoencoder_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.transformer_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ac_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # === Training state ===
        self.global_step = 0
        self.current_epoch = 0

    @classmethod
    def from_config(
        cls,
        config: IRISConfig | dict[str, Any] | str | Path | None = None,
        *,
        action_size: int,
        device: torch.device | str | None = None,
        **overrides: Any,
    ) -> "IRISAgent":
        """Build an IRIS agent from a config object, dict, YAML file, or YAML string."""

        args = apply_config_overrides(coerce_config(IRISConfig, config), overrides)
        torch_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        return cls(args, action_size=action_size, device=torch_device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        action_size: int | None = None,
        device: torch.device | str | None = None,
        config: IRISConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        **overrides: Any,
    ) -> "IRISAgent":
        """Load an IRIS agent checkpoint from a local path/directory or HF Hub."""

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("model.pt", "iris.pt", "checkpoint.pt", "ckpt.pt")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find an IRIS checkpoint for {pretrained_model_name_or_path!r}."
            )
        map_location = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
        checkpoint_config = checkpoint.get("config")
        if config is None and isinstance(checkpoint_config, IRISConfig):
            args = checkpoint_config
        elif config is None and isinstance(checkpoint_config, dict):
            args = IRISConfig.from_dict(checkpoint_config)
        elif config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "iris_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = IRISConfig.from_yaml(config_path)
        else:
            args = coerce_config(IRISConfig, config)
        args = apply_config_overrides(args, overrides)
        resolved_action_size = action_size or checkpoint.get("action_size")
        if resolved_action_size is None:
            raise ValueError(
                "action_size must be provided or present in the checkpoint."
            )
        agent = cls(args, action_size=int(resolved_action_size), device=map_location)
        agent.load(str(checkpoint_path))
        return agent

    def parameter_count(self, trainable_only: bool = False) -> int:
        return count_parameters(self, trainable_only=trainable_only)

    def summary(self) -> dict[str, Any]:
        return module_summary(
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "transformer": self.transformer,
                "cnn": self.cnn,
                "lstm": self.lstm,
                "actor_head": self.actor_head,
                "critic_head": self.critic_head,
            }
        )

    @staticmethod
    def _decay_param_groups(
        module: nn.Module, weight_decay: float
    ) -> list[dict[str, Any]]:
        """Split parameters into decayed and non-decayed groups (minGPT).

        Weight decay is applied to the weight matrices of Linear/Conv layers
        only. Biases, LayerNorm/GroupNorm affine parameters and embedding tables
        are excluded: decaying them shrinks the model's ability to represent
        offsets and token identities, and is not what "weight decay 0.01" in the
        paper's Table 4 refers to.
        """
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []

        decay_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
        skip_modules = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.Embedding)

        seen: set[int] = set()
        for submodule in module.modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                if param_name.endswith("bias") or isinstance(submodule, skip_modules):
                    no_decay.append(param)
                elif isinstance(submodule, decay_modules):
                    decay.append(param)
                else:
                    # Bare nn.Parameter (e.g. positional embeddings, VQ codebook
                    # scale): treat like an embedding and leave it undecayed.
                    no_decay.append(param)

        groups: list[dict[str, Any]] = []
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def _setup_optimizers(self) -> None:
        """Setup separate optimizers for each component.

        Paper Table 4 lists weight decay under the Transformer's hyperparameters,
        and Table 5 gives a single learning rate of 1e-4 with Adam. AdamW is used
        so the decay is decoupled from the gradient (plain Adam's
        ``weight_decay`` is an L2 penalty folded into the gradient, which
        interacts badly with adaptive scaling).
        """
        betas = (self.config.adam_beta1, self.config.adam_beta2)

        # Autoencoder: no weight decay (not specified by the paper for the
        # discrete autoencoder, and the VQ codebook must not be shrunk).
        self.autoencoder_opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.model_learning_rate,
            betas=betas,
        )

        # Transformer: weight decay on matmul weights only (paper Table 4).
        self.transformer_opt = optim.AdamW(
            self._decay_param_groups(self.transformer, self.config.weight_decay),
            lr=self.config.model_learning_rate,
            betas=betas,
        )

        # Actor-Critic: no weight decay (paper Table 6 lists none). The trunk is
        # shared (A.3), so it is optimised at the actor learning rate; only the
        # critic's own last layer uses value_learning_rate. With the paper's
        # settings both rates are 1e-4, making this a single group in practice.
        self.ac_opt = optim.Adam(
            [
                {
                    "params": (
                        list(self.cnn.parameters())
                        + list(self.lstm.parameters())
                        + list(self.actor_head.parameters())
                    ),
                    "lr": self.config.actor_learning_rate,
                },
                {
                    "params": list(self.critic_head.parameters()),
                    "lr": self.config.value_learning_rate,
                },
            ],
            lr=self.config.actor_learning_rate,
            betas=betas,
        )

    @staticmethod
    def _losses_to_floats(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        keys = list(losses.keys())
        values = torch.stack([losses[key].detach() for key in keys]).cpu().tolist()
        return dict(zip(keys, values))

    def forward_actor_critic(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through actor-critic.

        Args:
            frames: Input frames (B, T, C, H, W)
            hidden: Optional LSTM hidden state

        Returns:
            action_logits: (B, T, action_size)
            values: (B, T)
            hidden_state: (h, c)
        """
        B, T, C, H, W = frames.shape

        # CNN features
        frames_flat = frames.reshape(B * T, C, H, W)
        features = self.cnn(frames_flat)
        features = features.reshape(B, T, -1)

        # LSTM
        if hidden is None:
            hidden = self._init_lstm_hidden(B)

        lstm_out, new_hidden = self.lstm(features, hidden)

        # Action and value
        action_logits = self.actor_head(lstm_out)
        values = self.critic_head(lstm_out).squeeze(-1)

        return action_logits, values, new_hidden

    def _init_lstm_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(
            self.config.actor_layers,
            batch_size,
            self.config.actor_hidden_size,
            device=self.device,
        )
        c = torch.zeros(
            self.config.actor_layers,
            batch_size,
            self.config.actor_hidden_size,
            device=self.device,
        )
        return (h, c)

    def act(
        self,
        frame: torch.Tensor,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample action from policy.

        The policy is recurrent (paper A.3: CNN -> LSTM). Callers stepping an
        episode must thread ``hidden`` from one call to the next and reset it on
        episode boundaries; dropping it makes the policy effectively memoryless,
        which for games like Pong removes any way to infer the ball's direction
        from a single frame.

        Args:
            frame: Single frame (B, C, H, W)
            epsilon: Random action probability
            temperature: Action distribution temperature
            hidden: LSTM state from the previous step, or None to start fresh
            return_hidden: If True, also return the updated LSTM state

        Returns:
            actions: Selected actions (B,), and the updated LSTM state when
            ``return_hidden`` is set.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                B = frame.shape[0]
                frames = frame.unsqueeze(1)  # (B, 1, C, H, W)

                action_logits, _, new_hidden = self.forward_actor_critic(
                    frames, hidden=hidden
                )
                action_logits = action_logits.squeeze(1) / temperature

                # Sample from the policy (paper Table 5: epsilon-greedy combined
                # with sampling), then replace a fraction ``epsilon`` with
                # uniformly random actions. Taking the argmax here instead made
                # collection almost deterministic: an untrained policy repeated
                # one action for thousands of steps and never explored.
                probs = torch.softmax(action_logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
                if epsilon > 0:
                    random_mask = torch.rand(B, device=self.device) < epsilon
                    random_actions = torch.randint(
                        0, self.action_size, (B,), device=self.device
                    )
                    actions = torch.where(random_mask, random_actions, actions)
        finally:
            # Don't leave the module in eval mode if the caller was training.
            self.train(was_training)

        if return_hidden:
            return actions, new_hidden
        return actions

    @torch.no_grad()
    def burn_in(
        self, frames: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialise the LSTM state by replaying preceding frames.

        Paper A.3: "Before starting the imagination procedure from a given frame,
        we burn-in the 20 previous frames to initialize the hidden state"
        (Kapturowski et al., 2019). Without this the rollout begins from a zero
        state that the policy never sees at collection time.

        Args:
            frames: Preceding observations (B, T_burn, C, H, W), already in the
                reconstruction domain the policy is trained on.

        Returns:
            The LSTM state after the burn-in, or None if no frames were given.
        """
        if frames is None or frames.shape[1] == 0:
            return None
        was_training = self.training
        self.eval()
        try:
            _, _, hidden = self.forward_actor_critic(frames)
        finally:
            self.train(was_training)
        return hidden

    def transform_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply the configured reward transform.

        Atari rewards are unbounded integers -- in the thousands for games like
        Krull or UpNDown -- and feeding them raw into the value function and
        lambda-return makes the critic's target scale game-dependent. The
        standard Atari convention, which IRIS follows, is to take the sign.
        """
        if self.config.reward_transform == "sign":
            return torch.sign(rewards)
        if self.config.reward_transform == "none":
            return rewards
        raise ValueError(
            f"Unknown reward_transform {self.config.reward_transform!r}; "
            "expected 'sign' or 'none'."
        )

    @torch.no_grad()
    def reconstruct(self, frames: torch.Tensor) -> torch.Tensor:
        """Pass frames through the discrete autoencoder: ``D(E(x))``.

        Paper A.1: "during experience collection in the real environment, frames
        still go through the autoencoder to keep the input distribution of the
        policy unchanged". The policy only ever learns from reconstructions
        during imagination, so feeding it raw frames in the real environment is a
        distribution shift.

        The encoder is forced into eval mode for the duration. This is not
        cosmetic: the quantizer's dead-code revival only runs in training mode,
        and this method is called on single frames during experience collection
        and evaluation, when the agent is still in training mode. Leaving it
        there re-seeds most of the codebook from the 16 encoder outputs of one
        frame on *every* environment step, destroying the vocabulary the world
        model is being trained against.

        Args:
            frames: Real observations (B, C, H, W) or (B, T, C, H, W) in [0, 1].

        Returns:
            Reconstructions with the same shape, clamped to [0, 1].
        """
        leading = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:])
        if flat.shape[0] == 0:
            return frames

        encoder_was_training = self.encoder.training
        decoder_was_training = self.decoder.training
        self.encoder.eval()
        self.decoder.eval()
        try:
            # Chunked so a long burn-in batch (B x 20 frames) does not need the
            # decoder's full activation memory at once; on a small GPU that
            # spills to system memory and runs several times slower.
            chunk = self.reconstruct_chunk_size
            recon = torch.cat(
                [
                    self.decoder(self.encoder(part)[0]).clamp(0.0, 1.0)
                    for part in flat.split(chunk)
                ]
            )
        finally:
            self.encoder.train(encoder_was_training)
            self.decoder.train(decoder_was_training)
        return recon.reshape(*leading, *recon.shape[1:])

    def imagine_rollout(
        self,
        initial_frame: torch.Tensor,
        horizon: int = 20,
        burn_in_frames: Optional[torch.Tensor] = None,
        sample_tokens: bool = True,
        temperature: float = 1.0,
        stop_on_termination: bool = True,
    ) -> dict:
        """Generate imagined trajectories using the world model.

        Implements the imagination MDP of paper 2.3 / Figure 1: starting from a
        real observation, the policy acts on decoded frames while the Transformer
        rolls the dynamics forward, conditioned on the full imagined history.

        Args:
            initial_frame: Starting frame x_0 (B, C, H, W), in [0, 1].
            horizon: Number of steps H to imagine.
            burn_in_frames: The (B, T_burn, C, H, W) reconstructed observations
                preceding ``initial_frame``, used to initialise the policy's LSTM
                state (paper A.3). None starts from a zero state.
            sample_tokens: Sample next-frame tokens instead of taking the argmax.
            temperature: Sampling temperature for token generation.
            stop_on_termination: Cut the rollout short once every trajectory in
                the batch has hit a predicted episode end (paper 2.3). Set False
                to always return exactly ``horizon`` steps; correctness does not
                depend on it, since ``continues`` already zeroes the discount
                past a termination.

        Returns:
            trajectory: dict with ``frames`` (B, T+1, C, H, W), ``actions``
            (B, T), ``rewards`` (B, T) and ``continues`` (B, T). T may be shorter
            than ``horizon`` if every rollout predicted an episode end.
        """
        was_training = self.training
        self.eval()
        B = initial_frame.shape[0]
        K = self.config.tokens_per_frame
        tokens_per_dim = int(K**0.5)

        # Encode initial frame
        with torch.no_grad():
            _, initial_tokens, _ = self.encoder(initial_frame)
        current_tokens = initial_tokens.reshape(B, K)  # (B, K)

        # Prime the world model's KV cache with z_0. Every subsequent step
        # attends over the whole imagined history (paper 2.3, Fig 1):
        #   z_{t+1} ~ p(. | z_0, a_0, z_1, a_1, ..., z_t, a_t)
        # Sized to what this rollout can fill: z_0 plus ``horizon`` (action +
        # frame) blocks. The graphed decode step attends over the whole buffer,
        # so unused capacity would cost time without changing the result.
        cache = self.transformer.init_cache(
            B, self.device, max_len=K + horizon * (K + 1)
        )
        # Capture before priming: capture writes to position 0 of the cache.
        decode_step = (
            GraphedDecodeStep(self.transformer, cache, B)
            if self.use_cuda_graphs and self.device.type == "cuda"
            else None
        )
        pos = self.transformer.prime_cache(
            current_tokens.unsqueeze(1), None, cache, start_pos=0
        )
        # History kept so the cache can be rebuilt if it runs out of capacity.
        token_history = [current_tokens]
        action_history: list[torch.Tensor] = []
        # Retaining T frames and T-1 actions costs (T-1)*(K+1) + K positions, and
        # the step that follows needs another K+1. So the window must satisfy
        # T*(K+1) + K <= max_seq_len, otherwise a rebuild would leave no room to
        # generate and immediately overflow again.
        capacity_limit = (cache.max_len - K) // (K + 1)
        max_context = max(1, min(self.config.transformer_timesteps, capacity_limit))

        # LSTM state for the policy, initialised from the frames preceding x_0.
        hidden = self.burn_in(burn_in_frames) if burn_in_frames is not None else None

        # Lists to store trajectory. The policy consumes reconstructed frames, so
        # every stored frame is a decode of the current tokens (no raw/decoded
        # duplicate at t=0). We collect ``horizon`` (frame, action, reward)
        # triples plus one trailing frame -> ``horizon + 1`` frames total.
        frames_imagined = []
        actions_imagined = []
        rewards_imagined = []
        continues_imagined = []

        # Hard "this rollout has already ended" mask, used only to decide when to
        # stop early. The soft per-step continue probabilities are what feed the
        # lambda-return discount.
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for _step in range(horizon):
            grid = current_tokens.reshape(B, tokens_per_dim, tokens_per_dim)

            # Decode current tokens to get the "observation" the policy sees.
            with torch.no_grad():
                reconstructed_frame = self.decoder(
                    self.encoder.quantizer.decode_indices(grid)
                ).clamp(0.0, 1.0)
                frames_imagined.append(reconstructed_frame)

            # Get action from the recurrent policy, carrying its LSTM state.
            act_out = self.act(
                reconstructed_frame,
                epsilon=0.0,
                hidden=hidden,
                return_hidden=True,
            )
            assert isinstance(act_out, tuple)
            action, hidden = act_out

            # If the cache cannot fit another (action + frame) block, rebuild it
            # from the most recent ``max_context`` timesteps. Absolute positional
            # embeddings make in-place trimming incorrect, so the window is
            # re-primed from scratch instead.
            if pos + (K + 1) > cache.max_len:
                token_history = token_history[-max_context:]
                action_history = action_history[-(len(token_history) - 1) :]
                # Reset in place rather than allocating a new cache, so the
                # captured decode graph keeps pointing at the buffers in use.
                cache.reset()
                pos = self.transformer.prime_cache(
                    torch.stack(token_history, dim=1),
                    (torch.stack(action_history, dim=1) if action_history else None),
                    cache,
                    start_pos=0,
                )

            # Predict next tokens. Sampling (rather than argmax) keeps the
            # imagined futures diverse -- a greedy world model collapses every
            # rollout onto the same trajectory and the policy sees no variety.
            with torch.no_grad():
                _, next_tokens, action_hidden, pos = (
                    self.transformer.generate_frame_cached(
                        action,
                        cache,
                        start_pos=pos,
                        sample=sample_tokens,
                        temperature=temperature,
                        decode_step=decode_step,
                    )
                )

                # Get reward and termination predictions. expected_reward folds
                # the categorical head back to a scalar via its expectation.
                reward_pred = self.transformer.expected_reward(action_hidden)
                term_logits = self.transformer.termination_head(action_hidden)
                term_prob = torch.softmax(term_logits, dim=-1)[:, 1]

            actions_imagined.append(action)
            # Rewards are stored unscaled: the lambda-return recursion applies
            # gamma * (1 - d_t) to the *future* term, not to r_t itself
            # (paper eq. 4), so scaling here would discount r_t twice.
            rewards_imagined.append(reward_pred)
            continues_imagined.append(1.0 - term_prob)

            action_history.append(action)
            token_history.append(next_tokens)
            current_tokens = next_tokens

            # Paper 2.3: "We stop if an episode end is predicted before reaching
            # the horizon." Trigger on the actual argmax prediction rather than a
            # decaying probability product, which for an untrained termination
            # head would truncate every rollout after a handful of steps.
            finished = finished | (term_logits.argmax(dim=-1) == 1)
            if stop_on_termination and bool(finished.all()):
                break

        # Append the final imagined frame so ``frames`` has one more entry than
        # actions -- the trailing frame supplies the bootstrap value V(x_H).
        with torch.no_grad():
            final_grid = current_tokens.reshape(B, tokens_per_dim, tokens_per_dim)
            frames_imagined.append(
                self.decoder(self.encoder.quantizer.decode_indices(final_grid)).clamp(
                    0.0, 1.0
                )
            )

        self.train(was_training)

        return {
            "frames": torch.stack(frames_imagined, dim=1),  # (B, T+1, C, H, W)
            "actions": (
                torch.stack(actions_imagined, dim=1) if actions_imagined else None
            ),
            "rewards": (
                torch.stack(rewards_imagined, dim=1) if rewards_imagined else None
            ),
            # (B, T) soft "episode still running" mask from the termination head.
            "continues": (
                torch.stack(continues_imagined, dim=1) if continues_imagined else None
            ),
        }

    def update_autoencoder(self, frames: torch.Tensor) -> dict:
        """Update discrete autoencoder.

        Args:
            frames: Training frames (B, C, H, W)

        Returns:
            losses: Dictionary of loss values
        """
        self.encoder.train()
        self.decoder.train()

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Encode
            z_q, indices, vq_loss = self.encoder(frames)

            # Decode
            reconstruction = self.decoder(z_q)

            # Paper A.1: L1 + commitment + perceptual, equally weighted.
            recon_loss = F.l1_loss(reconstruction, frames)
            loss = self.config.reconstruction_weight * recon_loss + vq_loss["vq_loss"]

            if self.perceptual_loss is not None:
                # VGG expects [0, 1]; the decoder is unbounded, so clamp rather
                # than let out-of-range values distort the feature statistics.
                perc_loss = self.perceptual_loss(frames, reconstruction.clamp(0.0, 1.0))
                loss = loss + self.config.perceptual_weight * perc_loss
            else:
                perc_loss = torch.zeros((), device=frames.device)

        # Update
        self.autoencoder_opt.zero_grad(set_to_none=True)
        self.autoencoder_scaler.scale(loss).backward()
        self.autoencoder_scaler.unscale_(self.autoencoder_opt)
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.config.grad_clip_norm,
        )
        self.autoencoder_scaler.step(self.autoencoder_opt)
        self.autoencoder_scaler.update()

        losses = self._losses_to_floats(
            {
                "recon_loss": recon_loss,
                "vq_loss": vq_loss["vq_loss"],
                "perceptual_loss": perc_loss,
                "perplexity": vq_loss["perplexity"],
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Autoencoder update: {losses}")
        return losses

    def update_transformer(
        self,
        frames: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T)
        rewards: torch.Tensor,  # (B, T)
        terminals: torch.Tensor,  # (B, T)
    ) -> dict:
        """Update transformer world model.

        Args:
            frames: Frame sequence
            actions: Actions taken
            rewards: Rewards received
            terminals: Terminal flags

        Returns:
            losses: Dictionary of loss values
        """
        self.transformer.train()

        B, T_plus_1 = frames.shape[:2]

        # Encode all frames to tokens. The autoencoder is trained by its own
        # objective (paper A.1), so this is detached: letting the token-prediction
        # loss reach the encoder would both waste compute and leave stale
        # gradients on parameters the transformer optimiser does not own. The
        # whole sequence is encoded in one batched call rather than a per-step
        # Python loop.
        with torch.no_grad():
            self.encoder.eval()
            flat = frames.reshape(B * T_plus_1, *frames.shape[2:])
            _, indices, _ = self.encoder(flat)
            tokens = indices.reshape(B, T_plus_1, -1)  # (B, T+1, K)

        # Convert actions from (B, T, action_size) one-hot to (B, T) scalar
        # indices. The cast to long is not optional: nn.Embedding rejects float
        # indices with an error far from this call site, and the replay buffer
        # stores actions as float32.
        if actions.dim() == 3:
            actions = actions.argmax(dim=-1)  # (B, T)
        actions = actions.long()

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Get predictions. The transformer consumes the full T+1 frame
            # sequence (teacher forcing) and predicts frames 1..T.
            token_logits, rewards_pred, terms_pred = self.transformer(
                tokens,  # (B, T+1, K)
                actions,  # (B, T)
            )

            # Token prediction loss
            next_tokens = tokens[:, 1:]  # (B, T, K)
            token_loss = F.cross_entropy(
                token_logits.reshape(-1, self.config.vocab_size),
                next_tokens.reshape(-1),
            )

            # Reward loss. With sign-transformed rewards the target is a class
            # index in {-1, 0, +1} -> {0, 1, 2}; otherwise regress the scalar.
            if self.config.reward_loss == "cross_entropy":
                reward_targets = (
                    self.transform_reward(rewards).long()
                    + self.transformer.reward_classes // 2
                )
                reward_loss = F.cross_entropy(
                    rewards_pred.reshape(-1, self.transformer.reward_classes),
                    reward_targets.reshape(-1),
                )
            else:
                reward_loss = F.mse_loss(rewards_pred, self.transform_reward(rewards))

            # Termination loss (cross-entropy)
            term_loss = F.cross_entropy(
                terms_pred.reshape(-1, 2),
                terminals.reshape(-1),
            )

            # Total loss
            # Paper 2.2 lists the transition, reward and termination losses
            # without relative weights, so they are summed as-is. The previous
            # 0.1 factors on reward and termination had no basis in the paper
            # and made the world model slow to learn exactly the two signals the
            # RL objective depends on.
            loss = token_loss + reward_loss + term_loss

        # Update
        self.transformer_opt.zero_grad(set_to_none=True)
        self.transformer_scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.transformer_scaler.unscale_(self.transformer_opt)
        nn.utils.clip_grad_norm_(
            self.transformer.parameters(), self.config.grad_clip_norm
        )
        self.transformer_scaler.step(self.transformer_opt)
        self.transformer_scaler.update()

        losses = self._losses_to_floats(
            {
                "token_loss": token_loss,
                "reward_loss": reward_loss,
                "term_loss": term_loss,
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Transformer update: {losses}")
        return losses

    def update_actor_critic(
        self,
        imagined_trajectory: dict,
    ) -> dict:
        """Update actor-critic in imagination.

        Args:
            imagined_trajectory: Dictionary from imagine_rollout

        Returns:
            losses: Dictionary of loss values
        """
        self.train()

        frames = imagined_trajectory["frames"]  # (B, T+1, C, H, W)
        actions = imagined_trajectory["actions"]  # (B, T)
        rewards = imagined_trajectory["rewards"]  # (B, T)

        B, T_plus_1, C, H, W = frames.shape

        with torch.amp.autocast(
            device_type=getattr(self.device, "type", str(self.device)),
            enabled=self.use_amp,
        ):
            # Forward pass over all T+1 frames: the trailing frame supplies the
            # bootstrap value V(x_H) for the lambda-return. Previously only the
            # first T frames were forwarded and the bootstrap was hardcoded to
            # zero, which biased every return target toward 0.
            all_logits, all_values, _ = self.forward_actor_critic(
                frames
            )  # (B, T+1, A), (B, T+1)
            action_logits = all_logits[:, :-1]  # (B, T, A)
            values = all_values[:, :-1]  # (B, T)

            # Compute log probabilities
            action_dist = torch.softmax(action_logits, dim=-1)
            action_log_probs = torch.log(action_dist + 1e-8)

            # Gather log probs for taken actions
            actions_one_hot = F.one_hot(actions, self.action_size).float()
            taken_log_probs = (action_log_probs * actions_one_hot).sum(dim=-1)  # (B, T)

            # Compute λ-returns. Discounts are cut at predicted episode ends so
            # imagined rewards past a terminal state do not leak into the target.
            continues = imagined_trajectory.get("continues")
            discounts = torch.full_like(rewards, self.config.discount)
            if continues is not None:
                discounts = discounts * continues.to(discounts.dtype)
            lambda_returns = compute_lambda_return(
                rewards,
                all_values,  # (B, T+1); all_values[:, T] is the bootstrap V(x_H)
                discounts,
                self.config.td_lambda,
            )

            # Advantage
            advantages = lambda_returns - values  # (B, T)

            # Actor loss (REINFORCE with baseline)
            actor_loss = -(taken_log_probs * advantages.detach()).mean()

            # Entropy bonus
            entropy = -(action_dist * action_log_probs).sum(dim=-1).mean()
            actor_loss -= self.config.entropy_coef * entropy

            # Critic loss
            value_loss = F.mse_loss(values, lambda_returns.detach())

            # Total loss. Appendix B states L_V (eq. 5) and L_pi (eq. 6) as two
            # objectives with no relative weight, and the actor and critic share
            # a trunk (A.3), so they are summed as-is. The previous 0.5 on the
            # critic had no basis in the paper and quietly halved the value
            # head's effective learning rate -- and with it the quality of the
            # baseline every REINFORCE advantage is measured against.
            loss = actor_loss + value_loss

        # Update
        self.ac_opt.zero_grad(set_to_none=True)
        self.ac_scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.ac_scaler.unscale_(self.ac_opt)
        nn.utils.clip_grad_norm_(
            list(self.cnn.parameters())
            + list(self.lstm.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_head.parameters()),
            self.config.grad_clip_norm,
        )
        self.ac_scaler.step(self.ac_opt)
        self.ac_scaler.update()

        losses = self._losses_to_floats(
            {
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": loss,
            }
        )
        self.logger.debug(f"Actor-critic update: {losses}")
        return losses

    # Bumped when the module layout changes in a way that makes older
    # checkpoints unloadable.
    #   v2: Transformer's nn.TransformerEncoder stack replaced with GPT-2 blocks
    #       exposing per-layer KV caches.
    #   v3: encoder/decoder gained per-layer residual stacks and the decoder
    #       widened to 64 channels (Table 2); the actor-critic conv block moved
    #       from strided convolutions to conv + max-pool (A.3).
    #   v4: decoder gained self-attention at 8/16 and both halves moved their
    #       attention blocks into an `attentions` ModuleDict; the reward head
    #       became categorical over {-1, 0, +1} (2.2).
    #   v5: encoder/decoder convolutions hold a constant 64 channels instead of
    #       doubling per layer (Table 2), and the decoder's private, never-
    #       trained `index_to_embedding` table was removed.
    CHECKPOINT_FORMAT = 5

    def save(self, path: str) -> None:
        """Save agent state."""
        save_config_next_to_checkpoint(self.config, path)
        torch.save(
            {
                "checkpoint_format": self.CHECKPOINT_FORMAT,
                "config": self.config.to_dict(),
                "action_size": int(self.action_size),
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "transformer": self.transformer.state_dict(),
                "cnn": self.cnn.state_dict(),
                "lstm": self.lstm.state_dict(),
                "actor_head": self.actor_head.state_dict(),
                "critic_head": self.critic_head.state_dict(),
                "autoencoder_opt": self.autoencoder_opt.state_dict(),
                "transformer_opt": self.transformer_opt.state_dict(),
                "ac_opt": self.ac_opt.state_dict(),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        with torch.serialization.safe_globals([IRISConfig]):
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )

        found_format = int(checkpoint.get("checkpoint_format", 1))
        if found_format != self.CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"{path} was written in IRIS checkpoint format v{found_format}, "
                f"but this build expects v{self.CHECKPOINT_FORMAT}. The module "
                "layout has changed (GPT-2 Transformer blocks with key/value "
                "caches; per-layer residual stacks in the autoencoder; a "
                "max-pooling actor-critic conv block; constant-width "
                "encoder/decoder convolutions), so the weights cannot be "
                "mapped across. Retrain, or check out the older revision to use "
                "this checkpoint."
            )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.transformer.load_state_dict(checkpoint["transformer"])
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.lstm.load_state_dict(checkpoint["lstm"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])

        self.autoencoder_opt.load_state_dict(checkpoint["autoencoder_opt"])
        self.transformer_opt.load_state_dict(checkpoint["transformer_opt"])
        self.ac_opt.load_state_dict(checkpoint["ac_opt"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
