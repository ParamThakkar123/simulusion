"""Ha & Schmidhuber's World Models (ConvVAE + MDN-RNN) as a
:class:`~simulusion.world_model.WorldModel`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from simulusion.world_model.protocol import (
    ActionSpec,
    State,
    Transition,
    WorldModelBase,
    WorldModelSpec,
    module_device,
)


class MDRNNWorldModel(WorldModelBase):
    """A ConvVAE for pixels and an MDN-RNN for dynamics (Ha & Schmidhuber, 2018).

    ``encode`` maps each frame to its VAE latent mean and runs the MDN-RNN's
    LSTM over the context. ``step`` predicts a Gaussian mixture over the next
    latent and samples it, at ``temperature`` as in the paper: the mixture
    weights are sharpened or flattened by it and the standard deviations
    scaled by its square root. The reward and done heads give ``reward`` and
    ``terminated`` (done logit above 0). Actions are continuous and unbounded.

    Args:
        vae: The :class:`~simulusion.vision.VAE.ConvVAE`.
        mdrnn: The MDN-RNN, as trained (:class:`~simulusion.models.mdrnn.MDRNN`)
            or as a single-step :class:`~simulusion.models.mdrnn.MDRNNCell`.
        temperature: Sampling temperature ``tau``; ``0`` takes the mean of the
            most likely mixture component.
    """

    family = "world-models"

    def __init__(self, vae: Any, mdrnn: Any, *, temperature: float = 1.0) -> None:
        if not (hasattr(vae, "encoder") and hasattr(vae, "decoder")):
            raise TypeError(f"{type(vae).__name__} is not a ConvVAE.")
        if not hasattr(mdrnn, "gmm_linear"):
            raise TypeError(f"{type(mdrnn).__name__} is not an MDN-RNN.")
        self.vae = vae.eval()
        self.cell = _as_cell(mdrnn).eval()
        self.temperature = float(temperature)
        self.device = module_device(vae)
        self.cell.to(self.device)
        self.spec = WorldModelSpec(
            family=self.family,
            obs_shape=(int(vae.encoder.img_channels), 64, 64),
            action=ActionSpec("continuous", int(self.cell.actions)),
            context_frames=1,
            max_context=None,
            has_reward=True,
            has_termination=True,
            stochastic=self.temperature > 0,
        )

    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        if actions is None:
            # The LSTM conditions on the actions between frames; without them
            # only the last frame is usable context.
            frames = frames[:, -1:]
        return self._encode_steps(frames, actions)[-1]

    def _encode_steps(
        self, frames: torch.Tensor, actions: Optional[torch.Tensor]
    ) -> list[State]:
        B, T = frames.shape[:2]
        mu, _ = self.vae.encoder(frames.flatten(0, 1))
        latents = mu.reshape(B, T, -1)
        h, c = self.cell.get_init_hidden(B)
        states = [{"z": latents[:, 0], "h": h, "c": c}]
        for t in range(1, T):
            if actions is not None:
                *_, (h, c) = self.cell(actions[:, t - 1], latents[:, t - 1], (h, c))
            states.append({"z": latents[:, t], "h": h, "c": c})
        return states

    def _step(self, state: State, action: torch.Tensor) -> Transition:
        mus, sigmas, logpi, reward, done, (h, c) = self.cell(
            action, state["z"], (state["h"], state["c"])
        )
        B = mus.shape[0]
        rows = torch.arange(B, device=self.device)
        if self.temperature == 0:
            k = logpi.argmax(dim=-1)
            z = mus[rows, k]
        else:
            k = torch.multinomial(F.softmax(logpi / self.temperature, dim=-1), 1)[:, 0]
            noise = torch.randn_like(mus[rows, k])
            z = mus[rows, k] + sigmas[rows, k] * self.temperature**0.5 * noise
        return Transition(
            state={"z": z, "h": h, "c": c},
            reward=reward.reshape(B),
            terminated=done.reshape(B) > 0,
            info={"done_prob": torch.sigmoid(done.reshape(B))},
        )

    def _decode(self, state: State) -> torch.Tensor:
        return self.vae.decoder(state["z"])

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    def _save_weights(self, directory: Path) -> dict[str, str]:
        torch.save(
            {"vae": self.vae.state_dict(), "mdrnn": self.cell.state_dict()},
            directory / "model.pt",
        )
        return {"model": "model.pt"}

    @classmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "MDRNNWorldModel":
        file = path / "model.pt" if path.is_dir() else path
        checkpoint = (
            torch.load(file, map_location="cpu", weights_only=True)
            if file.exists()
            else None
        )
        if not (isinstance(checkpoint, dict) and {"vae", "mdrnn"} <= set(checkpoint)):
            raise FileNotFoundError(
                f"{path} is not a saved World Models world model. Training "
                "writes the VAE and the MDN-RNN separately; load those with "
                "MDRNNWorldModel.from_checkpoints(vae_path, mdrnn_path)."
            )
        return cls._build(checkpoint["vae"], checkpoint["mdrnn"], device, options)

    @classmethod
    def from_checkpoints(
        cls,
        vae_checkpoint: str | Path,
        mdrnn_checkpoint: str | Path,
        *,
        device: str | torch.device | None = None,
        temperature: float = 1.0,
    ) -> "MDRNNWorldModel":
        """Load the files ``train_convvae`` and ``train_mdn_rnn`` write."""
        from simulusion.world_model.protocol import _resolve_device

        def weights(file: str | Path) -> dict[str, torch.Tensor]:
            state = torch.load(file, map_location="cpu", weights_only=True)
            return state.get("state_dict", state)

        return cls._build(
            weights(vae_checkpoint),
            weights(mdrnn_checkpoint),
            _resolve_device(device),
            {"temperature": temperature},
        )

    @classmethod
    def _build(
        cls,
        vae_weights: dict[str, torch.Tensor],
        mdrnn_weights: dict[str, torch.Tensor],
        device: torch.device,
        options: dict[str, Any],
    ) -> "MDRNNWorldModel":
        from simulusion.models.mdrnn import MDRNN, MDRNNCell
        from simulusion.vision.VAE.ConvVAE import ConvVAE

        latents = int(vae_weights["encoder.fc_mu.weight"].shape[0])
        vae = ConvVAE(
            img_channels=int(vae_weights["encoder.conv1.weight"].shape[1]),
            latent_size=latents,
        )
        vae.load_state_dict(vae_weights)
        sequence = "rnn.weight_ih_l0" in mdrnn_weights
        ih = mdrnn_weights["rnn.weight_ih_l0" if sequence else "rnn.weight_ih"]
        gmm_out = int(mdrnn_weights["gmm_linear.weight"].shape[0])
        sizes = dict(
            latents=latents,
            actions=int(ih.shape[1]) - latents,
            hiddens=int(ih.shape[0]) // 4,
            gaussians=(gmm_out - 2) // (2 * latents + 1),
        )
        mdrnn: Any = MDRNN(**sizes) if sequence else MDRNNCell(**sizes)
        mdrnn.load_state_dict(mdrnn_weights)
        return cls(vae.to(device), mdrnn.to(device), **options)


def _as_cell(mdrnn: Any) -> Any:
    """An :class:`MDRNNCell` with the weights of ``mdrnn`` (copied if it is an MDRNN)."""
    from simulusion.models.mdrnn import MDRNNCell

    if isinstance(mdrnn.rnn, torch.nn.LSTMCell):
        return mdrnn
    cell = MDRNNCell(mdrnn.latents, mdrnn.actions, mdrnn.hiddens, mdrnn.gaussians)
    rnn = mdrnn.rnn
    cell.rnn.load_state_dict(
        {
            "weight_ih": rnn.weight_ih_l0,
            "weight_hh": rnn.weight_hh_l0,
            "bias_ih": rnn.bias_ih_l0,
            "bias_hh": rnn.bias_hh_l0,
        }
    )
    cell.gmm_linear.load_state_dict(mdrnn.gmm_linear.state_dict())
    return cell.to(module_device(mdrnn))
