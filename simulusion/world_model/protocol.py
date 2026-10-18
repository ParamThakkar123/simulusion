"""The protocol every Simulusion world model implements.

A world model predicts how observations evolve under actions. Each family
(Dreamer, PlaNet, IRIS, DIAMOND, Genie, the VAE + MDN-RNN World Models) does
this differently inside: an RNN latent, transformer tokens, a diffusion
sampler. :class:`WorldModel` gives them one outside:

.. code-block:: python

    state = model.encode(frames, actions)       # real context -> model state
    out = model.step(state, action)             # one imagined transition
    image = model.decode(out.state)             # model state -> pixels
    traj = model.rollout(state, action_seq)     # many steps at once

Conventions, the same for every family:

* **Frames** are RGB, channel-first, batch-first: ``(B, T, C, H, W)``, float
  in ``[0, 1]`` or ``uint8`` in ``[0, 255]``. A 4-D input is one sequence
  ``(T, C, H, W)`` and a 3-D input one frame. Frames of another size are
  resized to :attr:`WorldModelSpec.obs_shape`. :meth:`WorldModel.decode`
  always returns float ``(B, C, H, W)`` in ``[0, 1]``.
* **Actions** between context frames: ``actions[:, t]`` is the action taken at
  frame ``t``, which led to frame ``t + 1``, so ``T`` frames come with
  ``T - 1`` actions. Discrete actions are integer ``(B, ...)``, continuous ones
  float ``(B, ..., A)``.
* **State** is a ``dict[str, Tensor]`` with the batch on dim 0. It is plain
  data: copy it to branch a dream, stack it to batch dreams.
* **Rewards** and **terminations** come from the model's own heads, ``(B,)``
  per step, or ``None`` when the family has no such head (see
  :class:`WorldModelSpec`).
* **Randomness** comes from PyTorch's global RNG, so ``torch.manual_seed``
  before a rollout makes it repeatable.

Every method runs without autograd and puts the model in eval mode: this is
the inference surface. Training stays with each family's trainer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn.functional as F

State = Dict[str, torch.Tensor]

#: Written next to the weights by :meth:`WorldModelBase.save`.
MANIFEST_NAME = "world_model.json"
MANIFEST_FORMAT = "simulusion-world-model"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ActionSpec:
    """The actions a world model accepts.

    Attributes:
        kind: ``"discrete"`` (integer actions in ``[0, n)``) or
            ``"continuous"`` (float vectors of length ``n``).
        n: Number of actions (discrete) or action dimensions (continuous).
        low: Lower bound of continuous actions, or ``None`` if unbounded.
        high: Upper bound of continuous actions, or ``None`` if unbounded.
        latent: The actions were learned from video rather than taken from an
            environment (Genie), so what each one does is for the model to say.
    """

    kind: Literal["discrete", "continuous"]
    n: int
    low: Optional[float] = None
    high: Optional[float] = None
    latent: bool = False

    def as_tensor(
        self, action: Any, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """One step's actions as ``(B,)`` int64 or ``(B, n)`` float32.

        Accepts a tensor, array, list or scalar. A single action is repeated
        across the batch. Continuous actions are clipped to ``[low, high]``.
        """
        if self.kind == "discrete":
            tensor = torch.as_tensor(np.asarray(_to_numpy(action)), device=device)
            tensor = tensor.reshape(-1).long()
            if tensor.numel() == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size)
            if tensor.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} discrete actions, got {tensor.numel()}."
                )
            if bool(((tensor < 0) | (tensor >= self.n)).any()):
                raise ValueError(f"Actions must lie in [0, {self.n}).")
            return tensor
        tensor = torch.as_tensor(
            np.asarray(_to_numpy(action), dtype=np.float32), device=device
        )
        tensor = tensor.reshape(-1, self.n) if tensor.numel() else tensor
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, self.n)
        if tuple(tensor.shape) != (batch_size, self.n):
            raise ValueError(
                f"Expected continuous actions of shape ({batch_size}, {self.n}), "
                f"got {tuple(tensor.shape)}."
            )
        if self.low is not None or self.high is not None:
            tensor = tensor.clamp(self.low, self.high)
        return tensor


@dataclass(frozen=True)
class WorldModelSpec:
    """What a world model consumes and produces.

    Attributes:
        family: ``"dreamer"``, ``"planet"``, ``"iris"``, ``"diamond"``,
            ``"genie"`` or ``"world-models"``.
        obs_shape: ``(C, H, W)`` of the frames it reads and decodes.
        action: The actions :meth:`WorldModel.step` accepts.
        context_frames: The fewest frames :meth:`WorldModel.encode` needs.
        max_context: The most frames the model attends to; older ones are
            forgotten. ``None`` for recurrent models, which carry everything
            in their state.
        has_reward: :class:`Transition` carries a predicted reward.
        has_termination: :class:`Transition` carries a predicted episode end.
        stochastic: :meth:`WorldModel.step` samples, so repeated calls on the
            same state differ unless the RNG is reseeded.
    """

    family: str
    obs_shape: tuple[int, int, int]
    action: ActionSpec
    context_frames: int = 1
    max_context: Optional[int] = None
    has_reward: bool = True
    has_termination: bool = False
    stochastic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transition:
    """One imagined step, as returned by :meth:`WorldModel.step`.

    Attributes:
        state: The model state after the step.
        reward: Predicted reward ``(B,)``, or ``None`` without a reward head.
        terminated: Predicted episode end ``(B,)`` bool, or ``None``.
        info: Family-specific extras, such as Dreamer's continue probability.
    """

    state: State
    reward: Optional[torch.Tensor] = None
    terminated: Optional[torch.Tensor] = None
    info: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Rollout:
    """Many imagined steps, as returned by :meth:`WorldModel.rollout`.

    Attributes:
        state: The model state after the last step.
        frames: Decoded frames ``(B, H, C, H, W)`` in ``[0, 1]``, one per step,
            or ``None`` when the rollout was run with ``decode=False``.
        rewards: ``(B, H)``, or ``None`` without a reward head.
        terminated: ``(B, H)`` bool, or ``None`` without a termination head.
    """

    state: State
    frames: Optional[torch.Tensor]
    rewards: Optional[torch.Tensor]
    terminated: Optional[torch.Tensor]


@runtime_checkable
class WorldModel(Protocol):
    """The interface every world model in Simulusion implements.

    Wrap a trained model with :func:`~simulusion.world_model.as_world_model`,
    or load one with :func:`~simulusion.world_model.load_world_model`. See the
    module docstring for the conventions.
    """

    spec: WorldModelSpec
    device: torch.device

    def encode(self, frames: Any, actions: Any = None) -> State:
        """Turn real context frames (and the actions between them) into a state."""
        ...

    def step(self, state: State, action: Any) -> Transition:
        """Imagine one step: the state after taking ``action``."""
        ...

    def decode(self, state: State) -> torch.Tensor:
        """The current frame of ``state``: float ``(B, C, H, W)`` in ``[0, 1]``."""
        ...

    def rollout(self, state: State, actions: Any, *, decode: bool = True) -> Rollout:
        """Imagine ``H`` steps; ``actions`` is ``(B, H)`` or ``(B, H, A)``."""
        ...

    def save(self, path: str | Path) -> Path:
        """Write the model to directory ``path``; returns it."""
        ...


class WorldModelBase(ABC):
    """Shared plumbing for the built-in world models.

    Subclasses set :attr:`family`, build :attr:`spec` and :attr:`device` in
    ``__init__``, and implement :meth:`_encode`, :meth:`_step`,
    :meth:`_decode`, :meth:`_save_weights` and :meth:`_load_weights`. Input
    normalisation, autograd, :meth:`rollout` and the save format live here.
    """

    #: Registry key, also written to the manifest.
    family: ClassVar[str]
    spec: WorldModelSpec
    device: torch.device

    # ---------------------------------------------------------------- public

    def encode(self, frames: Any, actions: Any = None) -> State:
        video = self.as_frames(frames)
        if video.shape[1] < self.spec.context_frames:
            raise ValueError(
                f"{type(self).__name__} needs at least {self.spec.context_frames} "
                f"context frames, got {video.shape[1]}."
            )
        acts = self._context_actions(actions, video.shape[0], video.shape[1])
        with torch.no_grad():
            return self._encode(video, acts)

    def step(self, state: State, action: Any) -> Transition:
        batch = self.batch_size(state)
        tensor = self.spec.action.as_tensor(action, batch, self.device)
        with torch.no_grad():
            return self._step(state, tensor)

    def decode(self, state: State) -> torch.Tensor:
        with torch.no_grad():
            return self._decode(state).clamp(0.0, 1.0)

    def rollout(self, state: State, actions: Any, *, decode: bool = True) -> Rollout:
        sequence = _to_tensor(actions, self.device)
        batch = self.batch_size(state)
        if self.spec.action.kind == "continuous" and sequence.dim() == 2:
            sequence = sequence.unsqueeze(0).expand(batch, -1, -1)
        elif self.spec.action.kind == "discrete" and sequence.dim() == 1:
            sequence = sequence.unsqueeze(0).expand(batch, -1)
        if sequence.shape[0] != batch:
            raise ValueError(
                f"actions hold {sequence.shape[0]} sequences for a batch of {batch}."
            )
        frames, rewards, terminated = [], [], []
        for t in range(sequence.shape[1]):
            out = self.step(state, sequence[:, t])
            state = out.state
            if decode:
                frames.append(self.decode(state))
            if out.reward is not None:
                rewards.append(out.reward)
            if out.terminated is not None:
                terminated.append(out.terminated)
        return Rollout(
            state=state,
            frames=torch.stack(frames, dim=1) if frames else None,
            rewards=torch.stack(rewards, dim=1) if rewards else None,
            terminated=torch.stack(terminated, dim=1) if terminated else None,
        )

    def save(self, path: str | Path) -> Path:
        """Write ``world_model.json`` and the weights into directory ``path``.

        Read it back with :func:`~simulusion.world_model.load_world_model`.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        files = self._save_weights(out)
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "family": self.family,
            "spec": self.spec.to_dict(),
            "files": files,
            "options": self._options(),
        }
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        **options: Any,
    ) -> "WorldModelBase":
        """Load a directory written by :meth:`save`, or the family's own checkpoint.

        Options saved in the manifest (such as a sampling temperature) are
        restored; keyword ``options`` override them.
        """
        target = Path(path)
        manifest = read_manifest(target)
        if manifest is not None:
            options = {**manifest.get("options", {}), **options}
        return cls._load_weights(target, manifest, _resolve_device(device), options)

    # ------------------------------------------------------------ for adapters

    @staticmethod
    def batch_size(state: State) -> int:
        for value in state.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("The state holds no tensors.")

    def as_frames(self, frames: Any) -> torch.Tensor:
        """Frames as float ``(B, T, C, H, W)`` in ``[0, 1]`` on :attr:`device`."""
        tensor = _to_tensor(frames, self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 5:
            raise ValueError(
                "Expected frames shaped (B, T, C, H, W), (T, C, H, W) or "
                f"(C, H, W); got {tuple(tensor.shape)}."
            )
        channels = self.spec.obs_shape[0]
        if tensor.shape[2] != channels and tensor.shape[-1] == channels:
            tensor = tensor.permute(0, 1, 4, 2, 3)  # channel-last input
        if tensor.shape[2] != channels:
            raise ValueError(
                f"Expected {channels}-channel frames, got shape {tuple(tensor.shape)}."
            )
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
        size = tuple(self.spec.obs_shape[1:])
        if tuple(tensor.shape[-2:]) != size:
            B, T = tensor.shape[:2]
            tensor = F.interpolate(
                tensor.flatten(0, 1), size=size, mode="bilinear", align_corners=False
            ).reshape(B, T, channels, *size)
        # Contiguous: several encoders flatten with .view().
        return tensor.clamp(0.0, 1.0).contiguous()

    def _context_actions(
        self, actions: Any, batch: int, frames: int
    ) -> Optional[torch.Tensor]:
        """The ``T - 1`` actions between context frames, shaped per action kind."""
        if actions is None:
            return None
        spec = self.spec.action
        tensor = _to_tensor(actions, self.device)
        if spec.kind == "discrete":
            tensor = tensor.long()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.reshape(tensor.shape[0], -1)
            tensor = tensor.expand(batch, -1) if tensor.shape[0] == 1 else tensor
        else:
            tensor = tensor.float()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.expand(batch, -1, -1) if tensor.shape[0] == 1 else tensor
        if tensor.shape[0] != batch or tensor.shape[1] != frames - 1:
            raise ValueError(
                f"Expected {frames - 1} actions between {frames} context frames "
                f"for a batch of {batch}; got shape {tuple(tensor.shape)}."
            )
        return tensor

    def _options(self) -> dict[str, Any]:
        """Constructor options worth restoring on load (JSON-serialisable)."""
        return {}

    @abstractmethod
    def _encode(self, frames: torch.Tensor, actions: Optional[torch.Tensor]) -> State:
        """``frames`` float (B, T, C, H, W) in [0, 1]; ``actions`` (B, T-1[, A])."""

    @abstractmethod
    def _step(self, state: State, action: torch.Tensor) -> Transition:
        """``action`` is ``(B,)`` int64 or ``(B, A)`` float."""

    @abstractmethod
    def _decode(self, state: State) -> torch.Tensor:
        """Float ``(B, C, H, W)``, roughly in [0, 1]; the base class clamps."""

    @abstractmethod
    def _save_weights(self, directory: Path) -> dict[str, str]:
        """Write the weights into ``directory``; return ``{role: filename}``."""

    @classmethod
    @abstractmethod
    def _load_weights(
        cls,
        path: Path,
        manifest: Optional[dict[str, Any]],
        device: torch.device,
        options: dict[str, Any],
    ) -> "WorldModelBase":
        """Rebuild from :meth:`save` output (``manifest`` set) or a native checkpoint."""

    def __repr__(self) -> str:
        spec = self.spec
        return (
            f"{type(self).__name__}(obs={spec.obs_shape}, "
            f"actions={spec.action.kind}[{spec.action.n}], device={self.device})"
        )


def read_manifest(path: str | Path) -> Optional[dict[str, Any]]:
    """The ``world_model.json`` in directory ``path``, or ``None``."""
    target = Path(path)
    manifest_path = target / MANIFEST_NAME if target.is_dir() else None
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{manifest_path} is not a Simulusion world model manifest.")
    if int(manifest.get("version", 0)) > MANIFEST_VERSION:
        raise ValueError(
            f"{manifest_path} was written by a newer Simulusion (manifest "
            f"v{manifest['version']}); upgrade to load it."
        )
    return manifest


def _resolve_device(device: str | torch.device | None) -> torch.device:
    from simulusion.utils.device import resolve_device

    return torch.device(resolve_device(device))


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    array = np.asarray(value)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return torch.as_tensor(array, device=device)


def module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    return torch.device("cpu")


__all__: Sequence[str] = [
    "ActionSpec",
    "MANIFEST_NAME",
    "Rollout",
    "State",
    "Transition",
    "WorldModel",
    "WorldModelBase",
    "WorldModelSpec",
    "read_manifest",
]
