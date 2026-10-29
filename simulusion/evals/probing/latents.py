"""What each world model holds about every frame of a probe dataset.

:func:`extract_latents` runs a :class:`~simulusion.world_model.WorldModel` over
the episodes of a :class:`~simulusion.evals.probing.ProbeDataset` and keeps,
for every frame, the model's state as flat feature vectors: one array per
part of the state (Dreamer's ``deter`` and ``stoch``, World Models' ``z`` and
``h``, ...), each row aligned with the dataset's :class:`StateTargets`.

.. code-block:: python

    wm = load_world_model("runs/dreamer-cartpole")
    latents = extract_latents(wm, data)
    latents.features["deter"]              # (N, 200)
    targets = latents.match(data.targets())

Two controls go through the same pipeline: :func:`pixel_features` (the frames
themselves, downsampled) and :func:`reinitialized` (the same architecture with
fresh weights), so a probe's score can be read against what the input and an
untrained network already give.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from simulusion.evals.probing.data import ProbeDataset, StateTargets

State = Dict[str, torch.Tensor]
FeatureFn = Callable[[Any, State], Dict[str, torch.Tensor]]

# Inputs echoed back in a state, not something the model inferred.
_INPUT_KEYS = frozenset({"actions", "prev_actions"})

# The state parts each family's own heads read, offered together as "features".
_COMBINED = {
    "dreamer": ("deter", "stoch"),
    "planet": ("deter", "stoch"),
    "world-models": ("z", "h"),
}


@dataclass(frozen=True)
class Latents:
    """Flat latent features for frames of a probe dataset.

    Attributes:
        features: Feature name to ``(N, D)`` float32 array.
        episode: ``(N,)`` episode index of each row.
        step: ``(N,)`` frame index of each row within its episode.
        source: What produced them: a family name, ``"pixels"``, ...
        info: How they were made (context, whether actions were used).
    """

    features: Dict[str, np.ndarray]
    episode: np.ndarray
    step: np.ndarray
    source: str
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.episode.shape[0])

    def match(self, targets: StateTargets) -> StateTargets:
        """The rows of ``targets`` for these frames, in this order.

        A model that needs several context frames has no latent for the first
        few frames of each episode; their targets are dropped.
        """
        width = int(max(self.step.max(initial=0), targets.step.max(initial=0))) + 1
        keys = targets.episode * width + targets.step
        order = np.argsort(keys, kind="stable")
        wanted = self.episode * width + self.step
        found = np.searchsorted(keys, wanted, sorter=order)
        found = np.clip(found, 0, len(keys) - 1)
        rows = order[found]
        if len(keys) == 0 or not np.array_equal(keys[rows], wanted):
            raise ValueError(
                "Some latent rows have no target: were the latents and targets "
                "made from the same dataset?"
            )
        return StateTargets(
            targets.values[rows],
            targets.names,
            targets.episode[rows],
            targets.step[rows],
        )


def latent_features(model: Any, state: State) -> Dict[str, torch.Tensor]:
    """A world-model state as named ``(B, D)`` float feature vectors.

    Token states (IRIS, Genie) are looked up in the tokenizer's codebook for the
    latest frame: ``"tokens"`` keeps every position, ``"tokens_pooled"`` averages
    them. DIAMOND's ``obs_window`` is pixels, so only its latest frame is kept,
    as ``"pixels"``. Families listed in ``_COMBINED`` also get ``"features"``,
    the concatenation their own heads read.
    """
    family = getattr(model, "family", None) or model.spec.family
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in _INPUT_KEYS:
            continue
        if key == "tokens":
            out.update(_token_features(model, family, value))
        elif key == "obs_window":
            out["pixels"] = value[:, -1].flatten(1).float()
        else:
            out[key] = value.flatten(1).float()
    parts = _COMBINED.get(family, ())
    if parts and all(p in out for p in parts):
        out["features"] = torch.cat([out[p] for p in parts], dim=1)
    return out


def _token_features(
    model: Any, family: str, tokens: torch.Tensor
) -> Dict[str, torch.Tensor]:
    last = tokens[:, -1]  # (B, K): the latest frame's token ids
    if family == "iris":
        grid = last.reshape(last.shape[0], model.grid, model.grid)
        embedded = model.agent.encoder.quantizer.decode_indices(grid)  # (B, D, g, g)
        positions = embedded.flatten(2).transpose(1, 2)  # (B, K, D)
    elif family == "genie":
        embedded = model.model.video_tokenizer.decode_indices(last.unsqueeze(1))
        positions = embedded.reshape(last.shape[0], last.shape[1], -1)  # (B, K, D)
    else:
        raise ValueError(f"No codebook lookup for {family!r} tokens.")
    positions = positions.float()
    return {"tokens": positions.flatten(1), "tokens_pooled": positions.mean(dim=1)}


def extract_latents(
    model: Any,
    data: ProbeDataset,
    *,
    context: Optional[int] = None,
    actions: bool = True,
    action_fn: Optional[Callable[[np.ndarray], Any]] = None,
    keys: Optional[Iterable[str]] = None,
    chunk: int = 256,
    features: FeatureFn = latent_features,
) -> Latents:
    """Encode every frame of ``data`` and flatten the state into features.

    Args:
        model: A :class:`~simulusion.world_model.WorldModel`.
        data: The probe dataset; frames are resized to the model's input.
        context: Frames of history behind each latent; ``None`` uses all of
            them (recurrent families) or the model's window. ``context=1``
            asks what a single frame gives, which cannot hold velocity.
        actions: Pass the recorded actions between frames to the model.
        action_fn: Map an episode's ``(T - 1, A)`` DMC actions to the model's
            actions, for models trained with another action space (IRIS, Genie
            and DIAMOND take discrete actions). Without it a discrete-action
            model encodes without actions.
        keys: Keep only these feature names.
        chunk: Windows encoded per batch, for families without a one-pass encode.
        features: State to features; :func:`latent_features` by default.
    """
    spec = model.spec
    wanted = None if keys is None else set(keys)
    use_actions = actions
    if actions and action_fn is None and spec.action.kind == "discrete":
        warnings.warn(
            f"{spec.family} takes discrete actions and no action_fn maps DMC's "
            "continuous ones; encoding without actions.",
            stacklevel=2,
        )
        use_actions = False

    columns: Dict[str, list[np.ndarray]] = {}
    episode_rows, step_rows = [], []
    for index, episode in enumerate(data):
        acts: Optional[torch.Tensor] = None
        if use_actions and episode.actions is not None and len(episode.actions):
            raw = action_fn(episode.actions) if action_fn else episode.actions
            acts = _batched(raw)
            if spec.action.kind == "continuous" and acts.shape[-1] != spec.action.n:
                raise ValueError(
                    f"{data.task} has {acts.shape[-1]}-d actions but {spec.family} "
                    f"takes {spec.action.n}; pass action_fn or actions=False."
                )
        frames = episode.frames[None]
        states = _encode_steps(model, frames, acts, context, chunk)
        steps = [t for t, state in enumerate(states) if state is not None]
        for name, value in _features_by_shape(model, states, steps, features).items():
            if wanted is None or name in wanted:
                columns.setdefault(name, []).append(value)
        episode_rows.append(np.full(len(steps), index, dtype=np.int64))
        step_rows.append(np.asarray(steps, dtype=np.int64))

    if wanted is not None and not wanted <= set(columns):
        raise ValueError(
            f"No features named {sorted(wanted - set(columns))}; "
            f"this model gives {sorted(columns)}."
        )
    return Latents(
        features={name: np.concatenate(parts) for name, parts in columns.items()},
        episode=np.concatenate(episode_rows),
        step=np.concatenate(step_rows),
        source=spec.family,
        info={"context": context, "actions": use_actions, "task": data.task},
    )


def _batched(actions: Any) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(actions))
    return tensor.unsqueeze(0)


def _encode_steps(
    model: Any,
    frames: np.ndarray,
    actions: Optional[torch.Tensor],
    context: Optional[int],
    chunk: int,
) -> list:
    encode_steps = getattr(model, "encode_steps", None)
    if encode_steps is not None:
        return encode_steps(frames, actions, context=context, chunk=chunk)
    # A WorldModel without the convenience: one encode per frame.
    T = frames.shape[1]
    need = model.spec.context_frames
    window = context or model.spec.max_context or T
    states: list = [None] * T
    for t in range(need - 1, T):
        start = max(0, t + 1 - window)
        acts = None if actions is None else actions[:, start:t]
        states[t] = model.encode(frames[:, start : t + 1], acts)
    return states


def _features_by_shape(
    model: Any, states: list, steps: list[int], features: FeatureFn
) -> Dict[str, np.ndarray]:
    """Features for ``states[steps]``, batching states of equal shape."""
    groups: Dict[tuple, list[int]] = {}
    for position, t in enumerate(steps):
        signature = tuple((k, tuple(v.shape)) for k, v in sorted(states[t].items()))
        groups.setdefault(signature, []).append(position)
    out: Dict[str, np.ndarray] = {}
    for positions in groups.values():
        batch = {
            key: torch.cat([states[steps[p]][key] for p in positions])
            for key in states[steps[positions[0]]]
        }
        with torch.no_grad():
            values = features(model, batch)
        for name, value in values.items():
            array = out.get(name)
            if array is None:
                array = out[name] = np.empty(
                    (len(steps), value.shape[1]), dtype=np.float32
                )
            array[positions] = value.detach().cpu().numpy()
    return out


def pixel_features(data: ProbeDataset, size: int = 16) -> Latents:
    """The frames themselves, area-downsampled to ``size`` x ``size``.

    The floor every learned latent should clear: a linear probe on pixels
    shows what is decodable from the image with no model at all.
    """
    rows, episode_rows, step_rows = [], [], []
    for index, episode in enumerate(data):
        frames = torch.from_numpy(np.asarray(episode.frames)).permute(0, 3, 1, 2)
        small = F.interpolate(frames.float() / 255.0, size=(size, size), mode="area")
        rows.append(small.flatten(1).numpy())
        episode_rows.append(np.full(len(frames), index, dtype=np.int64))
        step_rows.append(np.arange(len(frames), dtype=np.int64))
    return Latents(
        features={"pixels": np.concatenate(rows).astype(np.float32)},
        episode=np.concatenate(episode_rows),
        step=np.concatenate(step_rows),
        source="pixels",
        info={"size": size, "task": data.task},
    )


def reinitialized(model: Any, seed: int = 0) -> Any:
    """A copy of ``model`` with every layer's weights drawn afresh.

    The untrained-network control: a random encoder already preserves a lot of
    what is in the image, so a probe has to beat this to credit training.
    """
    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    seen: set[int] = set()
    for module in _modules(clone, depth=3):
        for sub in module.modules():
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            reset = getattr(sub, "reset_parameters", None)
            if callable(reset):
                reset()
    return clone


def _modules(obj: Any, depth: int) -> Iterable[torch.nn.Module]:
    if isinstance(obj, torch.nn.Module):
        yield obj
        return
    if depth == 0 or not hasattr(obj, "__dict__"):
        return
    for value in vars(obj).values():
        yield from _modules(value, depth - 1)
