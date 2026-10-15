from __future__ import annotations

from typing import Any

import numpy as np

from simulusion.utils.gym_compat import gym


_IMAGE_KEYS = {"image", "pixels", "rgb"}
_PREFERRED_STATE_KEYS = {"state", "observation", "obs"}


def _looks_like_image_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) >= 2 and (
        shape[-1] in (1, 3, 4) or (len(shape) == 3 and shape[0] in (1, 3, 4))
    )


def flatten_box_space(space: Any) -> gym.spaces.Box | None:
    if not all(hasattr(space, attr) for attr in ("shape", "low", "high")):
        return None
    shape = tuple(int(dim) for dim in tuple(space.shape))
    if _looks_like_image_shape(shape):
        return None
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def iter_vector_spaces(space: Any) -> list[gym.spaces.Box]:
    flattened = flatten_box_space(space)
    if flattened is not None:
        return [flattened]
    if hasattr(space, "spaces"):
        preferred = []
        remaining = []
        for key, subspace in dict(space.spaces).items():
            if key in _IMAGE_KEYS:
                continue
            flattened_subspace = flatten_box_space(subspace)
            if flattened_subspace is None:
                continue
            if key in _PREFERRED_STATE_KEYS:
                preferred.append(flattened_subspace)
            else:
                remaining.append(flattened_subspace)
        return preferred + remaining
    return []


def infer_state_space_from_observation_space(space: Any) -> gym.spaces.Box | None:
    vector_spaces = iter_vector_spaces(space)
    if not vector_spaces:
        return None
    low = np.concatenate(
        [subspace.low.reshape(-1) for subspace in vector_spaces]
    ).astype(np.float32)
    high = np.concatenate(
        [subspace.high.reshape(-1) for subspace in vector_spaces]
    ).astype(np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def candidate_vector_arrays(obs: Any) -> list[np.ndarray]:
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        preferred = []
        remaining = []
        for key, value in obs.items():
            if key in _IMAGE_KEYS:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0 or _looks_like_image_shape(tuple(arr.shape)):
                continue
            if key in _PREFERRED_STATE_KEYS:
                preferred.append(arr)
            else:
                remaining.append(arr)
        return preferred + remaining
    arr = np.asarray(obs)
    if arr.ndim == 0 or _looks_like_image_shape(tuple(arr.shape)):
        return []
    return [arr]


def flatten_vector_observation(obs: Any) -> np.ndarray | None:
    vectors = candidate_vector_arrays(obs)
    if not vectors:
        return None
    flattened = [np.asarray(vec, dtype=np.float32).reshape(-1) for vec in vectors]
    return np.concatenate(flattened, axis=0)


def add_optional_state_space(
    image_space: gym.spaces.Box,
    *,
    state_space: gym.spaces.Box | None,
) -> gym.spaces.Dict:
    spaces: dict[str, gym.spaces.Space[Any]] = {"image": image_space}
    if state_space is not None:
        spaces["state"] = state_space
    return gym.spaces.Dict(spaces)


__all__ = [
    "add_optional_state_space",
    "candidate_vector_arrays",
    "flatten_box_space",
    "flatten_vector_observation",
    "infer_state_space_from_observation_space",
    "iter_vector_spaces",
]
