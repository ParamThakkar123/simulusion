"""The ST-transformer temporal KV cache must reproduce full-prefix logits."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simulusion.models.dynamics_model import DynamicsModel  # noqa: E402


def _model() -> DynamicsModel:
    torch.manual_seed(0)
    return DynamicsModel(
        num_frames=6,
        image_size=16,
        vocab_size=20,
        action_vocab_size=5,
        dim=32,
        depth=2,
        num_heads=4,
        patch_size=4,
        gradient_checkpointing=False,
    ).eval()


def _data(batch: int = 2, frames: int = 5):
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 20, (batch, frames, 16), generator=gen)
    actions = torch.randint(0, 5, (batch, frames), generator=gen)
    return tokens, actions


def test_cached_logits_match_full_forward():
    model = _model()
    tokens, actions = _data()
    with torch.no_grad():
        full = model(tokens, actions, mask_prob=0.0)
        cache = model.init_cache(tokens.shape[0])
        prompt = model.forward_cached(tokens[:, :2], actions[:, :2], cache)
        steps = [
            model.forward_cached(tokens[:, t : t + 1], actions[:, t : t + 1], cache)
            for t in range(2, tokens.shape[1])
        ]
    cached = torch.cat([prompt, *steps], dim=1)
    assert cache.length == tokens.shape[1]
    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


def test_uncommitted_forward_can_be_repeated():
    model = _model()
    tokens, actions = _data()
    with torch.no_grad():
        cache = model.init_cache(2)
        model.forward_cached(tokens[:, :2], actions[:, :2], cache)
        # A candidate frame that is thrown away must not disturb the cache.
        model.forward_cached(
            torch.zeros_like(tokens[:, 2:3]), actions[:, 2:3], cache, commit=False
        )
        assert cache.length == 2
        step = model.forward_cached(tokens[:, 2:3], actions[:, 2:3], cache)
        full = model(tokens[:, :3], actions[:, :3], mask_prob=0.0)
    torch.testing.assert_close(step[:, 0], full[:, 2], atol=1e-5, rtol=1e-5)


def test_multi_frame_step_after_prompt_is_rejected():
    model = _model()
    tokens, actions = _data()
    with torch.no_grad():
        cache = model.init_cache(2)
        model.forward_cached(tokens[:, :2], actions[:, :2], cache)
        with pytest.raises(ValueError, match="one frame"):
            model.forward_cached(tokens[:, 2:4], actions[:, 2:4], cache)


def test_cache_overflow_is_rejected():
    model = _model()
    tokens, actions = _data(frames=5)
    with torch.no_grad():
        cache = model.init_cache(2)
        model.forward_cached(tokens, actions, cache)
        model.forward_cached(tokens[:, :1], actions[:, :1], cache)
        with pytest.raises(ValueError):
            model.forward_cached(tokens[:, :1], actions[:, :1], cache)


def test_autoregressive_sample_cached_matches_uncached():
    model = _model()
    tokens, actions = _data()
    prompt = tokens[:, :1]
    with torch.no_grad():
        torch.manual_seed(123)
        uncached = model.autoregressive_sample(prompt, actions, num_frames=5)
        torch.manual_seed(123)
        cached = model.autoregressive_sample(
            prompt, actions, num_frames=5, use_cache=True
        )
    assert cached.shape == (2, 5, 16)
    assert torch.equal(cached, uncached)
