"""DreamerV3's reference weight initialization (trunc_normal_in + outscales)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from simulusion.layers.block_gru import BlockLinear
from simulusion.layers.initializers import (
    dreamer_v3_init_,
    scale_output_,
    trunc_normal_in_,
)


def test_trunc_normal_in_has_unit_variance_per_fan_in():
    torch.manual_seed(0)
    w = trunc_normal_in_(torch.empty(400, 900), fan_in=900)
    assert abs(float(w.std()) - 1 / math.sqrt(900)) < 0.02 / math.sqrt(900)
    # Truncated at two standard deviations of the unit normal, before rescaling.
    assert float(w.abs().max()) <= 2 * 1.1368 / math.sqrt(900) + 1e-6


@pytest.mark.parametrize(
    ("layer", "fan_in"),
    [
        (nn.Linear(64, 32), 64),
        (nn.Conv2d(8, 16, 4, stride=2), 8 * 16),
        (nn.ConvTranspose2d(16, 8, 4, stride=2), 16 * 16),
        (BlockLinear(64, 96, blocks=4), 64),
    ],
)
def test_each_layer_uses_the_reference_fan_in(layer, fan_in):
    torch.manual_seed(0)
    layer.bias.data.fill_(1.0) if layer.bias is not None else None
    dreamer_v3_init_(nn.Sequential(layer))
    std = float(layer.weight.std())
    assert abs(std - 1 / math.sqrt(fan_in)) < 0.15 / math.sqrt(fan_in)
    if layer.bias is not None:
        assert float(layer.bias.abs().max()) == 0.0


def test_output_scale():
    torch.manual_seed(0)
    layer = nn.Linear(256, 10)
    scale_output_(layer, 0.01)
    assert abs(float(layer.weight.std()) - 0.01 / 16) < 0.2 * 0.01 / 16
    scale_output_(layer, 0.0)
    assert float(layer.weight.abs().max()) == 0.0


def test_dreamer_v3_starts_with_a_near_uniform_policy_and_zero_heads():
    from simulusion.configs.dreamer_v3_config import DreamerV3Config
    from simulusion.models.dreamer_v3 import DreamerV3

    cfg = DreamerV3Config(
        model_size="12m",
        env="catch/0",
        env_backend="bsuite",
        bsuite_obs_type="state",
        buffer_size=1000,
        actor_dist="onehot",
    )
    model = DreamerV3(cfg, obs_shape={"state": (50,)}, action_size=3, device="cpu")
    assert model.discrete_actions
    first = model.actor.model[0]
    assert isinstance(first, nn.Linear)
    feats = torch.randn(64, first.in_features)
    probs = model.actor.dist(feats).probs  # type: ignore[attr-defined]
    assert float((probs - 1 / 3).abs().max()) < 0.02
    for head in (model.reward_model, model.value_model):
        output = head.model[-1]
        assert isinstance(output, nn.Linear)
        assert float(output.weight.abs().max()) == 0.0
    for a, b in zip(model.value_model.parameters(), model.slow_value_model.parameters()):
        assert torch.equal(a, b)
