"""Genie head counts must divide widths, and configs must reach every component."""

import inspect

import pytest

from simulusion import create_model
from simulusion.blocks.st_transformer import STSpatialAttention, STTemporalAttention
from simulusion.configs.genie_config import DynamicsModelConfig, GenieConfig
from simulusion.models.dynamics_model import DynamicsModel, create_dynamics_model
from simulusion.models.genie import (
    Genie,
    create_genie,
    create_genie_large,
    genie_kwargs_from_config,
)


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


@pytest.mark.parametrize("fn", [Genie, create_genie])
def test_genie_default_dynamics_heads_divide_width(fn):
    defaults = _defaults(fn)
    assert defaults["dynamics_dim"] % defaults["dynamics_num_heads"] == 0


@pytest.mark.parametrize("fn", [DynamicsModel, create_dynamics_model])
def test_dynamics_default_heads_divide_width(fn):
    defaults = _defaults(fn)
    assert defaults["dim"] % defaults["num_heads"] == 0


def test_dynamics_config_heads_divide_width():
    config = DynamicsModelConfig()
    assert config.dim % config.num_heads == 0


def test_genie_large_heads_divide_width(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "simulusion.models.genie.Genie.__init__",
        lambda self, **kwargs: captured.update(kwargs),
    )
    create_genie_large()
    assert captured["dynamics_dim"] % captured["dynamics_num_heads"] == 0


@pytest.mark.parametrize("cls", [STSpatialAttention, STTemporalAttention])
def test_attention_rejects_indivisible_heads(cls):
    with pytest.raises(ValueError, match="divisible"):
        cls(dim=5120, num_heads=36)


def test_config_mapping_covers_renamed_fields():
    config = GenieConfig(
        tokenizer_encoder_dim=64,
        tokenizer_decoder_dim=96,
        tokenizer_encoder_depth=2,
        tokenizer_decoder_depth=3,
        action_encoder_dim=48,
        action_decoder_dim=80,
        action_encoder_depth=5,
    )
    kwargs = genie_kwargs_from_config(config)
    assert kwargs["tokenizer_encoder_dim"] == 64
    assert kwargs["tokenizer_decoder_dim"] == 96
    assert kwargs["encoder_depth"] == 2
    assert kwargs["decoder_depth"] == 3
    assert kwargs["action_encoder_dim"] == 48
    assert kwargs["action_decoder_dim"] == 80
    assert kwargs["latent_action_depth"] == 5
    assert set(kwargs) <= set(inspect.signature(Genie).parameters)


def test_create_model_genie_uses_config_widths_and_depths(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "simulusion.models.genie.Genie.__init__",
        lambda self, **kwargs: captured.update(kwargs),
    )
    create_model("genie", use_bfloat16=True)
    config = GenieConfig()
    assert captured["tokenizer_encoder_dim"] == config.tokenizer_encoder_dim
    assert captured["tokenizer_decoder_dim"] == config.tokenizer_decoder_dim
    assert captured["encoder_depth"] == config.tokenizer_encoder_depth
    assert captured["decoder_depth"] == config.tokenizer_decoder_depth
    assert captured["action_encoder_dim"] == config.action_encoder_dim
    assert captured["latent_action_depth"] == config.action_encoder_depth
    assert captured["use_bfloat16"] is True


def test_head_count_fields_reach_the_components():
    config = GenieConfig(
        num_frames=2,
        image_size=16,
        tokenizer_encoder_dim=32,
        tokenizer_decoder_dim=32,
        tokenizer_encoder_depth=1,
        tokenizer_decoder_depth=1,
        tokenizer_num_heads=4,
        action_encoder_dim=32,
        action_decoder_dim=32,
        action_encoder_depth=1,
        action_num_heads=2,
        dynamics_dim=32,
        dynamics_depth=1,
        dynamics_num_heads=4,
    )
    model = Genie.from_config(config)
    tokenizer_heads = {
        m.num_heads
        for m in model.video_tokenizer.modules()
        if isinstance(m, (STSpatialAttention, STTemporalAttention))
    }
    action_heads = {
        m.num_heads
        for m in model.latent_action_model.modules()
        if isinstance(m, (STSpatialAttention, STTemporalAttention))
    }
    assert tokenizer_heads == {4}
    assert action_heads == {2}
    assert model.config.action_decoder_dim == 32


def test_config_head_defaults_match_what_genie_builds():
    defaults = _defaults(Genie)
    config = GenieConfig()
    assert config.tokenizer_num_heads == defaults["tokenizer_num_heads"]
    assert config.action_num_heads == defaults["action_num_heads"]
