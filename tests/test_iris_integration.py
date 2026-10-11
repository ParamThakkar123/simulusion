import pytest
from simulusion.configs.iris_config import IRISConfig


@pytest.mark.integration
def test_iris_config_construct():
    config = IRISConfig()
    assert config.vocab_size == 512
    assert config.env == "ALE/Pong-v5"
    assert config.get_frame_shape() == (3, 64, 64)
    assert "vocab_size" in config.get_autoencoder_config()
    assert "embed_dim" in config.get_transformer_config()
    assert "imagination_horizon" in config.get_rl_config()
