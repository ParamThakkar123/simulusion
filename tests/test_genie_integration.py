import pytest
from simulusion.models.genie import Genie


@pytest.mark.integration
def test_genie_construct_no_crash():
    genie = Genie(
        num_frames=2,
        image_size=16,
        tokenizer_encoder_dim=32,
        tokenizer_decoder_dim=32,
        action_encoder_dim=32,
        action_decoder_dim=32,
        dynamics_dim=32,
        dynamics_depth=1,
        dynamics_num_heads=2,
        encoder_depth=1,
        decoder_depth=1,
        latent_action_depth=1,
    )
    assert genie is not None
