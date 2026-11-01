import pytest
import torch
from simulusion.models.rssm import RecurrentStateSpaceModel


class TestRecurrentStateSpaceModel:
    @pytest.fixture
    def rssm(self):
        model = RecurrentStateSpaceModel(
            action_size=2,
            state_size=50,
            latent_size=10,
            hidden_size=50,
            embed_size=256,
            activation_function="relu",
        )
        return model

    def test_init(self, rssm):
        assert rssm.state_size == 50
        assert rssm.action_size == 2
        assert rssm.latent_size == 10

    def test_get_init_state(self, rssm):
        enc = torch.randn(4, 256)

        h_t, s_t = rssm.get_init_state(enc)

        assert h_t.shape == (4, 50)
        assert s_t.shape == (4, 10)

    def test_get_init_state_with_existing_state(self, rssm):
        enc = torch.randn(4, 256)
        h_existing = torch.randn(4, 50)
        s_existing = torch.randn(4, 10)

        h_t, s_t = rssm.get_init_state(enc, h_t=h_existing, s_t=s_existing)

        assert h_t.shape == (4, 50)
        assert s_t.shape == (4, 10)

    def test_get_init_state_mean(self, rssm):
        enc = torch.randn(4, 256)

        h_t, s_t = rssm.get_init_state(enc, mean=True)

        assert h_t.shape == (4, 50)
        assert s_t.shape == (4, 10)

    def test_deterministic_state_fwd(self, rssm):
        h_t = torch.randn(4, 50)
        s_t = torch.randn(4, 10)
        a_t = torch.randn(4, 2)

        h_new = rssm.deterministic_state_fwd(h_t, s_t, a_t)

        assert h_new.shape == (4, 50)

    def test_deterministic_state_fwd_unbatched(self, rssm):
        h_t = torch.randn(4, 50)
        s_t = torch.randn(4, 10)
        a_t = torch.randn(2)

        h_new = rssm.deterministic_state_fwd(h_t, s_t, a_t)

        assert h_new.shape == (4, 50)

    def test_deterministic_state_fwd_batch_mismatch_expand(self, rssm):
        h_t = torch.randn(4, 50)
        s_t = torch.randn(4, 10)
        a_t = torch.randn(1, 2)

        h_new = rssm.deterministic_state_fwd(h_t, s_t, a_t)

        assert h_new.shape == (4, 50)

    def test_state_prior(self, rssm):
        h_t = torch.randn(4, 50)

        result = rssm.state_prior(h_t)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (4, 10)
        assert result[1].shape == (4, 10)

    def test_state_prior_sample(self, rssm):
        h_t = torch.randn(4, 50)

        s_t = rssm.state_prior(h_t, sample=True)

        assert s_t.shape == (4, 10)

    def test_state_posterior(self, rssm):
        h_t = torch.randn(4, 50)
        e_t = torch.randn(4, 256)

        result = rssm.state_posterior(h_t, e_t)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (4, 10)
        assert result[1].shape == (4, 10)

    def test_state_posterior_sample(self, rssm):
        h_t = torch.randn(4, 50)
        e_t = torch.randn(4, 256)

        s_t = rssm.state_posterior(h_t, e_t, sample=True)

        assert s_t.shape == (4, 10)

    def test_pred_reward(self, rssm):
        h_t = torch.randn(4, 50)
        s_t = torch.randn(4, 10)

        r = rssm.pred_reward(h_t, s_t)

        assert r.shape == (4,)

    def test_forward(self, rssm):
        B = 2
        T = 3
        x = torch.randn(B, T + 1, 3, 64, 64)
        u = torch.randn(B, T, 2)

        states, priors, posteriors = rssm.forward(x, u)

        assert len(states) == T
        assert len(priors) == T
        assert len(posteriors) == T
        assert states[0].shape == (B, 50)

    def test_forward_single_step(self, rssm):
        B = 2
        T = 1
        x = torch.randn(B, T + 1, 3, 64, 64)
        u = torch.randn(B, T, 2)

        states, priors, posteriors = rssm.forward(x, u)

        assert len(states) == 1
        assert states[0].shape == (B, 50)
