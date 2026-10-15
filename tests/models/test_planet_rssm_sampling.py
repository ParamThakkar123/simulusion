"""The PlaNet RSSM must sample latents from N(mean, std), not a uniform."""

import torch

from simulusion.models.rssm import RecurrentStateSpaceModel


def _rssm():
    torch.manual_seed(0)
    return RecurrentStateSpaceModel(
        action_size=2, state_size=8, latent_size=4, hidden_size=8, embed_size=16
    )


def _standardised(samples, mean, std):
    return ((samples - mean) / std).flatten()


def test_state_prior_samples_are_standard_normal():
    rssm = _rssm()
    h = torch.randn(20000, 8)
    with torch.no_grad():
        mean, std = rssm.state_prior(h)
        z = _standardised(rssm.state_prior(h, sample=True), mean, std)
    assert abs(z.mean().item()) < 0.05
    assert abs(z.std().item() - 1.0) < 0.05
    assert (z < 0).float().mean().item() > 0.4


def test_state_posterior_samples_are_standard_normal():
    rssm = _rssm()
    h = torch.randn(20000, 8)
    e = torch.randn(20000, 16)
    with torch.no_grad():
        mean, std = rssm.state_posterior(h, e)
        z = _standardised(rssm.state_posterior(h, e, sample=True), mean, std)
    assert abs(z.mean().item()) < 0.05
    assert abs(z.std().item() - 1.0) < 0.05


def test_rollout_prior_returns_stacked_tensors():
    rssm = _rssm()
    actions = torch.randn(5, 3, 2)
    with torch.no_grad():
        states, latents = rssm.rollout_prior(
            actions, torch.zeros(3, 8), torch.zeros(3, 4)
        )
    assert states.shape == (5, 3, 8)
    assert latents.shape == (5, 3, 4)


def test_get_init_state_posterior_uses_updated_hidden_state():
    rssm = _rssm()
    enc = torch.randn(3, 16)
    h, s, a = torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 2)
    with torch.no_grad():
        h_next, s_next = rssm.get_init_state(enc, h, s, a)
        expected, _ = rssm.state_posterior(rssm.deterministic_state_fwd(h, s, a), enc)
    assert torch.allclose(s_next, expected)


def test_get_init_state_defaults_to_posterior_mean_and_can_sample():
    rssm = _rssm()
    enc = torch.randn(3, 16)
    with torch.no_grad():
        _, mean_a = rssm.get_init_state(enc)
        _, mean_b = rssm.get_init_state(enc)
        _, sampled = rssm.get_init_state(enc, sample=True)
    assert torch.equal(mean_a, mean_b)
    assert not torch.equal(mean_a, sampled)
