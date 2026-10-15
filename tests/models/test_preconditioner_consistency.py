import pytest

import torch

from simulusion.models.diffusion.diamond_diffusion import (
    EDMPreconditioner,
    DiffusionUNet,
    EulerSampler,
)

pytestmark = [pytest.mark.integration]


class DummyModel(DiffusionUNet):
    def __init__(self):
        super().__init__()
        self.last_t = None

    def forward(self, x, t, obs_history=None, actions=None):
        # record the t that was passed (to check consistency)
        self.last_t = t.detach().cpu()
        # return a zero tensor with expected output shape
        return torch.zeros_like(x)


def test_preconditioner_consistency():
    B = 3
    # create a preconditioner and use a fixed scalar sigma so sampling and
    # training paths are comparable (sampler uses scalar schedule values)
    pre = EDMPreconditioner()
    sigma_val = 0.5
    sigma = torch.full((B,), sigma_val)

    # compute preconditioners used during training path
    sigma_view = sigma.view(B, 1, 1, 1)
    preconds = pre.get_preconditioners(sigma_view)
    t_train = preconds["c_noise"].squeeze(-1).squeeze(-1)

    # instantiate dummy model and call pre.denoise (training-style path)
    model = DummyModel()
    x = torch.randn(B, 3, 64, 64)
    _ = pre.denoise(model, x, sigma, obs_history=None, actions=None)
    # model.last_t should match t_train
    assert model.last_t.shape == t_train.shape
    assert torch.allclose(model.last_t, t_train, atol=1e-6)

    # now test sampling path: use EulerSampler with injected preconditioner
    # configure a single-step sampler that will use the same scalar sigma_val
    sampler = EulerSampler(num_steps=1, edm_precond=pre)
    sampler.t_steps = torch.tensor([sigma_val])
    sampler.t_next = torch.tensor([0.0])

    model2 = DummyModel()
    sampler.sample(model2, (B, 3, 64, 64), device=torch.device("cpu"))
    # in the sampling path the model should have been called with c_noise too
    assert model2.last_t.shape == t_train.shape
    # match values
    assert torch.allclose(model2.last_t, t_train, atol=1e-6)

    print("preconditioner consistency test passed")


if __name__ == "__main__":
    test_preconditioner_consistency()
