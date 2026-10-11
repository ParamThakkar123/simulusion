"""Tests for the linear Controller model (Ha & Schmidhuber 2018)."""

import torch
import pytest
from simulusion.models.controller import Controller


class TestController:
    @pytest.fixture
    def controller(self):
        return Controller(latent_size=32, hidden_size=256, action_size=3)

    def test_forward_shape(self, controller):
        bs = 4
        h = torch.randn(bs, 256)
        z = torch.randn(bs, 32)
        state = torch.cat([z, h], dim=-1)
        actions = controller(state)
        assert actions.shape == (bs, 3)

    def test_forward_with_separate_inputs(self, controller):
        bs = 4
        h = torch.randn(bs, 256)
        z = torch.randn(bs, 32)
        state = torch.cat([z, h], dim=-1)
        actions = controller(state)
        assert actions.shape == (bs, 3)

    def test_differentiable(self, controller):
        bs = 2
        h = torch.randn(bs, 256)
        z = torch.randn(bs, 32)
        state = torch.cat([z, h], dim=-1)
        actions = controller(state)
        loss = actions.sum()
        loss.backward()
        for name, param in controller.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_init_attributes(self, controller):
        assert controller.latent_size == 32
        assert controller.hidden_size == 256
        assert controller.action_size == 3

    def test_different_configurations(self):
        configs = [
            (16, 128, 2),
            (32, 256, 5),
            (64, 512, 10),
        ]
        for latent_size, hidden_size, action_size in configs:
            ctrl = Controller(latent_size, hidden_size, action_size)
            bs = 2
            state = torch.cat(
                [torch.randn(bs, latent_size), torch.randn(bs, hidden_size)], dim=-1
            )
            actions = ctrl(state)
            assert actions.shape == (bs, action_size)
