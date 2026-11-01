"""Tests for MDRNN and MDRNNCell models and gmm_loss."""

import torch
import pytest
from simulusion.models.mdrnn import MDRNN, MDRNNCell, _MDRNNBase


class TestMDRNN:
    @pytest.fixture
    def model(self):
        return MDRNN(latents=32, actions=3, hiddens=256, gaussians=5)

    def test_forward_shapes(self, model):
        seq_len, bs = 10, 4
        actions = torch.randn(seq_len, bs, 3)
        latents = torch.randn(seq_len, bs, 32)
        mus, sigmas, logpi, rs, ds = model(actions, latents)
        assert mus.shape == (seq_len, bs, 5, 32)
        assert sigmas.shape == (seq_len, bs, 5, 32)
        assert logpi.shape == (seq_len, bs, 5)
        assert rs.shape == (seq_len, bs)
        assert ds.shape == (seq_len, bs)

    def test_sigmas_positive(self, model):
        seq_len, bs = 10, 4
        actions = torch.randn(seq_len, bs, 3)
        latents = torch.randn(seq_len, bs, 32)
        _, sigmas, _, _, _ = model(actions, latents)
        assert (sigmas > 0).all()

    def test_logpi_valid_log_softmax(self, model):
        seq_len, bs = 10, 4
        actions = torch.randn(seq_len, bs, 3)
        latents = torch.randn(seq_len, bs, 32)
        _, _, logpi, _, _ = model(actions, latents)
        assert torch.allclose(logpi.exp().sum(dim=-1), torch.ones(seq_len, bs))

    def test_differentiable(self, model):
        seq_len, bs = 10, 2
        actions = torch.randn(seq_len, bs, 3)
        latents = torch.randn(seq_len, bs, 32)
        mus, sigmas, logpi, rs, ds = model(actions, latents)
        loss = mus.sum() + sigmas.sum() + logpi.sum() + rs.sum() + ds.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_get_init_hidden_shape(self, model):
        h, c = model.get_init_hidden(4)
        assert h.shape == (4, 256)
        assert c.shape == (4, 256)

    def test_batch_independence(self, model):
        seq_len, bs = 10, 2
        actions = torch.randn(seq_len, bs, 3)
        latents_a = torch.randn(seq_len, bs, 32)
        latents_b = latents_a.clone()
        latents_b[:, 1, :] = torch.randn(seq_len, 32)
        mus_a, _, _, _, _ = model(actions, latents_a)
        mus_b, _, _, _, _ = model(actions, latents_b)
        assert torch.allclose(mus_a[:, 0, :, :], mus_b[:, 0, :, :])


class TestMDRNNCell:
    @pytest.fixture
    def cell(self):
        return MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)

    def test_forward_shapes(self, cell):
        bs = 4
        action = torch.randn(bs, 3)
        latent = torch.randn(bs, 32)
        hidden = cell.get_init_hidden(bs)
        mus, sigmas, logpi, r, d, next_hidden = cell(action, latent, hidden)
        assert mus.shape == (bs, 5, 32)
        assert sigmas.shape == (bs, 5, 32)
        assert logpi.shape == (bs, 5)
        assert r.shape == (bs,)
        assert d.shape == (bs,)
        assert next_hidden[0].shape == (bs, 256)
        assert next_hidden[1].shape == (bs, 256)

    def test_sigmas_positive(self, cell):
        bs = 4
        action = torch.randn(bs, 3)
        latent = torch.randn(bs, 32)
        hidden = cell.get_init_hidden(bs)
        _, sigmas, _, _, _, _ = cell(action, latent, hidden)
        assert (sigmas > 0).all()

    def test_logpi_valid_log_softmax(self, cell):
        bs = 4
        action = torch.randn(bs, 3)
        latent = torch.randn(bs, 32)
        hidden = cell.get_init_hidden(bs)
        _, _, logpi, _, _, _ = cell(action, latent, hidden)
        assert torch.allclose(logpi.exp().sum(dim=-1), torch.ones(bs))

    def test_differentiable(self, cell):
        bs = 2
        action = torch.randn(bs, 3)
        latent = torch.randn(bs, 32)
        hidden = cell.get_init_hidden(bs)
        mus, sigmas, logpi, r, d, _ = cell(action, latent, hidden)
        loss = mus.sum() + sigmas.sum() + logpi.sum() + r.sum() + d.sum()
        loss.backward()
        for name, param in cell.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_get_init_hidden_shape(self, cell):
        h, c = cell.get_init_hidden(4)
        assert h.shape == (4, 256)
        assert c.shape == (4, 256)

    def test_hidden_state_updates(self, cell):
        bs = 2
        action = torch.randn(bs, 3)
        latent = torch.randn(bs, 32)
        h0, c0 = cell.get_init_hidden(bs)
        _, _, _, _, _, (h1, c1) = cell(action, latent, (h0, c0))
        assert not torch.allclose(h0, h1)
        assert not torch.allclose(c0, c1)

    def test_weight_transfer_from_mdrnn(self):
        batch_rnn = MDRNN(latents=32, actions=3, hiddens=256, gaussians=5)
        cell_rnn = MDRNNCell(latents=32, actions=3, hiddens=256, gaussians=5)
        cell_rnn.rnn.weight_ih.data.copy_(batch_rnn.rnn.weight_ih_l0.data)
        cell_rnn.rnn.weight_hh.data.copy_(batch_rnn.rnn.weight_hh_l0.data)
        cell_rnn.rnn.bias_ih.data.copy_(batch_rnn.rnn.bias_ih_l0.data)
        cell_rnn.rnn.bias_hh.data.copy_(batch_rnn.rnn.bias_hh_l0.data)
        cell_rnn.gmm_linear.load_state_dict(batch_rnn.gmm_linear.state_dict())

        seq_len, bs = 5, 2
        actions = torch.randn(seq_len, bs, 3)
        latents = torch.randn(seq_len, bs, 32)

        mus_batch, sigmas_batch, logpi_batch, rs_batch, ds_batch = batch_rnn(
            actions, latents
        )

        h, c = cell_rnn.get_init_hidden(bs)
        cell_outs = []
        for t in range(seq_len):
            out = cell_rnn(actions[t], latents[t], (h, c))
            cell_outs.append(out)
            mus_t, sigmas_t, logpi_t, r_t, d_t, (h, c) = out

        for t in range(seq_len):
            mus_t, sigmas_t, logpi_t, r_t, d_t, _ = cell_outs[t]
            assert torch.allclose(mus_t, mus_batch[t], atol=1e-4, rtol=1e-3), (
                f"mus mismatch at t={t}"
            )
            assert torch.allclose(sigmas_t, sigmas_batch[t], atol=1e-4, rtol=1e-3), (
                f"sigmas mismatch at t={t}"
            )
            assert torch.allclose(logpi_t, logpi_batch[t], atol=1e-4, rtol=1e-3), (
                f"logpi mismatch at t={t}"
            )
            assert torch.allclose(r_t, rs_batch[t], atol=1e-4, rtol=1e-3), (
                f"r mismatch at t={t}"
            )
            assert torch.allclose(d_t, ds_batch[t], atol=1e-4, rtol=1e-3), (
                f"d mismatch at t={t}"
            )


class TestMDRNNBase:
    def test_abstract_forward_raises(self):
        base = _MDRNNBase(latents=32, actions=3, hiddens=256, gaussians=5)
        with pytest.raises(NotImplementedError):
            base(torch.randn(1, 3))
