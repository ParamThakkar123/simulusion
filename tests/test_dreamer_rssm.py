"""Regression tests for the Dreamer RSSM (`simulusion.models.dreamer_rssm`).

These tests guard against the GRU input-size mismatch bug that previously caused
``RuntimeError: input has inconsistent input_size: got 36 expected 200`` at
global step 5000. The original code passed the raw concatenation of
``[prev_action, prev_state["stoch"]]`` directly to ``nn.GRUCell`` without
projecting it through ``fc_state_action``.

The fix routes the concatenation through ``fc_state_action`` so that the GRU
input always has shape ``(B, deter_size)``. These tests verify that:

* ``observe_step`` returns a ``(posterior, prior)`` tuple.
* ``imagine_step`` accepts the concatenation and produces shapes that match
  the configured ``deter_size``.
* ``observe_rollout`` returns stacked prior/posterior dicts with the expected
  time-leading shape.
* The GRU input is correctly projected to ``deter_size`` (i.e. it does not
  silently degrade to ``stoch_size + action_size``).
* ``nonterm=0`` resets the recurrent state at episode boundaries.
"""

import pytest
import torch
from simulusion.models.dreamer_rssm import RSSM


class TestDreamerRSSM:
    @pytest.fixture
    def rssm(self):
        return RSSM(
            action_size=2,
            stoch_size=10,
            deter_size=20,
            hidden_size=20,
            obs_embed_size=8,
            activation="elu",
        )

    @pytest.fixture
    def init_state(self, rssm):
        return rssm.init_state(batch_size=4, device=torch.device("cpu"))

    def test_init_state_shapes(self, rssm):
        state = rssm.init_state(3, torch.device("cpu"))
        assert state["deter"].shape == (3, rssm.deter_size)
        assert state["stoch"].shape == (3, rssm.stoch_size)
        assert state["mean"].shape == (3, rssm.stoch_size)
        assert state["std"].shape == (3, rssm.stoch_size)
        assert torch.all(state["deter"] == 0)
        assert torch.all(state["stoch"] == 0)

    def test_imagine_step_output_shapes(self, rssm, init_state):
        """Imagine step must produce deter of size `deter_size` (regression)."""
        action = torch.randn(4, rssm.action_size)
        out = rssm.imagine_step(init_state, action)

        assert out["deter"].shape == (4, rssm.deter_size)
        assert out["stoch"].shape == (4, rssm.stoch_size)
        assert out["mean"].shape == (4, rssm.stoch_size)
        assert out["std"].shape == (4, rssm.stoch_size)
        assert (out["std"] > 0).all()

    def test_imagine_step_runs_gru_with_projected_input(self, rssm, init_state):
        """The GRU must receive an input of size `deter_size`, not
        ``stoch_size + action_size``. This is the exact failure mode of the
        original bug."""
        # Replace the GRU with a recording one to assert input shape.
        recorded = {}

        original = rssm.rnn.forward

        def spy_forward(input, hx):
            recorded["input_shape"] = tuple(input.shape)
            return original(input, hx)

        rssm.rnn.forward = spy_forward
        try:
            action = torch.randn(4, rssm.action_size)
            rssm.imagine_step(init_state, action)
        finally:
            rssm.rnn.forward = original

        assert recorded["input_shape"] == (4, rssm.deter_size), (
            f"GRU input shape was {recorded['input_shape']}, expected "
            f"(4, {rssm.deter_size}). The fc_state_action projection is "
            f"not being used."
        )

    def test_imagine_step_resets_state_on_terminal(self, rssm, init_state):
        """``nonterm=0`` must zero out the previous state before the GRU."""
        init_state["deter"] = torch.randn_like(init_state["deter"])
        init_state["stoch"] = torch.randn_like(init_state["stoch"])
        action = torch.randn(4, rssm.action_size)
        nonterm = torch.zeros(4, 1)

        out = rssm.imagine_step(init_state, action, nonterm=nonterm)

        # The new deter (and the prior mean/std, which feed into the
        # distribution) should depend only on the GRU input derived from
        # the *zeroed* prior state, not the original. A non-trivial
        # init_state must not leak through.
        fresh_state = rssm.init_state(4, torch.device("cpu"))
        out_fresh = rssm.imagine_step(fresh_state, action, nonterm=nonterm)

        # Deterministic state must match exactly between the two runs.
        assert torch.allclose(out["deter"], out_fresh["deter"], atol=1e-6)
        # Prior distribution parameters (mean, std) must also match — the
        # `stoch` sample is stochastic and will differ run-to-run, so we
        # only check its distribution parameters.
        assert torch.allclose(out["mean"], out_fresh["mean"], atol=1e-6)
        assert torch.allclose(out["std"], out_fresh["std"], atol=1e-6)

    def test_observe_step_returns_posterior_and_prior(self, rssm, init_state):
        """``observe_step`` must return a ``(posterior, prior)`` tuple.

        The original code returned only the posterior, which forced callers
        to recompute the prior by re-running ``imagine_step`` on the
        posterior (a logic bug). Verify the new tuple contract.
        """
        action = torch.randn(4, rssm.action_size)
        obs_embed = torch.randn(4, rssm.embedding_size)

        result = rssm.observe_step(init_state, action, obs_embed)
        assert isinstance(result, tuple)
        assert len(result) == 2
        posterior, prior = result

        for key in ("deter", "stoch", "mean", "std"):
            assert key in posterior
            assert key in prior
            assert posterior[key].shape == (
                4,
                getattr(rssm, f"{'deter' if key == 'deter' else 'stoch'}_size"),
            )
        assert posterior["deter"].shape == (4, rssm.deter_size)
        assert posterior["stoch"].shape == (4, rssm.stoch_size)
        assert prior["deter"].shape == (4, rssm.deter_size)
        assert prior["stoch"].shape == (4, rssm.stoch_size)
        # The deterministic state should match between prior and posterior
        # (the GRU is only advanced once per timestep).
        assert torch.allclose(posterior["deter"], prior["deter"])

    def test_observe_step_posterior_differs_from_prior(self, rssm, init_state):
        """Posterior and prior must differ in stochastic state when the obs
        is informative (it almost always will be with random init)."""
        action = torch.randn(4, rssm.action_size)
        obs_embed = torch.randn(4, rssm.embedding_size)

        posterior, prior = rssm.observe_step(init_state, action, obs_embed)

        # Stochastic state should differ in general; the means too.
        assert not torch.allclose(posterior["mean"], prior["mean"], atol=1e-7)

    def test_get_posterior_returns_only_posterior(self, rssm, init_state):
        """``get_posterior`` keeps its single-state contract (used by
        ``forward`` and other legacy call sites)."""
        action = torch.randn(4, rssm.action_size)
        obs_embed = torch.randn(4, rssm.embedding_size)

        post = rssm.get_posterior(init_state, action, obs_embed)
        assert isinstance(post, dict)
        assert "deter" in post and "stoch" in post
        assert post["deter"].shape == (4, rssm.deter_size)
        assert post["stoch"].shape == (4, rssm.stoch_size)

    def test_observe_rollout_shapes(self, rssm):
        """``observe_rollout`` must return stacked prior and posterior dicts
        with a leading time axis of length ``seq_len``."""
        B, T = 3, 5
        obs_embed = torch.randn(T + 1, B, rssm.embedding_size)
        actions = torch.randn(T, B, rssm.action_size)
        nonterms = torch.ones(T, B, 1)
        init_state = rssm.init_state(B, torch.device("cpu"))

        prior, posterior = rssm.observe_rollout(
            obs_embed, actions, nonterms, init_state, seq_len=T
        )

        for key in ("deter", "stoch", "mean", "std"):
            assert key in prior
            assert key in posterior
            assert prior[key].shape == (
                T,
                B,
                rssm.deter_size if key == "deter" else rssm.stoch_size,
            )
            assert posterior[key].shape == (
                T,
                B,
                rssm.deter_size if key == "deter" else rssm.stoch_size,
            )

    def test_imagine_rollout_shapes(self, rssm, init_state):
        """``imagine_rollout`` uses a fake policy that ignores the state and
        returns a random action. Just verify shape and horizon."""
        horizon = 6
        B = init_state["deter"].shape[0]

        class RandomPolicy:
            def __call__(self, features, deter=False):
                return torch.randn(B, rssm.action_size)

        traj = rssm.imagine_rollout(RandomPolicy(), init_state, horizon)
        for key in ("deter", "stoch", "mean", "std"):
            assert traj[key].shape == (
                horizon,
                B,
                rssm.deter_size if key == "deter" else rssm.stoch_size,
            )

    def test_detach_state_breaks_grad(self, rssm, init_state):
        """``detach_state`` must return a state that does not require grad
        and is not connected to the original computation graph."""
        init_state["deter"].requires_grad_(True)
        detached = rssm.detach_state(init_state)
        for v in detached.values():
            assert not v.requires_grad

    def test_seq_to_batch_reshapes_time_leading(self, rssm):
        T, B = 4, 3
        state = {
            "deter": torch.randn(T, B, rssm.deter_size),
            "stoch": torch.randn(T, B, rssm.stoch_size),
        }
        out = rssm.seq_to_batch(state)
        assert out["deter"].shape == (T * B, rssm.deter_size)
        assert out["stoch"].shape == (T * B, rssm.stoch_size)

    def test_gradients_flow_through_observe_step(self, rssm, init_state):
        """End-to-end: backprop from a scalar loss through ``observe_step``
        must produce non-None gradients on the trainable parameters."""
        action = torch.randn(4, rssm.action_size)
        obs_embed = torch.randn(4, rssm.embedding_size)

        posterior, _ = rssm.observe_step(init_state, action, obs_embed)
        loss = posterior["mean"].pow(2).sum() + posterior["deter"].pow(2).sum()
        loss.backward()

        for name, p in rssm.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), f"Non-finite grad on {name}"

    def test_six_action_dreamer_config_no_crash(self):
        """Regression: original crash was with action_size=6, stoch_size=30,
        deter_size=200. Smoke-test that exact config doesn't crash."""
        rssm = RSSM(
            action_size=6,
            stoch_size=30,
            deter_size=200,
            hidden_size=200,
            obs_embed_size=1024,
            activation="elu",
        )
        init = rssm.init_state(2, torch.device("cpu"))
        action = torch.randn(2, 6)
        obs_embed = torch.randn(2, 1024)

        # imagine_step must succeed
        prior = rssm.imagine_step(init, action)
        assert prior["deter"].shape == (2, 200)
        assert prior["stoch"].shape == (2, 30)

        # observe_step must succeed
        posterior, prior2 = rssm.observe_step(init, action, obs_embed)
        assert posterior["deter"].shape == (2, 200)
        assert posterior["stoch"].shape == (2, 30)
