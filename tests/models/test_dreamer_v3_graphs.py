"""CUDA-graph replay of DreamerV3's rollouts matches eager execution.

The graph tests need a CUDA device. The sampler tests run anywhere: they check
that the sync-free samplers draw from the intended distributions.
"""

import gc

import numpy as np
import pytest
import torch

from simulusion.configs.dreamer_v3_config import DreamerV3Config
from simulusion.models.categorical_rssm import CategoricalRSSM
from simulusion.models.dreamer_v3 import DreamerV3
from simulusion.vision.dreamer_v3_nets import DreamerV3Actor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _release():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tiny_config(**overrides):
    defaults = dict(
        model_size="12m",
        hidden_size=32,
        recurrent_units=64,
        cnn_depth=4,
        latent_classes=4,
        latent_dim=4,
        gru_blocks=4,
        mlp_layers=1,
        num_buckets=41,
        batch_size=3,
        train_seq_len=5,
        imagine_horizon=4,
        buffer_size=200,
        actor_dist="normal",
    )
    defaults.update(overrides)
    return DreamerV3Config(**defaults)


def make_deterministic(rssm):
    """Replace latent sampling by a straight-through argmax, in place."""

    def sample(logits):
        probs = torch.softmax(logits, dim=-1)
        index = probs.argmax(-1)
        hard = torch.nn.functional.one_hot(index, probs.shape[-1]).to(probs.dtype)
        return hard + probs - probs.detach()

    rssm._sample = sample


# ----------------------------------------------------------------------
# Samplers (any device)
# ----------------------------------------------------------------------


def test_latent_sampler_draws_from_the_categorical():
    torch.manual_seed(0)
    rssm = CategoricalRSSM(
        action_size=2,
        embed_size=8,
        deter_size=16,
        hidden_size=16,
        latent_dim=1,
        latent_classes=4,
        gru_blocks=2,
        unimix=0.0,
    )
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    logits = probs.log().expand(200_000, 1, 4)
    sample = rssm._sample(logits)
    torch.testing.assert_close(sample.detach().sum(-1), torch.ones(200_000, 1))
    freq = sample.detach().mean(0)[0]
    assert torch.allclose(freq, probs, atol=0.005)
    # Straight-through: gradients reach the probabilities.
    logits = torch.zeros(3, 1, 4, requires_grad=True)
    (rssm._sample(logits) * torch.arange(4.0)).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_discrete_actor_sampler_draws_from_its_distribution():
    torch.manual_seed(0)
    actor = DreamerV3Actor(8, 3, discrete=True, layers=1, units=16)
    features = torch.randn(1, 8).expand(100_000, 8)
    with torch.no_grad():
        actions = actor(features)
        expected = actor.dist(features[:1]).probs[0]
    assert torch.equal(actions.sum(-1), torch.ones(100_000))
    assert torch.allclose(actions.mean(0), expected, atol=0.006)


def test_discrete_actor_mode_is_the_most_likely_action():
    torch.manual_seed(0)
    actor = DreamerV3Actor(8, 5, discrete=True, layers=1, units=16)
    features = torch.randn(64, 8)
    with torch.no_grad():
        actions = actor(features, deterministic=True)
        probs = actor.dist(features).probs
    assert torch.equal(actions.argmax(-1), probs.argmax(-1))
    assert torch.equal(actions.sum(-1), torch.ones(64))


def test_continuous_actor_sampler_matches_the_normal():
    torch.manual_seed(0)
    actor = DreamerV3Actor(8, 2, discrete=False, layers=1, units=16)
    features = torch.randn(1, 8).expand(200_000, 8)
    with torch.no_grad():
        actions = actor(features)
        dist = actor.dist(features[:1])
    assert torch.allclose(actions.mean(0), dist.mean[0], atol=0.01)
    assert torch.allclose(actions.std(0), dist.stddev[0], atol=0.01)


# ----------------------------------------------------------------------
# Graph replay (CUDA)
# ----------------------------------------------------------------------


def build(**config):
    torch.manual_seed(0)
    model = DreamerV3(tiny_config(**config), {"state": (6,)}, 2, "cuda")
    make_deterministic(model.rssm)
    return model


def observe_inputs(model, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T, B = 5, 3
    embed = torch.randn(
        T, B, model.obs_encoder.embed_size, device="cuda", generator=gen
    ).requires_grad_(True)
    actions = torch.rand(T, B, 2, device="cuda", generator=gen) * 2 - 1
    is_first = torch.zeros(T, B, 1, device="cuda")
    is_first[0] = 1.0
    is_first[3, 1] = 1.0
    init = model.rssm.init_state(B, model.device)
    init["deter"] = torch.randn(
        B, init["deter"].shape[-1], device="cuda", generator=gen
    )
    return embed, actions, is_first, init


def rollout_loss(posterior, prior):
    return (
        posterior["deter"].square().mean()
        + posterior["logit"].exp().mean()
        + prior["logit"].sin().mean()
        + posterior["stoch"]
        .mul(torch.linspace(0, 1, posterior["stoch"].shape[-1], device="cuda"))
        .mean()
    )


@cuda
def test_graphed_observe_rollout_matches_eager_forward_and_backward():
    model = build()
    params = list(model.rssm.parameters())
    # Capture before any eager autograd graph exists: CUDA graphs cannot be
    # captured while an older graph's gradient accumulators are still alive.
    model._observe_rollout(*observe_inputs(model, 0))
    assert model._observe_graph is not None
    for seed in (1, 2, 3):  # Replays with fresh inputs, not just the captured ones.
        inputs = observe_inputs(model, seed)
        graph_post, graph_prior = model._observe_rollout(*inputs)
        graph_loss = rollout_loss(graph_post, graph_prior)
        graph_grads = torch.autograd.grad(
            graph_loss, [inputs[0]] + params, allow_unused=True
        )

        inputs = observe_inputs(model, seed)
        eager_post, eager_prior = model.rssm.observe_rollout(*inputs)
        eager_loss = rollout_loss(eager_post, eager_prior)
        eager_grads = torch.autograd.grad(
            eager_loss, [inputs[0]] + params, allow_unused=True
        )

        for key in ("logit", "stoch", "deter"):
            torch.testing.assert_close(graph_post[key], eager_post[key])
        torch.testing.assert_close(graph_prior["logit"], eager_prior["logit"])
        for eager, graph in zip(eager_grads, graph_grads):
            if eager is None:
                assert graph is None or not graph.abs().any()
            else:
                torch.testing.assert_close(graph, eager, rtol=1e-4, atol=1e-6)


@cuda
def test_graphed_imagination_matches_eager():
    model = build()
    # Deterministic policy so both runs take the same actions.
    model.actor.forward = lambda feat, deterministic=False: model.actor.dist(feat).mean
    for seed in (1, 2):
        torch.manual_seed(seed)
        start = model.rssm.init_state(6, model.device)
        start["deter"] = torch.randn_like(start["deter"])
        start["logit"] = torch.randn(6, 4, 4, device="cuda")
        start["stoch"] = torch.softmax(start["logit"], -1)
        with torch.no_grad():
            eager_states, eager_actions = model.rssm.imagine_rollout(
                model.actor.forward, start, 4
            )
            graph_states, graph_actions = model._imagine_rollout(start, 4)
        assert model._imagine_graph is not None
        torch.testing.assert_close(graph_actions, eager_actions)
        for key in ("logit", "stoch", "deter"):
            torch.testing.assert_close(graph_states[key], eager_states[key])


def imagination_start(model, seed):
    torch.manual_seed(seed)
    start = model.rssm.init_state(6, model.device)
    start["deter"] = torch.randn_like(start["deter"])
    start["logit"] = torch.randn(6, 4, 4, device="cuda")
    start["stoch"] = torch.softmax(start["logit"], -1)
    return start


@cuda
def test_graphed_imagination_matches_eager_for_discrete_actions():
    # The discrete head once built a OneHotCategorical inside the graph, whose
    # argument check syncs with the host and invalidated every capture.
    torch.manual_seed(0)
    model = DreamerV3(tiny_config(actor_dist="onehot"), {"state": (6,)}, 3, "cuda")
    make_deterministic(model.rssm)
    sample = model.actor.forward
    model.actor.forward = lambda feat, deterministic=False: sample(feat, True)
    for seed in (1, 2):
        start = imagination_start(model, seed)
        with torch.no_grad():
            eager_states, eager_actions = model.rssm.imagine_rollout(
                model.actor.forward, start, 4
            )
            graph_states, graph_actions = model._imagine_rollout(start, 4)
        assert model._imagine_graph is not None
        torch.testing.assert_close(graph_actions, eager_actions)
        for key in ("logit", "stoch", "deter"):
            torch.testing.assert_close(graph_states[key], eager_states[key])


@cuda
def test_failed_capture_falls_back_to_eager():
    model = build()
    act = model.actor.forward

    def syncing_policy(feat, deterministic=False):
        if torch.cuda.is_current_stream_capturing():
            feat.sum().item()  # A host sync, which no CUDA graph can contain.
        return act(feat, deterministic)

    model.actor.forward = syncing_policy
    start = imagination_start(model, 1)
    with torch.no_grad():
        states, actions = model._imagine_rollout(start, 4)
    assert model._imagine_graph is None
    assert not model._graphs_enabled()
    assert torch.cuda.current_stream() == torch.cuda.default_stream()
    assert torch.isfinite(actions).all() and torch.isfinite(states["deter"]).all()
    # CUDA keeps working afterwards.
    assert torch.ones(3, device="cuda").sum().item() == 3.0


@cuda
def test_graphs_follow_in_place_weight_updates():
    model = build()
    model._observe_rollout(*observe_inputs(model, 1))  # Capture.
    with torch.no_grad():
        for param in model.rssm.parameters():
            param.mul_(0.5).add_(0.01)
    eager_post, _ = model.rssm.observe_rollout(*observe_inputs(model, 4))
    graph_post, _ = model._observe_rollout(*observe_inputs(model, 4))
    torch.testing.assert_close(graph_post["deter"], eager_post["deter"])


@pytest.mark.parametrize("actor_dist", ["normal", "onehot"])
@cuda
def test_training_with_graphs_runs_and_can_be_disabled(actor_dist):
    rng = np.random.default_rng(0)
    for enabled in (True, False):
        torch.manual_seed(0)
        config = tiny_config(cuda_graphs=enabled, actor_dist=actor_dist)
        model = DreamerV3(config, {"state": (6,)}, 2, "cuda")
        for t in range(60):
            if actor_dist == "onehot":
                action = np.eye(2, dtype=np.float32)[rng.integers(2)]
            else:
                action = rng.uniform(-1, 1, 2).astype(np.float32)
            model.data_buffer.add(
                {"state": rng.normal(size=6).astype(np.float32)},
                action,
                float(rng.normal()),
                is_first=t % 20 == 0,
                is_last=t % 20 == 19,
            )
            if t % 20 == 19:
                model.data_buffer.end_episode()
        for _ in range(3):
            losses = model.train_one_batch()
            assert all(np.isfinite(losses))
        assert (model._observe_graph is not None) == enabled
        assert (model._imagine_graph is not None) == enabled
        assert all(isinstance(v, float) for v in model.metrics.values())
