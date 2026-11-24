"""LeWorldModel: components, objective, world model adapter and goal planning.

The defaults are pinned to the paper (Maes et al., 2026, arXiv:2603.19312) and
the authors' code (github.com/lucas-maes/le-wm), so a change to any of them
is deliberate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from simulusion.agents import GoalCEMPlanner, GoalPlanningAgent
from simulusion.configs.lewm_config import LeWMConfig
from simulusion.models.lewm import (
    ARPredictor,
    ActionEmbedder,
    BNProjector,
    LeWM,
    LeWMEncoder,
    SIGReg,
    epps_pulley,
    lr_lambda,
    surprise,
    temporal_straightness,
)
from simulusion.world_model import Rollout, as_world_model, load_world_model
from simulusion.world_model.lewm import LeWMWorldModel


def _tiny(**overrides: Any) -> LeWMConfig:
    values: dict[str, Any] = dict(
        img_size=28,
        patch_size=14,
        encoder_dim=32,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=2,
        projector_hidden_dim=16,
        action_dim=2,
        frameskip=2,
        history_size=2,
        predictor_depth=1,
        predictor_heads=2,
        predictor_dim_head=8,
        predictor_mlp_dim=16,
        sigreg_num_proj=16,
    )
    values.update(overrides)
    return LeWMConfig(**values)


# ------------------------------------------------------------- paper values


def test_defaults_follow_the_paper_and_code():
    cfg = LeWMConfig()
    # Sec. 3.1 and App. D: ViT-tiny, patch 14, 224 px frames.
    assert (cfg.img_size, cfg.patch_size) == (224, 14)
    assert (cfg.encoder_dim, cfg.encoder_depth, cfg.encoder_heads) == (192, 12, 3)
    # Sec. 3.1: 6-layer, 16-head predictor with 10% dropout; history 3.
    assert (cfg.predictor_depth, cfg.predictor_heads) == (6, 16)
    assert cfg.predictor_dropout == 0.1 and cfg.history_size == 3
    assert (cfg.predictor_dim_head, cfg.predictor_mlp_dim) == (64, 2048)
    # Sec. 3.1: M = 1024 projections and lambda 0.1; code: 17 knots on [0, 3].
    assert (cfg.sigreg_num_proj, cfg.sigreg_knots, cfg.sigreg_t_max) == (1024, 17, 3.0)
    assert cfg.sigreg_weight == 0.1
    # App. D: frame skip 5, batch 128, windows of 4 frames.
    assert cfg.frameskip == 5 and cfg.batch_size == 128
    assert cfg.history_size + cfg.num_preds == 4
    assert cfg.window_frames == 16
    # Code: AdamW 5e-5, weight decay 1e-3, grad clip 1.
    assert (cfg.lr, cfg.weight_decay, cfg.grad_clip) == (5e-5, 1e-3, 1.0)
    # App. B and D: CEM with 300 samples, 30 elites, 30 iterations, horizon 5.
    assert (cfg.plan_samples, cfg.plan_elites, cfg.plan_iterations) == (300, 30, 30)
    assert cfg.plan_horizon == 5 and cfg.plan_init_std == 1.0


def test_parameter_counts_match_the_paper():
    model = LeWM(LeWMConfig(action_dim=2))

    def millions(module: torch.nn.Module) -> float:
        return sum(p.numel() for p in module.parameters()) / 1e6

    # "ViT tiny (~5M parameters)" and a "~10M" predictor (Sec. 3.1).
    assert 5.0 < millions(model.encoder) < 6.0
    assert 10.0 < millions(model.predictor) < 11.5


# ------------------------------------------------------------------ SIGReg


def test_sigreg_matches_the_epps_pulley_statistic():
    torch.manual_seed(0)
    z = torch.randn(3, 50, 8)
    torch.manual_seed(1)
    value = SIGReg(knots=17, num_proj=32)(z)
    torch.manual_seed(1)
    directions = torch.randn(8, 32)
    directions = directions / directions.norm(dim=0)
    expected = epps_pulley(z @ directions).mean()
    assert torch.allclose(value, expected, atol=1e-5)


def test_sigreg_separates_gaussian_from_collapsed_embeddings():
    torch.manual_seed(0)
    sigreg = SIGReg(num_proj=64)
    gaussian = sigreg(torch.randn(1, 512, 16))
    collapsed = sigreg(torch.zeros(1, 512, 16) + 0.5)
    shifted = sigreg(torch.randn(1, 512, 16) * 3.0)
    # The statistic is scaled by the sample count, as in the reference, so a
    # true Gaussian scores about 1 (0.7-1.6 over seeds) and a degenerate batch
    # scores in the hundreds.
    assert gaussian < 2.0
    assert collapsed > 50 * gaussian and shifted > 50 * gaussian


def test_sigreg_gradient_pushes_towards_isotropy():
    torch.manual_seed(0)
    z = (0.1 * torch.randn(256, 8)).requires_grad_()
    opt = torch.optim.SGD([z], lr=0.5)
    sigreg = SIGReg(num_proj=64)
    start = float(sigreg(z))
    for _ in range(50):
        opt.zero_grad()
        sigreg(z).backward()
        opt.step()
    assert float(sigreg(z)) < 0.5 * start
    assert 0.5 < float(z.detach().std()) < 1.5


# --------------------------------------------------------------- components


def test_encoder_returns_one_cls_embedding_per_frame():
    enc = LeWMEncoder(img_size=28, patch_size=14, embed_dim=32, depth=1, num_heads=2)
    assert enc(torch.randn(3, 3, 28, 28)).shape == (3, 32)
    # Other sizes interpolate the position embeddings.
    assert enc(torch.randn(2, 3, 42, 42)).shape == (2, 32)


def test_predictor_is_causal_and_starts_action_free():
    torch.manual_seed(0)
    pred = ARPredictor(num_frames=3, dim=16, depth=2, heads=2, dim_head=8, mlp_dim=32)
    pred.eval()
    emb, act = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    out = pred(emb, act)
    # AdaLN-zero: at initialization the actions change nothing.
    assert torch.allclose(out, pred(emb, torch.randn(2, 3, 16)))
    for p in pred.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    out = pred(emb, act)
    later = emb.clone()
    # A random direction: a constant shift of every dimension would be removed
    # by the LayerNorms.
    later[:, 2] += torch.randn(16)
    changed = pred(later, act)
    assert torch.allclose(out[:, :2], changed[:, :2], atol=1e-6)
    assert not torch.allclose(out[:, 2], changed[:, 2])
    with pytest.raises(ValueError, match="at most 3"):
        pred(torch.randn(1, 4, 16), torch.randn(1, 4, 16))


def test_action_embedder_and_projector_shapes():
    assert ActionEmbedder(10, 32)(torch.randn(2, 3, 10)).shape == (2, 3, 32)
    proj = BNProjector(32, 64)
    assert proj(torch.randn(4, 3, 32)).shape == (4, 3, 32)
    assert isinstance(proj.net[1], torch.nn.BatchNorm1d)


# ------------------------------------------------------------------- model


def test_loss_has_the_two_terms_and_trains_end_to_end():
    torch.manual_seed(0)
    model = LeWM(_tiny())
    frames = torch.rand(4, 3, 3, 28, 28)
    actions = torch.randn(4, 2, model.action_block_dim)
    out = model.loss(frames, actions)
    assert set(out) == {"loss", "pred_loss", "sigreg_loss"}
    assert torch.allclose(out["loss"], out["pred_loss"] + 0.1 * out["sigreg_loss"])
    out["loss"].backward()

    def reached(name):
        grads = [p.grad for p in getattr(model, name).parameters()]
        return any(g is not None and g.abs().sum() > 0 for g in grads)

    # Gradients reach every part: no stop-gradient or frozen target.
    for name in ("encoder", "projector", "predictor", "pred_proj"):
        assert reached(name), name
    # AdaLN-zero starts every action modulation at zero, so the action encoder
    # gets its first gradient once a step has moved those weights.
    assert not reached("action_encoder")
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    model.zero_grad()
    model.loss(frames, actions)["loss"].backward()
    assert reached("action_encoder")


def test_actions_are_z_scored_with_tiled_statistics_and_nan_padding():
    model = LeWM(_tiny())
    model.set_action_stats(torch.tensor([1.0, -1.0]), torch.tensor([2.0, 0.5]))
    assert model.action_mean.tolist() == [1.0, -1.0, 1.0, -1.0]
    raw = torch.tensor([[[3.0, -0.5, float("nan"), -1.0]]])
    assert model.normalize_actions(raw).tolist() == [[[1.0, 1.0, 0.0, 0.0]]]
    back = model.denormalize_actions(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    assert back.tolist() == [[3.0, -0.5, 1.0, -1.0]]


def test_rollout_predicts_one_embedding_per_action():
    model = LeWM(_tiny()).eval()
    emb = model.encode(torch.rand(2, 1, 3, 28, 28))
    out = model.rollout(emb, torch.zeros(2, 1, 4), torch.zeros(2, 3, 4))
    assert out.shape == (2, 5, 32)


def test_diagnostics():
    line = torch.arange(5.0).view(1, 5, 1).expand(2, 5, 3)
    assert torch.allclose(temporal_straightness(line), torch.tensor(1.0))
    zigzag = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])
    assert torch.allclose(temporal_straightness(zigzag), torch.tensor(-1.0))
    assert surprise(torch.zeros(2, 3, 4), torch.ones(2, 3, 4)).shape == (2, 3)


def test_lr_schedule_warms_up_from_zero_then_anneals_to_zero():
    f = lr_lambda(10, 110)
    assert f(0) == 0.0 and f(5) == 0.5 and f(10) == 1.0
    assert abs(f(60) - 0.5) < 1e-9 and f(110) == 0.0


def test_create_model_builds_lewm():
    import simulusion

    model = simulusion.create_model("lewm", action_dim=2, img_size=28)
    assert isinstance(model, LeWM) and model.config.action_dim == 2


# ------------------------------------------------------------ world model


def test_world_model_keeps_a_history_window_and_roundtrips(tmp_path: Path):
    torch.manual_seed(0)
    wm = as_world_model(LeWM(_tiny()))
    assert isinstance(wm, LeWMWorldModel)
    spec = wm.spec
    assert spec.family == "lewm" and spec.action.n == 4 and not spec.has_reward
    assert spec.max_context == 2 and not spec.stochastic

    state = wm.encode(torch.rand(1, 3, 3, 28, 28), torch.randn(1, 2, 4))
    assert state["emb"].shape == (1, 2, 32) and state["act"].shape == (1, 1, 4)
    out = wm.rollout(state, torch.randn(1, 3, 4))
    assert out.frames is None and out.rewards is None
    assert out.state["emb"].shape == (1, 2, 32) and out.state["act"].shape == (1, 1, 4)
    with pytest.raises(NotImplementedError, match="no pixel decoder"):
        wm.decode(state)

    # A single step equals the model's own prediction from that history.
    step = wm.step(state, torch.ones(1, 4))
    acts = torch.cat([state["act"], torch.ones(1, 1, 4)], dim=1)
    expected = wm.model.predict(state["emb"], acts)[:, -1]
    assert torch.allclose(step.state["emb"][:, -1], expected, atol=1e-6)

    saved = wm.save(tmp_path / "wm")
    loaded = load_world_model(saved, device="cpu")
    assert isinstance(loaded, LeWMWorldModel)
    again = loaded.encode(torch.zeros(1, 1, 3, 28, 28))
    assert torch.allclose(again["emb"], wm.encode(torch.zeros(1, 1, 3, 28, 28))["emb"])


def test_goal_cost_is_the_squared_latent_distance():
    wm = LeWMWorldModel(LeWM(_tiny()))
    state = wm.encode(torch.rand(2, 1, 3, 28, 28))
    goal = wm.encode_goal(torch.rand(1, 3, 28, 28))
    cost = wm.goal_cost(state, goal)
    expected = (state["emb"][:, -1] - goal).pow(2).sum(-1)
    assert cost.shape == (2,) and torch.allclose(cost, expected)


# ---------------------------------------------------------------- planning


class _ToyLatent:
    """A latent model whose state moves by the sum of the (raw) actions."""

    def __init__(self, scale: float = 1.0) -> None:
        from simulusion.world_model import ActionSpec, WorldModelSpec

        self.device = torch.device("cpu")
        self.spec = WorldModelSpec(
            family="toy",
            obs_shape=(3, 8, 8),
            action=ActionSpec("continuous", 2),
            has_reward=False,
        )
        self.action_mean = torch.tensor([5.0, -5.0])
        self.action_std = torch.full((2,), scale)

    def rollout(self, state: Any, actions: torch.Tensor, *, decode: bool) -> Rollout:
        assert not decode
        pos = state["pos"] + actions.sum(dim=1)
        return Rollout(state={"pos": pos}, frames=None, rewards=None, terminated=None)

    def goal_cost(self, state: Any, goal: torch.Tensor) -> torch.Tensor:
        return (state["pos"] - goal).pow(2).sum(-1)


def test_goal_cem_reaches_the_goal_in_raw_action_units():
    torch.manual_seed(0)
    model = _ToyLatent(scale=2.0)
    planner = GoalCEMPlanner(horizon=3, candidates=200, iterations=15, elites=20)
    goal = torch.tensor([16.0, -14.0])
    plan = planner.plan(model, {"pos": torch.zeros(1, 2)}, goal)
    assert plan.shape == (3, 2)
    assert torch.allclose(plan.sum(0), goal, atol=0.5)


def test_goal_cem_needs_a_latent_model():
    with pytest.raises(TypeError, match="goal_cost"):
        GoalCEMPlanner().plan(object(), {}, torch.zeros(2))


def test_goal_agent_follows_the_goal_reaching_protocol():
    # reset(goal) with the goal frame as {"image": (3, H, W)}, then act(obs)
    # per environment step: a plan of horizon blocks splits into frameskip
    # actions each, and is replanned only when used up.
    torch.manual_seed(0)
    wm = LeWMWorldModel(LeWM(_tiny()))
    planner = GoalCEMPlanner(horizon=2, candidates=8, iterations=2, elites=2)
    agent = GoalPlanningAgent(wm, planner=planner, frameskip=2)
    obs = {"image": np.zeros((3, 28, 28), np.uint8)}
    with pytest.raises(RuntimeError, match="No goal"):
        agent.act(obs)
    agent.reset({"image": np.full((3, 28, 28), 255, np.uint8)})
    actions = [agent.act(obs) for _ in range(4)]
    assert all(a.shape == (2,) for a in actions)
    assert len(agent._queue) == 0
    first = agent.goal.clone()
    agent.reset({"image": np.zeros((3, 28, 28), np.uint8)})
    assert not torch.allclose(agent.goal, first)
