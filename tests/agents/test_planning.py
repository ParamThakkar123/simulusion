"""CEMPlanner / PlanningAgent act through the WorldModel protocol alone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simulusion.agents import CEMPlanner, PlanningAgent, expand_state, run_episode
from simulusion.world_model.protocol import (
    ActionSpec,
    Transition,
    WorldModelBase,
    WorldModelSpec,
)


class _TargetModel(WorldModelBase):
    """A toy world model whose reward is highest for actions near ``target``.

    Discrete variant: reward 1 for action ``target`` and 0 otherwise, and the
    episode ends after action 0.
    """

    family = "toy"

    def __init__(self, kind: str = "continuous", target: float = 0.6) -> None:
        self.device = torch.device("cpu")
        n = 2 if kind == "continuous" else 4
        self.spec = WorldModelSpec(
            family="toy",
            obs_shape=(3, 8, 8),
            action=ActionSpec(
                kind,
                n,
                -1.0 if kind == "continuous" else None,
                1.0 if kind == "continuous" else None,
            ),
            context_frames=1,
            has_termination=kind == "discrete",
        )
        self.target = target

    def _encode(self, frames, actions):
        return {"x": frames[:, -1].mean(dim=(1, 2, 3)).unsqueeze(-1)}

    def _step(self, state, action):
        if self.spec.action.kind == "continuous":
            reward = -((action - self.target) ** 2).sum(-1)
            return Transition(state=state, reward=reward)
        reward = (action == 3).float()
        return Transition(state=state, reward=reward, terminated=action == 0)

    def _decode(self, state):
        return state["x"].view(-1, 1, 1, 1).expand(-1, 3, 8, 8)

    def _save_weights(self, directory: Path):
        return {}

    @classmethod
    def _load_weights(cls, path, manifest, device, options):
        return cls()


def test_cem_finds_the_rewarded_continuous_action() -> None:
    torch.manual_seed(0)
    model = _TargetModel("continuous", target=0.6)
    state = model.encode(np.zeros((1, 3, 8, 8), np.float32))
    action = CEMPlanner(horizon=3, candidates=200, iterations=6, elites=20).plan(
        model, state
    )
    assert action.shape == (2,)
    assert torch.allclose(action, torch.full((2,), 0.6), atol=0.1)


def test_cem_picks_the_rewarded_discrete_action() -> None:
    torch.manual_seed(0)
    model = _TargetModel("discrete")
    state = model.encode(np.zeros((1, 3, 8, 8), np.float32))
    action = CEMPlanner(horizon=4, candidates=64, iterations=4, elites=8).plan(
        model, state
    )
    assert int(action) == 3


def test_expand_state_requires_a_single_state() -> None:
    state = {"h": torch.zeros(1, 4)}
    assert expand_state(state, 5)["h"].shape == (5, 4)


class _Env:
    def __init__(self) -> None:
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros((3, 8, 8), np.uint8), {}

    def step(self, action):
        self.t += 1
        return np.zeros((3, 8, 8), np.uint8), float(action == 3), False, self.t >= 5, {}


def test_planning_agent_runs_an_episode_in_a_real_env() -> None:
    torch.manual_seed(0)
    agent = PlanningAgent(
        _TargetModel("discrete"),
        CEMPlanner(horizon=3, candidates=32, iterations=3, elites=4),
    )
    assert run_episode(_Env(), agent) == 5.0


def test_planning_with_a_dreamer_world_model_trained_from_data(tmp_path: Path) -> None:
    """The model/agent split end to end: dreamer-wm trains, a CEM agent acts."""
    from simulusion.training.run import run
    from simulusion.world_model import load_world_model

    from tests.training.tasks.conftest import write_episodes

    data = write_episodes(
        tmp_path / "eps", count=3, length=8, size=64, actions=("continuous", 2)
    )
    result = run(
        overrides=[
            "trainer.device=cpu",
            f"trainer.out_dir={tmp_path.as_posix()}/runs",
            "trainer.console=false",
            "trainer.max_steps=2",
            f"data.path={data.as_posix()}",
            "data.batch_size=2",
            "model.train_seq_len=4",
            "model.obs_embed_size=32",
            "model.num_units=16",
            "model.deter_size=16",
            "model.stoch_size=4",
            "model.use_amp=false",
        ],
        task="dreamer-wm",
    )
    model = load_world_model(result.world_model_path, device="cpu")
    agent = PlanningAgent(
        model, CEMPlanner(horizon=3, candidates=16, iterations=2, elites=4), context=3
    )
    agent.reset(np.zeros((64, 64, 3), np.uint8))
    action = agent.act()
    assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
    agent.observe(action, np.zeros((64, 64, 3), np.uint8))
    assert agent.act().shape == (2,)
