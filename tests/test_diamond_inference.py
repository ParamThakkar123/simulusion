"""DIAMOND inference must feed the world model what it was trained on.

Two bugs in the play/record loops made every dream collapse within a few
frames, whatever the checkpoint:

* Frames were scaled to [0, 1] although the diffusion model and the policy are
  trained on [-1, 1] (``to_model_domain`` / ``_normalize_frame``), and the
  sampler's [-1, 1] output was clipped to [0, 1] and fed back as context.
* The action window was built before the current action was appended, so the
  model predicted ``obs[t+1]`` from ``a[t-L..t-1]`` instead of ``a[t-L+1..t]``,
  the window ``SequenceDataset`` trains it on.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("cv2", reason="play_diamond imports OpenCV")

from simulusion.inference.play_diamond import imagine_next_frame, to_display_frame

L = 4
SIZE = 8


class SpySampler:
    """Records what the world model is conditioned on; returns a fixed frame."""

    def __init__(self, value: float):
        self.value = value
        self.calls: list[dict] = []

    def sample(self, model, shape, device, obs_history, actions):
        self.calls.append(
            {"obs_history": obs_history.clone(), "actions": actions.clone()}
        )
        return torch.full(shape, self.value)


class FakeEnv:
    def reset(self):
        return np.zeros((SIZE, SIZE, 3), dtype=np.uint8), {}

    def close(self):
        pass


class FakeActorCritic:
    """Takes actions 1, 2, 3, ... so each step's action is identifiable."""

    def __init__(self):
        self.t = 0
        self.seen: list[torch.Tensor] = []

    def init_hidden(self, batch, device):
        return None

    def get_action(self, obs, hidden, deterministic=True):
        self.seen.append(obs.clone())
        self.t += 1
        return torch.tensor(self.t), hidden


def fake_agent(sample_value: float = 0.0) -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(num_conditioning_frames=L, obs_size=SIZE),
        device=torch.device("cpu"),
        sampler=SpySampler(sample_value),
        diffusion_model=None,
        actor_critic=FakeActorCritic(),
        env=FakeEnv(),
    )


def _load_record_diamond():
    path = Path(__file__).resolve().parents[1] / "demos" / "record_diamond.py"
    spec = importlib.util.spec_from_file_location("_demo_record_diamond", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imagine_conditions_on_latest_action_and_left_pads():
    agent = fake_agent()
    frames = [np.full((SIZE, SIZE, 3), -1.0, dtype=np.float32)] * L

    imagine_next_frame(agent, frames, [7])
    assert agent.sampler.calls[-1]["actions"].tolist() == [[0, 0, 0, 7]]

    imagine_next_frame(agent, frames, [1, 2, 3, 4, 5])
    assert agent.sampler.calls[-1]["actions"].tolist() == [[2, 3, 4, 5]]


def test_imagine_returns_model_domain_frame_unclipped():
    agent = fake_agent(sample_value=-0.8)
    frames = [np.zeros((SIZE, SIZE, 3), dtype=np.float32)] * L

    out = imagine_next_frame(agent, frames, [0])

    assert out.shape == (SIZE, SIZE, 3)
    np.testing.assert_allclose(out, -0.8)
    np.testing.assert_allclose(to_display_frame(out), 0.1, atol=1e-6)


def test_record_dream_rollout_uses_model_domain_and_current_action():
    record_diamond = _load_record_diamond()
    agent = fake_agent(sample_value=-1.0)

    frames, _ = record_diamond.rollout(
        agent, steps=5, dream=True, deterministic=True, scale=1
    )

    calls = agent.sampler.calls
    # A black reset frame is -1 in the model domain, not 0.
    assert calls[0]["obs_history"].min().item() == -1.0
    assert calls[0]["obs_history"].max().item() == -1.0
    assert agent.actor_critic.seen[0].max().item() == -1.0
    # Step t's window ends with the action the policy took at step t.
    assert calls[0]["actions"].tolist() == [[0, 0, 0, 1]]
    assert calls[4]["actions"].tolist() == [[2, 3, 4, 5]]
    # -1 in the model domain is black on screen.
    assert frames[0].max() == 0
