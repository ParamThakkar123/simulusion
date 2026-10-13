"""DIAMOND's exported steps must reproduce the eager sampler and policy.

The web player runs ``dream_step`` and ``policy_step`` instead of the Python
loop, so a dream in the browser is only the model's dream if these match
``EulerSampler.sample`` and ``ActorCriticNetwork`` exactly.
"""

from types import SimpleNamespace

import pytest
import torch

from simulusion.export import (
    DiamondDreamStepExport,
    DiamondPolicyStepExport,
    _resolve_export_module,
    export_any,
)
from simulusion.models.diffusion.actor_critic import ActorCriticNetwork
from simulusion.models.diffusion.diamond_diffusion import DiffusionUNet, EulerSampler

L = 4
SIZE = 16
ACTIONS = 4
LSTM = 32


@pytest.fixture
def agent():
    torch.manual_seed(0)
    return SimpleNamespace(
        config=SimpleNamespace(
            obs_size=SIZE, num_conditioning_frames=L, actor_lstm_dim=LSTM
        ),
        device=torch.device("cpu"),
        diffusion_model=DiffusionUNet(
            base_channels=32,
            channel_multipliers=(1, 1),
            num_res_blocks=1,
            cond_dim=32,
            action_dim=ACTIONS,
        ).eval(),
        sampler=EulerSampler(num_steps=3),
        actor_critic=ActorCriticNetwork(
            action_dim=ACTIONS, channels=(32, 32), lstm_dim=LSTM, frame_size=SIZE
        ).eval(),
    )


def test_resolves_dream_and_policy_steps(agent):
    assert isinstance(_resolve_export_module(agent), DiamondDreamStepExport)
    assert isinstance(
        _resolve_export_module(agent, "policy_step"), DiamondPolicyStepExport
    )


def test_dream_step_matches_euler_sampler(agent):
    obs = torch.rand(1, L, 3, SIZE, SIZE) * 2 - 1
    actions = torch.tensor([[0, 1, 2, 3]])
    shape = (1, 3, SIZE, SIZE)

    # EulerSampler.sample draws its starting noise with randn(shape) first.
    torch.manual_seed(1)
    noise = torch.randn(shape)
    torch.manual_seed(1)
    eager = agent.sampler.sample(
        agent.diffusion_model, shape, torch.device("cpu"), obs, actions
    )
    with torch.no_grad():
        step = DiamondDreamStepExport(agent)(obs, actions, noise)
    torch.testing.assert_close(step, eager)


def test_policy_step_threads_lstm_state(agent):
    frames = torch.rand(3, 1, 3, SIZE, SIZE) * 2 - 1
    module = DiamondPolicyStepExport(agent.actor_critic)
    hidden = agent.actor_critic.init_hidden(1, torch.device("cpu"))
    with torch.no_grad():
        eager, _, _ = agent.actor_critic(frames.transpose(0, 1), hidden)
        h, c = hidden
        for t in range(3):
            logits, h, c = module(frames[t], h, c)
            torch.testing.assert_close(logits, eager[:, t])


def test_onnx_export_matches_eager(agent, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    path = export_any(
        agent,
        tmp_path / "dream_step.onnx",
        input_names=["obs_history", "actions", "noise"],
        output_names=["frame"],
        dynamo=False,
    )
    obs = torch.rand(1, L, 3, SIZE, SIZE) * 2 - 1
    actions = torch.tensor([[3, 2, 1, 0]])
    noise = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        eager = DiamondDreamStepExport(agent)(obs, actions, noise).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = session.run(
        None,
        {
            "obs_history": obs.numpy(),
            "actions": actions.numpy(),
            "noise": noise.numpy(),
        },
    )
    assert abs(out - eager).max() < 1e-4
