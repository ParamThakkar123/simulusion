import pytest

import torch

from simulusion.models.diffusion.actor_critic import ActorCriticNetwork

pytestmark = [pytest.mark.integration]


def test_policy_hidden_state_updates_over_imagined_steps():
    """Ensure the policy LSTM hidden state updates across successive imagined steps.

    This test simulates an imagination-like loop by repeatedly calling
    ActorCriticNetwork.get_actions() with successive (random) observations and
    verifies that the returned LSTM hidden states change between steps.
    """
    device = torch.device("cpu")
    B = 2  # batch size
    C, H, W = 3, 64, 64
    steps = 5

    actor = ActorCriticNetwork(obs_channels=C, action_dim=6)
    actor.to(device)

    # initialize hidden state for the policy LSTM
    hidden = actor.init_hidden(B, device)

    # generate a sequence of random observations (simulate imagined frames)
    obs_seq = [torch.randn(B, C, H, W, device=device) for _ in range(steps)]

    hidden_states = []
    for obs in obs_seq:
        actions, hidden = actor.get_actions(obs, hidden, deterministic=False)
        # copy hidden h and c to CPU for comparison later
        h, c = hidden
        hidden_states.append((h.detach().cpu().clone(), c.detach().cpu().clone()))

    # ensure we have as many hidden states as steps
    assert len(hidden_states) == steps

    # check that at least one of the hidden tensors changed between consecutive steps
    changed = False
    for i in range(1, steps):
        h_prev, c_prev = hidden_states[i - 1]
        h_curr, c_curr = hidden_states[i]
        if not torch.allclose(h_prev, h_curr) or not torch.allclose(c_prev, c_curr):
            changed = True
            break

    assert changed, "Policy hidden state did not change across imagined steps"


if __name__ == "__main__":
    test_policy_hidden_state_updates_over_imagined_steps()
