import pytest

import torch

from simulusion.configs.diamond_config import DiamondConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_hidden_state_save_load_and_broadcast(tmp_path):
    from simulusion.envs.diamond_atari import make_diamond_atari_env

    try:
        make_diamond_atari_env(
            game="Breakout-v4", frameskip=4, max_noop=30, resize=(64, 64), seed=0
        )
    except Exception:
        pytest.skip("Atari environment (Breakout) not available")

    from simulusion.training.train_diamond import DiamondAgent

    cfg = DiamondConfig(preset="small")
    cfg.game = "Breakout-v4"
    cfg.device = "cpu"
    cfg.batch_size = 2
    cfg.burn_in_length = 2
    cfg.imagination_horizon = 2

    agent = DiamondAgent(cfg)

    # create deterministic hidden states (CPU) with batch dim = 1
    h = torch.arange(0, agent.actor_critic.get_hidden_size(), dtype=torch.float32).view(
        1, 1, -1
    )
    c = (h + 1.0).clone()
    agent.last_policy_hidden = (h.clone(), c.clone())
    agent.last_reward_hidden = (h.clone(), c.clone())

    # save checkpoint
    ckpt = tmp_path / "hidden_ckpt.pt"
    # Path-like is accepted by save_checkpoint
    agent.save_checkpoint(ckpt)

    # load into fresh agent
    agent2 = DiamondAgent(cfg)
    agent2.load_checkpoint(str(ckpt))

    # ensure tensors restored and on cpu
    assert agent2.last_policy_hidden is not None
    h2, c2 = agent2.last_policy_hidden
    assert h2.device.type == "cpu"
    assert torch.allclose(h2.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0))

    # Now test broadcasting behavior when used with a larger batch. We'll
    # monkeypatch _imagine_trajectory to capture the policy_hidden passed in
    captured = {}

    def fake_imagine(obs_history, action_history, reward_hidden, policy_hidden=None):
        # capture the policy_hidden passed in and return minimal fake outputs
        captured["policy_hidden"] = policy_hidden
        B = obs_history.shape[0]
        H = cfg.imagination_horizon
        C = 3
        S = cfg.obs_size
        # return zeros with expected shapes: obs_imag [B, H, C, S, S],
        # rewards [B, H], dones [B, H], policy_actions [B, H]
        obs_imag = torch.zeros((B, H, C, S, S), device=agent2.device)
        rewards = torch.zeros((B, H), device=agent2.device)
        dones = torch.zeros((B, H), dtype=torch.bool, device=agent2.device)
        policy_actions = torch.zeros((B, H), dtype=torch.long, device=agent2.device)
        return obs_imag, rewards, dones, policy_actions, reward_hidden

    agent2._imagine_trajectory = fake_imagine

    # build a fake batch with batch size = cfg.batch_size (2)
    B = cfg.batch_size
    seq_T = cfg.burn_in_length + cfg.imagination_horizon
    obs_seq = torch.zeros(
        (B, seq_T, 3, cfg.obs_size, cfg.obs_size), device=agent2.device
    )
    action_seq = torch.zeros((B, seq_T), dtype=torch.long, device=agent2.device)
    batch = {"obs_seq": obs_seq, "action_seq": action_seq}

    # run the actor-critic update which will compute policy_hidden_init and
    # call our fake_imagine capturing the broadcasted hidden state
    agent2._update_actor_critic(batch)

    assert "policy_hidden" in captured
    ph = captured["policy_hidden"]
    # Expect hidden to have shape (num_layers=1, B, hidden_size)
    assert ph is not None
    assert ph[0].shape[1] == B
    # original saved vector should appear in each batch slot
    for b in range(B):
        assert torch.allclose(ph[0][:, b, :], h.squeeze(0))
