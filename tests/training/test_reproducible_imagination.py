import pytest

import torch
import numpy as np

from simulusion.configs.diamond_config import DiamondConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def set_seeds(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_reproducible_imagination(tmp_path):
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
    cfg.num_epochs = 0
    cfg.num_sampling_steps = 3
    cfg.imagination_horizon = 4
    cfg.burn_in_length = 4
    cfg.batch_size = 2

    agent = DiamondAgent(cfg)

    # populate replay buffer with random but deterministic frames
    for i in range(32):
        frame = np.random.randint(
            0, 256, (cfg.obs_size, cfg.obs_size, 3), dtype=np.uint8
        )
        agent.replay_buffer.add(frame, 0, 0.0, False, frame)

    # sample a dataset batch to obtain burn-in obs and actions
    dataset = agent.replay_buffer.sample_sequence(
        cfg.batch_size,
        cfg.burn_in_length + cfg.imagination_horizon,
        burn_in=cfg.burn_in_length,
    )

    # ReplayBuffer.sample_sequence returns keys: obs, actions, rewards, dones, next_obs
    obs_seq = dataset["obs"]  # [B, T, C, H, W]
    action_seq = dataset["actions"]

    # pick a small batch B=cfg.batch_size for imagination
    obs_history = obs_seq[:, : cfg.burn_in_length].to(agent.device)
    action_history = action_seq[:, : cfg.burn_in_length].to(agent.device)

    # set deterministic seeds and run imagination once
    set_seeds(1234)
    obs_imag_1, rewards_1, dones_1, policy_actions_1, _ = agent._imagine_trajectory(
        obs_history,
        action_history,
        agent.reward_model.init_hidden(cfg.batch_size, agent.device),
        policy_hidden=None,
    )

    # save checkpoint into the temporary directory (torch + separate np/npy files)
    ckpt = tmp_path / "ckpt.pt"
    # Pass a Path-like object (string accepted) to the new save_checkpoint API
    # Save checkpoint (Path-like supported)
    agent.save_checkpoint(ckpt)

    # reload into a fresh agent instance
    agent2 = DiamondAgent(cfg)
    agent2.load_checkpoint(str(ckpt))

    # ensure replay buffer and obs history restored
    assert len(agent2.replay_buffer) > 0

    # re-run with same seed and inputs
    set_seeds(1234)
    obs_imag_2, rewards_2, dones_2, policy_actions_2, _ = agent2._imagine_trajectory(
        obs_history,
        action_history,
        agent2.reward_model.init_hidden(cfg.batch_size, agent2.device),
        policy_hidden=None,
    )

    # compare imagined observations
    assert torch.allclose(obs_imag_1, obs_imag_2)
    assert torch.equal(rewards_1, rewards_2)
    assert torch.equal(dones_1, dones_2)
    assert torch.equal(policy_actions_1, policy_actions_2)


if __name__ == "__main__":
    test_reproducible_imagination()
