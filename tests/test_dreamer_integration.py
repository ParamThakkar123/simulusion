import pytest
import torch
from simulusion.configs.dreamer_config import DreamerConfig


@pytest.mark.integration
def test_dreamer_pendulum_no_crash(tmp_path):
    pytest.importorskip("gymnasium")
    from simulusion.models.dreamer import DreamerAgent

    config = DreamerConfig()
    config.env = "Pendulum-v1"
    config.env_backend = "gym"
    # train_seq_len must be >= 2: with 1, `train_one_batch` returns before any
    # optimizer step, so this test used to pass without training anything.
    config.total_steps = 100
    config.seed_steps = 100
    config.action_repeat = 1
    config.collect_steps = 1
    config.update_steps = 1
    config.batch_size = 2
    config.train_seq_len = 4
    config.imagine_horizon = 2
    config.buffer_size = 1000
    config.obs_embed_size = 64
    config.num_units = 32
    config.deter_size = 32
    config.stoch_size = 8
    config.no_gpu = True
    config.test_interval = 100000
    config.checkpoint_interval = 100000
    config.log_video_freq = -1
    config.seed = 42
    config.logdir = str(tmp_path / "dreamer_integration_test")
    config.enable_wandb = False
    config.enable_tensorboard = False
    config.enable_console_metrics = False
    config.enable_jsonl = False

    agent = DreamerAgent(config)
    before = [p.detach().clone() for p in agent.dreamer.rssm.parameters()]
    agent.train(total_steps=100)

    after = list(agent.dreamer.rssm.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "world model parameters did not change: no training update ran"
    )
    assert agent.dreamer.metrics, "train_one_batch recorded no loss metrics"
    checkpoints = list((tmp_path / "dreamer_integration_test").rglob("*_ckpt.pt"))
    assert checkpoints, "training finished without writing a final checkpoint"
