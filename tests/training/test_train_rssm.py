"""The standalone PlaNet RSSM script: training, evaluation frames, checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("gymnasium")

from simulusion.training import train_rssm as tr  # noqa: E402

pytestmark = pytest.mark.timeout(180)

TINY = [
    "--env=Pendulum-v1",
    "--device=cpu",
    "--episodes=2",
    "--eval_every=1",
    "--checkpoint_every=2",
    "--iterations=1",
    "--batch_size=2",
    "--candidates=8",
    "--planning_horizon=2",
    "--max_episode_steps=6",
    "--warmup_episodes=2",
    "--test_episodes=1",
    "--state_size=8",
    "--latent_size=4",
    "--embedding_size=16",
]


def test_main_trains_evaluates_and_checkpoints(tmp_path: Path) -> None:
    data, out = tmp_path / "data", tmp_path / "out"
    args = TINY + [f"--data_dir={data}", f"--out_dir={out}"]
    tr.main(args)
    assert {"train_exp_replay.pth", "test_exp_replay.pth"} <= {
        p.name for p in data.iterdir()
    }
    written = {p.name for p in out.iterdir()}
    assert {"eps_1.png", "eps_2.png", "ckpt_2.pth"} <= written
    state = torch.load(out / "ckpt_2.pth", weights_only=True)
    assert any(key.startswith("encoder.") for key in state)

    # A second run reads the replay files it wrote instead of collecting again.
    stamps = {p.name: p.stat().st_mtime_ns for p in data.iterdir()}
    tr.main(args)
    assert {p.name: p.stat().st_mtime_ns for p in data.iterdir()} == stamps


def test_train_rssm_reports_one_loss_per_iteration() -> None:
    from simulusion.controller.rollout_generator import RolloutGenerator
    from simulusion.memory.planet_memory import Memory
    from simulusion.models.rssm import RecurrentStateSpaceModel
    from simulusion.utils.utils import TorchImageEnvWrapper

    torch.manual_seed(0)
    rssm = RecurrentStateSpaceModel(1, 8, 4, 16)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    env = TorchImageEnvWrapper("Pendulum-v1", tr.BIT_DEPTH)
    memory = Memory(4)
    memory.append(
        RolloutGenerator(env, "cpu", max_episode_steps=6).rollout_n(
            2, random_policy=True
        )
    )
    metrics = tr.train_rssm(
        memory, rssm, optimizer, record_grads=False, iterations=3, batch_size=2
    )
    assert len(metrics["kl_losses"]) == len(metrics["rec_losses"]) == 3
    assert all(loss == loss for loss in metrics["rec_losses"])  # not NaN
