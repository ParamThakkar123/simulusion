"""`dreamer` (online agent) and `dreamer-wm` (world model from a dataset)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simulusion.training.run import run
from simulusion.world_model import load_world_model

from .conftest import write_episodes

# End-to-end training runs: each builds a model, trains it for a few steps,
# checkpoints, resumes and exports. On a CPU runner that takes longer than the
# suite-wide --timeout=30 allows unit tests.
pytestmark = pytest.mark.timeout(180)

TINY = [
    "model.batch_size=2",
    "model.train_seq_len=4",
    "model.imagine_horizon=2",
    "model.obs_embed_size=32",
    "model.num_units=16",
    "model.deter_size=16",
    "model.stoch_size=4",
    "model.perf_defaults=false",
    "model.use_amp=false",
    "model.enable_wandb=false",
]


@pytest.mark.parametrize("task", ["dreamer", "dreamer-v2"])
def test_dreamer_agent_collects_trains_resumes_and_exports(
    tmp_path: Path, trainer_overrides, task
) -> None:
    pytest.importorskip("gymnasium")
    args = [
        *trainer_overrides,
        *TINY,
        "model.env=Pendulum-v1",
        "model.env_backend=gym",
        "model.action_repeat=2",
        "model.seed_steps=40",
        "model.collect_steps=20",
        "model.update_steps=2",
        "model.buffer_size=500",
        "model.test_episodes=1",
        "trainer.max_steps=60",
    ]
    first = run(overrides=args, task=task)
    # 20 seed + 10 collected per round, in env steps after action repeat 2.
    assert first.progress == 60 and first.iterations == 1
    assert "eval_avg_reward" in first.metrics
    ckpt = first.checkpoint
    assert (ckpt / "task" / "replay.npz").exists()

    resumed = run(
        overrides=[*args, "trainer.max_steps=100", "trainer.resume=auto"], task=task
    )
    # Seed episodes are not collected again: the replay came back.
    assert resumed.progress == 100 and resumed.iterations == 3

    model = load_world_model(resumed.world_model_path, device="cpu")
    assert model.spec.family == "dreamer" and model.spec.action.kind == "continuous"
    state = model.encode(torch.rand(1, 2, *model.spec.obs_shape))
    out = model.rollout(state, torch.zeros(1, 3, model.spec.action.n))
    assert out.frames.shape[1] == 3 and out.rewards.shape == (1, 3)


@pytest.mark.parametrize(
    "actions", [3, ("continuous", 2)], ids=["discrete", "continuous"]
)
def test_dreamer_world_model_trains_from_episodes(
    tmp_path: Path, trainer_overrides, actions
) -> None:
    data = write_episodes(
        tmp_path / "eps", count=4, length=10, size=64, actions=actions
    )
    args = [
        *trainer_overrides,
        *TINY,
        f"data.path={data.as_posix()}",
        "data.batch_size=2",
        "data.val_fraction=0.25",
        "trainer.max_steps=3",
        "trainer.eval_every=3",
    ]
    result = run(overrides=args, task="dreamer-wm")
    assert result.progress == 3
    assert "val_model_loss" in result.metrics
    model = load_world_model(result.world_model_path, device="cpu")
    n = 3 if actions == 3 else 2
    assert model.spec.action.n == n
    state = model.encode(torch.rand(2, 3, 3, 64, 64))
    out = model.rollout(state, torch.zeros(2, 2, n))
    assert out.frames.shape == (2, 2, 3, 64, 64)

    resumed = run(
        overrides=[*args, "trainer.max_steps=5", "trainer.resume=auto"],
        task="dreamer-wm",
    )
    assert resumed.iterations == 5


def test_dreamer_world_model_batches_match_the_replay_layout(
    tmp_path: Path, trainer_overrides
) -> None:
    """Row t of a dataset batch holds obs_t and the transition that left it."""
    from simulusion.training.run import resolve_run_config
    from simulusion.training.trainer import Trainer

    data = write_episodes(
        tmp_path / "eps", count=2, length=6, size=64, actions=("continuous", 2)
    )
    cls, tcfg, cfg, _ = resolve_run_config(
        overrides=[
            *trainer_overrides,
            *TINY,
            f"data.path={data.as_posix()}",
            "data.batch_size=2",
        ],
        task="dreamer-wm",
    )
    task = cls(cfg)
    trainer = Trainer(tcfg)
    task.attach(trainer)
    task.setup(trainer)
    batch = {
        "frames": torch.zeros(2, 4, 3, 64, 64, dtype=torch.uint8),
        "actions": torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2),
        "rewards": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "terminated": torch.tensor([[False, False, True], [False, False, False]]),
        "truncated": torch.tensor([[False, False, False], [False, True, False]]),
    }
    obs, acs, rews, nonterms, continues = task._tensors(batch)
    assert obs.shape == (4, 2, 3, 64, 64)
    assert acs.shape == (4, 2, 2) and torch.equal(acs[1, 0], torch.tensor([2.0, 3.0]))
    assert rews[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 0.0]
    # A truncation is a boundary (nonterm 0) but not a termination (continue 1).
    assert nonterms[:, 1, 0].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert continues[:, 1, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert continues[:, 0, 0].tolist() == [1.0, 1.0, 0.0, 1.0]
