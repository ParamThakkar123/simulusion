"""The World Models V -> M -> C pipeline script, end to end on Pendulum."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from simulusion.training import train_world_model as twm  # noqa: E402

pytestmark = pytest.mark.timeout(180)


def test_generate_rollouts_records_the_frames_the_controller_sees(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    with warnings.catch_warnings():
        # Not the grey-band stand-in GymImageEnv uses when nothing renders.
        warnings.filterwarnings("error", message=".*grey bands.*")
        twm.generate_rollouts(str(data), "Pendulum-v1", 2, seq_len=12, num_workers=1)
    files = sorted(data.glob("*.npz"))
    assert len(files) == 2
    rollout = np.load(files[0])
    # Pendulum's observations are 3 numbers; the rollouts hold rendered frames.
    assert rollout["observations"].shape == (12, 64, 64, 3)
    assert rollout["observations"].dtype == np.uint8
    assert rollout["observations"].std() > 0
    assert rollout["actions"].shape == (12, 1)
    assert rollout["rewards"].shape == rollout["terminals"].shape == (12,)

    # Only the missing ones are generated, and none once there are enough.
    twm.generate_rollouts(str(data), "Pendulum-v1", 3, seq_len=12, num_workers=4)
    assert len(list(data.glob("*.npz"))) == 3
    twm.generate_rollouts(str(data), "Pendulum-v1", 3, seq_len=12, num_workers=1)
    assert len(list(data.glob("*.npz"))) == 3


def test_discrete_actions_are_recorded_as_plus_minus_one(tmp_path: Path) -> None:
    data = tmp_path / "data"
    twm.generate_rollouts(str(data), "CartPole-v1", 1, seq_len=12, num_workers=1)
    files = list(data.glob("*.npz"))
    if not files:  # a random CartPole episode can end before 10 steps
        pytest.skip("the random episode was too short to keep")
    actions = np.load(files[0])["actions"]
    assert actions.shape[1] == twm.env_action_size("CartPole-v1") == 2
    # +1 for the action taken, -1 for the others: what GymImageEnv decodes.
    np.testing.assert_array_equal(
        np.sort(actions, axis=1), [[-1.0, 1.0]] * len(actions)
    )


def test_pipeline_trains_v_m_c_and_plays_them(tmp_path: Path) -> None:
    pytest.importorskip("cma")
    data, logdir = tmp_path / "data", tmp_path / "results"
    twm.generate_rollouts(str(data), "Pendulum-v1", 4, seq_len=12, num_workers=1)
    twm.main(
        [
            "--env=Pendulum-v1",
            f"--data_dir={data}",
            f"--logdir={logdir}",
            "--device=cpu",
            "--num_rollouts=4",
            "--latent_size=8",
            "--rnn_hidden=16",
            "--gmm_components=2",
            "--seq_len=4",
            "--vae_epochs=1",
            "--vae_batch_size=4",
            "--rnn_epochs=1",
            "--rnn_batch_size=2",
            "--ctrl_pop_size=4",
            "--ctrl_samples=1",
            "--ctrl_time_limit=3",
            # Any return passes: CMA-ES stops after its first generation.
            "--ctrl_target=-1e9",
            "--no_amp",
        ]
    )
    for stage in ("vae", "mdrnn", "ctrl"):
        assert (logdir / stage / "best.tar").exists(), stage
    ctrl = torch.load(logdir / "ctrl" / "best.tar", weights_only=True)
    assert ctrl["state_dict"]["fc.weight"].shape == (1, 8 + 16)

    returns = twm.test_trained_model(
        str(logdir), "Pendulum-v1", action_size=1, num_episodes=2, time_limit=5
    )
    assert len(returns) == 2 and all(np.isfinite(returns))
    with pytest.raises(ValueError, match="1 actions, not 3"):
        twm.test_trained_model(str(logdir), "Pendulum-v1", action_size=3)


def test_trained_model_reports_missing_checkpoints(tmp_path: Path, capsys) -> None:
    assert twm.test_trained_model(str(tmp_path), "Pendulum-v1") == []
    assert "Missing:" in capsys.readouterr().out


def test_carracing_script_defaults_the_environment(monkeypatch) -> None:
    from simulusion.training import train_world_model_carracing as carracing

    seen: list[list[str]] = []
    monkeypatch.setattr(carracing, "_main", lambda argv: seen.append(list(argv)))
    carracing.main(["--stage", "vae"])
    carracing.main(["--env=Pendulum-v1", "--test"])
    carracing.main(["--env", "BipedalWalker-v3"])
    assert seen == [
        ["--env", "CarRacing-v3", "--stage", "vae"],
        ["--env=Pendulum-v1", "--test"],
        ["--env", "BipedalWalker-v3"],
    ]
