"""The older per-family entrypoints, called as existing scripts call them.

Each now runs on the shared Trainer through simulusion.training.legacy; these
tests pin what callers rely on: signatures, return values and the files each
one has always written, next to the trainer's own run files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simulusion.training.legacy import METRICS_NAME, RUN_CONFIG_NAME

# End-to-end training runs: each builds a model, trains it for a few steps,
# checkpoints, resumes and exports. On a CPU runner that takes longer than the
# suite-wide --timeout=30 allows unit tests.
pytestmark = pytest.mark.timeout(180)


def _trainer_files(run_dir: Path) -> None:
    """The trainer's files, under names that never clash with a family's own."""
    assert (run_dir / "run.json").exists()
    assert (run_dir / RUN_CONFIG_NAME).exists()
    assert (run_dir / METRICS_NAME).exists()
    assert any((run_dir / "checkpoints").glob("step_*"))


# ------------------------------------------------------------------ Dreamer


def test_dreamer_agent_train(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from simulusion.configs.dreamer_config import DreamerConfig
    from simulusion.models.dreamer import DreamerAgent

    config = DreamerConfig()
    for key, value in dict(
        env="Pendulum-v1",
        env_backend="gym",
        seed_steps=40,
        action_repeat=2,
        collect_steps=20,
        update_steps=1,
        batch_size=2,
        train_seq_len=4,
        imagine_horizon=2,
        buffer_size=500,
        obs_embed_size=32,
        num_units=16,
        deter_size=16,
        stoch_size=4,
        test_interval=40,
        test_episodes=1,
        checkpoint_interval=40,
        log_video_freq=-1,
        use_amp=False,
        perf_defaults=False,
        enable_wandb=False,
        logdir=str(tmp_path / "dreamer"),
    ).items():
        setattr(config, key, value)
    agent = DreamerAgent(config)
    assert agent.train(total_steps=80) is None

    logdir = Path(agent.logdir)
    _trainer_files(logdir)
    ckpts = sorted(p.name for p in (logdir / "ckpts").glob("*_ckpt.pt"))
    # A checkpoint each time progress crosses a checkpoint_interval multiple,
    # named by the step it was written at, plus the final one.
    assert ckpts == ["60_ckpt.pt", "80_ckpt.pt"]
    # The family's own config.yaml is untouched by the trainer.
    assert "env_backend" in (logdir / "config.yaml").read_text()
    # Metrics still reach the agent's own logger.
    rows = [json.loads(x) for x in (logdir / "metrics.jsonl").read_text().splitlines()]
    assert any("model_loss" in r for r in rows) and any(
        "eval_avg_reward" in r for r in rows
    )
    # The agent keeps working after training: evaluate() uses its test env.
    rewards, _, _ = agent.dreamer.evaluate(agent.test_env, 1)
    assert len(rewards) == 1


# --------------------------------------------------------------------- IRIS


def test_iris_trainer_train(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from simulusion.configs.iris_config import IRISConfig
    from simulusion.training.train_iris import IRISTrainer

    from tests.training.tasks.test_iris_tasks import TINY, FakeAtari

    config = IRISConfig()
    for key, value in {
        **TINY,
        "collection_epochs": 10,
        "env_steps_per_epoch": 20,
        "training_steps_per_epoch": 1,
        "transformer_steps_per_epoch": 1,
        "actor_critic_steps_per_epoch": 1,
        "start_autoencoder_after": 0,
        "start_transformer_after": 1,
        "start_actor_critic_after": 1,
        "eval_episodes": 1,
        "checkpoint_interval": 2,
    }.items():
        setattr(config, key, value)
    trainer = IRISTrainer(
        game="Fake/Game-v0", device="cpu", config=config, env=FakeAtari()
    )
    save_dir = tmp_path / "iris"
    metrics = trainer.train(total_epochs=3, eval_interval=2, save_dir=str(save_dir))

    assert len(metrics["env_steps"]) == 3 and "recon_loss" in metrics
    names = {p.name for p in save_dir.iterdir()}
    assert {"best_Game-v0.pt", "final_Game-v0.pt", "checkpoint_2.pt"} <= names
    assert isinstance(trainer.final_eval_return, float)
    _trainer_files(save_dir)


# ------------------------------------------------------------------ DIAMOND


def test_diamond_agent_train_and_early_stopping(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from simulusion.configs.diamond_config import DiamondConfig
    from simulusion.training.train_diamond import DiamondAgent

    from tests.training.tasks.test_diamond_tasks import TINY, FakeDiamondEnv

    config = DiamondConfig(
        **{k: v for k, v in TINY.items()},
        device="cpu",
        num_epochs=20,
        eval_interval=1,
        save_interval=2,
        log_interval=1,
        early_stopping=True,
        patience=0,
        checkpoint_dir=str(tmp_path / "diamond"),
    )
    agent = DiamondAgent(config, env=FakeDiamondEnv())
    agent.train()
    files = {p.name for p in (tmp_path / "diamond").iterdir()}
    assert "checkpoint_2.pt" in files
    _trainer_files(tmp_path / "diamond")
    # The fake env's reward never changes under evaluation, so early
    # stopping ends the run long before 20 epochs.
    done = json.loads(
        next(
            (tmp_path / "diamond" / "checkpoints").glob("step_*/checkpoint.json")
        ).read_text()
    )
    assert done["progress"] < 20


# ------------------------------------------------------------------- PlaNet


def test_planet_train_and_cli(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("gymnasium")
    from simulusion.models.planet import Planet
    from simulusion.training.train_planet import main

    planet = Planet(
        "Pendulum-v1",
        device=torch.device("cpu"),
        state_size=16,
        latent_size=4,
        memory_size=10,
        policy_cfg={
            "planning_horizon": 2,
            "num_candidates": 8,
            "num_iterations": 1,
            "top_candidates": 2,
        },
        headless=True,
        max_episode_steps=12,
    )
    out = planet.train(
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        H=6,
        save_every=1,
        results_dir=str(tmp_path / "planet"),
    )
    assert out == str(tmp_path / "planet")
    names = {p.name for p in (tmp_path / "planet").iterdir()}
    assert {"ckpt_1.pth", "ckpt_2.pth"} <= names
    assert any(n.startswith("vid_1") for n in names)
    _trainer_files(tmp_path / "planet")

    # The CLI plans with PlaNet's full 1000-candidate CEM; a small planner
    # keeps this a test of its wiring rather than of planning speed.
    import simulusion.models.planet as planet_module

    class SmallPlanet(Planet):
        def __init__(self, *args, **kwargs) -> None:
            kwargs.update(
                state_size=16,
                latent_size=4,
                policy_cfg={
                    "planning_horizon": 2,
                    "num_candidates": 8,
                    "num_iterations": 1,
                    "top_candidates": 2,
                },
            )
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(planet_module, "Planet", SmallPlanet)
    main(
        [
            "--env",
            "Pendulum-v1",
            "--epochs",
            "1",
            "--iters",
            "1",
            "--max-episode-steps",
            "55",
            "--outdir",
            str(tmp_path / "cli"),
            "--device",
            "cpu",
        ]
    )
    assert (tmp_path / "cli" / "ckpt_1.pth").exists()


# -------------------------------------------------------------------- Genie


def test_genie_trainer_train(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    from simulusion.training.train_genie import GenieConfig, create_genie_trainer

    config = GenieConfig(
        num_frames=4,
        image_size=16,
        tokenizer_vocab_size=16,
        tokenizer_embedding_dim=8,
        tokenizer_encoder_dim=16,
        tokenizer_decoder_dim=16,
        tokenizer_encoder_depth=1,
        tokenizer_decoder_depth=1,
        action_encoder_dim=16,
        action_decoder_dim=16,
        action_encoder_depth=1,
        dynamics_dim=16,
        dynamics_depth=1,
        dynamics_num_heads=2,
        warmup_steps=1,
        max_steps=4,
        early_stopping=True,
        patience=0,
    )
    trainer, _ = create_genie_trainer(config, device=torch.device("cpu"))
    clips = torch.rand(6, 3, 4, 16, 16)
    loader = DataLoader(clips, batch_size=2)
    trainer.train(
        loader,
        val_dataloader=DataLoader(clips[:2], batch_size=2),
        num_steps=4,
        log_interval=1,
        val_interval=1,
        checkpoint_dir=str(tmp_path / "genie"),
        checkpoint_interval=2,
    )
    assert trainer.global_step >= 2
    names = {p.name for p in (tmp_path / "genie").iterdir()}
    assert any(n.startswith("genie_step_") for n in names)
    _trainer_files(tmp_path / "genie")
    with pytest.raises(ValueError, match="val_dataloader"):
        trainer.train(loader, num_steps=1)


# ---------------------------------------------------------------------- DiT


def test_dit_fit(tmp_path: Path) -> None:
    from simulusion.models.diffusion.DiT import DiT

    from tests.training.tasks.test_offline_tasks import _image_folder

    root = _image_folder(tmp_path / "imgs")
    workdir = tmp_path / "dit"
    DiT.fit(
        epochs=2,
        dataset="imagefolder",
        batch_size=4,
        img_size=32,
        patch=8,
        width=32,
        depth=1,
        heads=2,
        timesteps=10,
        workdir=str(workdir),
        root_path=str(root),
        image_folder="train",
        num_workers=0,
        val_split=0.25,
        early_stopping=True,
        patience=5,
        checkpoint_every=1,
    )
    names = {p.name for p in workdir.iterdir()}
    assert {"dit_model.pth", "dit_model_epoch1.pth", "generated_samples.png"} <= names
    _trainer_files(workdir)
    assert (workdir / "checkpoints" / "best" / "checkpoint.json").exists()


# --------------------------------------------------------------------- JEPA


def test_jepa_agent_train_and_resume(tmp_path: Path) -> None:
    from simulusion.configs.jepa_config import JEPAConfig
    from simulusion.models.jepa_agent import JEPAAgent

    from tests.training.tasks.test_offline_tasks import _image_folder

    root = _image_folder(tmp_path / "imgs", n=16)
    config = JEPAConfig()
    for key, value in dict(
        dataset="imagefolder",
        root_path=str(root),
        image_folder="train",
        model_name="vit_tiny",
        crop_size=64,
        patch_size=8,
        pred_depth=1,
        pred_emb_dim=48,
        batch_size=4,
        num_workers=0,
        pin_mem=False,
        min_keep=2,
        epochs=2,
        warmup=1,
        write_tag="run",
        folder=str(tmp_path / "jepa"),
    ).items():
        setattr(config, key, value)
    JEPAAgent(config).train()
    latest = tmp_path / "jepa" / "run-latest.pth.tar"
    state = torch.load(latest, weights_only=True)
    assert state["epoch"] == 2 and {
        "encoder",
        "predictor",
        "target_encoder",
        "opt",
    } <= set(state)
    _trainer_files(tmp_path / "jepa")

    # load_checkpoint resumes from it, as train_jepa did.
    config.load_checkpoint = True
    config.epochs = 3
    JEPAAgent(config).train()
    assert torch.load(latest, weights_only=True)["epoch"] == 3


# ------------------------------------------------------------- World Models


def _rollouts(root: Path, count: int = 4, length: int = 8, actions: int = 1) -> Path:
    """World Models' rollout files: per-step observations/actions/rewards/terminals."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    for i in range(count):
        np.savez(
            root / f"rollout_{i}.npz",
            observations=rng.integers(0, 255, (length, 64, 64, 3), dtype=np.uint8),
            actions=rng.uniform(-1, 1, (length, actions)).astype(np.float32),
            rewards=rng.normal(size=length).astype(np.float32),
            terminals=np.r_[np.zeros(length - 1), 1.0].astype(np.float32),
        )
    return root


def test_world_models_v_m_c_entrypoints(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("cma")
    from simulusion.configs.wm_config import (
        WMControllerConfig,
        WMMDNRNNConfig,
        WMVAEConfig,
    )
    from simulusion.training.train_controller import train_controller
    from simulusion.training.train_convvae import train_convae
    from simulusion.training.train_mdn_rnn import train_mdn_rnn

    data = _rollouts(tmp_path / "data")
    logdir = str(tmp_path / "results")
    vae_config = WMVAEConfig(
        height=64,
        width=64,
        latent_size=8,
        num_epochs=2,
        train_batch_size=4,
        data_dir=str(data),
        logdir=logdir,
        sample_interval=1,
    )
    train_convae(vae_config)
    vae_dir = tmp_path / "results" / "vae"
    assert {"checkpoint.tar", "best.tar", "convae_final.pth"} <= {
        p.name for p in vae_dir.iterdir()
    }
    assert (vae_dir / "samples" / "sample_epoch_2.png").exists()
    best = torch.load(vae_dir / "best.tar", weights_only=True)
    assert {"epoch", "state_dict", "precision", "optimizer", "scheduler"} <= set(best)
    _trainer_files(vae_dir)

    mdrnn_config = WMMDNRNNConfig(
        latent_size=8,
        action_size=1,
        hidden_size=16,
        gmm_components=2,
        device="cpu",
        batch_size=2,
        seq_len=4,
        num_epochs=1,
        data_dir=str(data),
        logdir=logdir,
    )
    train_mdn_rnn(vae_config, mdrnn_config, use_amp=False)
    rnn_dir = tmp_path / "results" / "mdrnn"
    assert {"checkpoint.tar", "best.tar", "mdrnn_final.pth"} <= {
        p.name for p in rnn_dir.iterdir()
    }

    ctrl_config = WMControllerConfig(
        env_name="Pendulum-v1",
        pop_size=4,
        n_samples=1,
        time_limit=3,
        # Any return passes: the search stops after its first generation.
        target_return=-1e9,
        logdir=logdir,
    )
    train_controller(ctrl_config)
    best_ctrl = torch.load(
        tmp_path / "results" / "ctrl" / "best.tar", weights_only=True
    )
    assert {"epoch", "reward", "state_dict"} <= set(best_ctrl)
    assert best_ctrl["state_dict"]["fc.weight"].shape == (1, 8 + 16)
    done = json.loads(
        next(
            (tmp_path / "results" / "ctrl" / "checkpoints").glob(
                "step_*/checkpoint.json"
            )
        ).read_text()
    )
    assert done["progress"] == 1


def test_episode_reads_per_step_rollouts(tmp_path: Path) -> None:
    from simulusion.data.episode import Episode

    path = _rollouts(tmp_path / "data", count=1, length=5) / "rollout_0.npz"
    ep = Episode.load(path)
    assert len(ep) == 5 and ep.actions.shape == (4, 1) and ep.rewards.shape == (4,)
    # The step that ended the episode moves onto the last transition kept.
    assert ep.terminated.tolist() == [False, False, False, True]
