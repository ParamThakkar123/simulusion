from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any


from simulusion.configs.jepa_config import JEPAConfig
from simulusion.models.model_io import (
    apply_config_overrides,
    coerce_config,
    resolve_pretrained_file,
)
from simulusion.export import ExportableAgentMixin


class JEPAAgent(ExportableAgentMixin):
    """Convenience interface for configuring and launching JEPA training runs.

    Accepts a `JEPAConfig` plus keyword overrides, prepares output folders,
    and delegates execution to the JEPA training entrypoint.
    """

    def __init__(self, config: JEPAConfig | None = None, **kwargs: Any) -> None:
        self.cfg = coerce_config(JEPAConfig, config)
        for key, val in kwargs.items():
            if key == "logdir":
                self.cfg.folder = val
            elif hasattr(self.cfg, key):
                setattr(self.cfg, key, val)
            else:
                raise ValueError(f"Invalid argument: {key}")
        if not getattr(self.cfg, "write_tag", None):
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.cfg.write_tag = f"jepa_{ts}"
        os.makedirs(self.cfg.folder, exist_ok=True)
        self.cfg.to_yaml(Path(self.cfg.folder) / "config.yaml")

    @classmethod
    def from_config(
        cls,
        config: JEPAConfig | dict[str, Any] | str | Path | None = None,
        **overrides: Any,
    ) -> "JEPAAgent":
        """Build a JEPA agent from a config object, dict, YAML file, or YAML string."""

        return cls(apply_config_overrides(coerce_config(JEPAConfig, config), overrides))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: JEPAConfig | dict[str, Any] | str | Path | None = None,
        checkpoint_filename: str | None = None,
        config_filename: str = "config.yaml",
        repo_type: str | None = None,
        revision: str | None = None,
        **overrides: Any,
    ) -> "JEPAAgent":
        """Create a JEPA agent from local/HF Hub config and checkpoint metadata."""

        if config is None:
            config_path = resolve_pretrained_file(
                pretrained_model_name_or_path,
                (config_filename, "jepa_config.yaml", "config.yml"),
                repo_type=repo_type,
                revision=revision,
            )
            if config_path is None:
                raise FileNotFoundError(
                    "No config was provided and no config YAML was found beside "
                    f"{pretrained_model_name_or_path!r}."
                )
            args = JEPAConfig.from_yaml(config_path)
        else:
            args = coerce_config(JEPAConfig, config)

        checkpoint_candidates = (
            (checkpoint_filename,)
            if checkpoint_filename is not None
            else ("jepa-latest.pth.tar", "checkpoint.pth.tar", "model.pt")
        )
        checkpoint_path = resolve_pretrained_file(
            pretrained_model_name_or_path,
            checkpoint_candidates,
            repo_type=repo_type,
            revision=revision,
        )
        if checkpoint_path is not None:
            args.load_checkpoint = True
            args.read_checkpoint = str(checkpoint_path)
        return cls(apply_config_overrides(args, overrides))

    def parameter_count(self, trainable_only: bool = False) -> int:
        """JEPA models are constructed inside training, so no parameters are resident."""

        return 0

    def summary(self) -> dict[str, Any]:
        """Return the configured JEPA run metadata."""

        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "model_name": self.cfg.model_name,
            "config": self.cfg.to_dict(),
        }

    def train(self) -> None:
        """Train for ``cfg.epochs`` epochs.

        Runs on the shared :class:`~simulusion.training.Trainer` through the
        ``jepa`` task, in ``cfg.folder``: per-epoch checkpoints in
        ``train_jepa``'s format (``<write_tag>-latest.pth.tar``, and
        ``<write_tag>-ep<N>.pth.tar`` every 50 epochs), resuming from
        ``read_checkpoint`` when ``load_checkpoint`` is set, bf16 with
        ``use_bfloat16``, and per-epoch held-out early stopping with
        ``early_stopping``. W&B sweeps (``enable_sweep``) still run through
        ``train_jepa``.
        """
        if self.cfg.enable_sweep:
            from simulusion.training.train_jepa import main as train_jepa_main

            train_jepa_main(self.cfg)
            return

        from simulusion.training.legacy import fit_legacy
        from simulusion.training.tasks.jepa import JEPATask, JEPATaskConfig

        cfg = self.cfg
        early = bool(cfg.early_stopping)
        fit_legacy(
            JEPATask(
                JEPATaskConfig(model=cfg, val_batches=1 if early else 0),
                legacy_folder=cfg.folder,
            ),
            cfg.folder,
            precision="bf16" if cfg.use_bfloat16 else "fp32",
            log_every=1,
            eval_every=1 if early else 0,
            checkpoint_every=1,
            early_stopping_metric="val_loss" if early else None,
            early_stopping_mode="min",
            patience=cfg.patience,
            min_delta=cfg.min_delta,
        )
