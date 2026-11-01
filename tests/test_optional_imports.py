"""Base-install imports must not require gym, gymnasium, or wandb."""

from types import SimpleNamespace

import torch


def test_train_jepa_imports_without_wandb():
    from simulusion.training.train_jepa import build_loss_fn

    loss = build_loss_fn("l2")
    pred = torch.zeros(2, 4, 8)
    target = pred.clone()
    assert loss(pred, target).ndim == 0


def test_train_iris_helpers_import_without_gym():
    from simulusion.training.train_iris import (
        FREEWAY_COLLECT_TEMPERATURE,
        _action_size,
        default_collect_temperature,
    )

    assert (
        default_collect_temperature("ALE/Freeway-v5", 1.0)
        == FREEWAY_COLLECT_TEMPERATURE
    )
    assert _action_size(SimpleNamespace(n=6)) == 6
    assert _action_size(SimpleNamespace(shape=(2, 3), n=None)) == 6
