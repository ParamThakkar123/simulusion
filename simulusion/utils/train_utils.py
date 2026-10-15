"""Training utilities for World Models.

This module provides utility classes for training neural networks including
early stopping and learning rate scheduling.
"""

from typing import Any
from functools import partial
from torch.optim import Optimizer


class EarlyStopping:
    """Early stopping handler to stop training when validation metric stops improving.

    This class monitors a validation metric and stops training when no improvement
    is seen for a specified number of epochs (patience). This helps prevent
    overfitting and reduces unnecessary computation.

    Args:
        mode: One of 'min' or 'max'. In 'min' mode, training stops when the
            metric stops decreasing; in 'max' mode, when it stops increasing.
        patience: Number of epochs with no improvement after which to stop training.
        threshold: Minimum change to qualify as an improvement.
        threshold_mode: One of 'rel' or 'abs'. In 'rel' mode, dynamic threshold
            is relative to best value; in 'abs' mode, it's absolute.

    Attributes:
        stop: Property that returns True if training should stop.

    Example:
        >>> early_stopping = EarlyStopping(mode='min', patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     early_stopping.step(val_loss)
        ...     if early_stopping.stop:
        ...         print(f"Stopped at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        mode: str = "min",
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self) -> None:
        """Reset the internal state for a new training run."""
        self.best: float = self.mode_worse
        self.num_bad_epochs: int = 0

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Update early stopping state with new metric value.

        Args:
            metrics: Current epoch's metric value.
            epoch: Current epoch number. If None, auto-increments from last epoch.
        """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self) -> bool:
        """bool: True if training should stop due to no improvement."""
        return self.num_bad_epochs > self.patience

    def _cmp(
        self, mode: str, threshold_mode: str, threshold: float, a: float, best: float
    ) -> bool:
        """Compare two values based on mode and threshold settings."""
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = 1.0 + threshold
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """Initialize the comparison function."""
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        if mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")
        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing early stopping state.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("is_better",)
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary containing early stopping state.
        """
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)


class ReduceLROnPlateau:
    """Reduce learning rate when a metric stops improving.

    This scheduler reduces the learning rate by a factor when a validation
    metric stops improving for a specified number of epochs. This helps
    models converge better by reducing the step size as they approach
    optimal weights.

    Args:
        optimizer: The PyTorch optimizer to adjust.
        mode: One of 'min' or 'max'. In 'min' mode, lr is reduced when metric
            stops decreasing; in 'max' mode, when it stops increasing.
        factor: Factor by which to reduce the learning rate.
        patience: Number of epochs with no improvement after which to reduce lr.
        threshold: Minimum change to qualify as an improvement.
        threshold_mode: One of 'rel' or 'abs'.
        min_lr: Minimum learning rate to reduce to.
        eps: Minimum decay for lr.

    Attributes:
        lr: Current learning rates for each parameter group.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        >>> for epoch in range(100):
        ...     train_loss = train()
        ...     val_loss = validate()
        ...     scheduler.step(val_loss)
        ...     if scheduler.stop:
        ...         break
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        min_lr: float = 0,
        eps: float = 1e-8,
    ) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self) -> None:
        """Reset the internal state for a new training run."""
        self.best: float = self.mode_worse
        self.num_bad_epochs: int = 0

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Update learning rate based on metric value.

        Args:
            metrics: Current epoch's metric value.
            epoch: Current epoch number. If None, auto-increments from last epoch.
        """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self) -> None:
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    @property
    def lr(self) -> list:
        """list: Current learning rates for each parameter group."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def _cmp(
        self, mode: str, threshold_mode: str, threshold: float, a: float, best: float
    ) -> bool:
        """Compare two values based on mode and threshold settings."""
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon
        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = 1.0 + threshold
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """Initialize the comparison function."""
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")
        self.mode_worse: float
        if mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")
        self.is_better: Any = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("is_better",)
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state.
        """
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)
