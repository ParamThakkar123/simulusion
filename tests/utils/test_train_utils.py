"""Tests for training utility classes (EarlyStopping, ReduceLROnPlateau)."""

import torch
import pytest
from simulusion.utils.train_utils import EarlyStopping, ReduceLROnPlateau


class TestEarlyStopping:
    def test_min_mode_stops_when_no_improvement(self):
        es = EarlyStopping(mode="min", patience=3)
        for i in range(5):
            es.step(1.0 - i * 0.01)
        assert es.stop is False
        for i in range(4):
            es.step(0.97)
        assert es.stop is True

    def test_min_mode_does_not_stop_with_improvement(self):
        es = EarlyStopping(mode="min", patience=3)
        values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        for v in values:
            es.step(v)
        assert es.stop is False

    def test_max_mode_stops_when_no_improvement(self):
        es = EarlyStopping(mode="max", patience=2)
        es.step(0.5)
        es.step(0.6)
        es.step(0.61)
        assert es.stop is False
        es.step(0.6)
        es.step(0.6)
        es.step(0.6)
        assert es.stop is True

    def test_max_mode_does_not_stop_with_improvement(self):
        es = EarlyStopping(mode="max", patience=2)
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            es.step(v)
        assert es.stop is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="invalid")

    def test_invalid_threshold_mode_raises(self):
        with pytest.raises(ValueError):
            EarlyStopping(mode="min", threshold_mode="invalid")

    def test_state_dict_roundtrip(self):
        es1 = EarlyStopping(mode="min", patience=5)
        es1.step(1.0)
        es1.step(0.9)
        es1.step(0.95)
        state = es1.state_dict()

        es2 = EarlyStopping(mode="min", patience=5)
        es2.load_state_dict(state)
        assert es2.best == es1.best
        assert es2.num_bad_epochs == es1.num_bad_epochs
        assert es2.last_epoch == es1.last_epoch

    def test_auto_epoch_increment(self):
        es = EarlyStopping(mode="min", patience=3)
        es.step(1.0)
        assert es.last_epoch == 0
        es.step(0.9)
        assert es.last_epoch == 1


class TestReduceLROnPlateau:
    @pytest.fixture
    def optimizer(self):
        model = torch.nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def test_lr_reduction_on_plateau(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        for i in range(10):
            scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] < 0.09

    def test_lr_not_reduced_with_improvement(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        scheduler.step(1.0)
        scheduler.step(0.9)
        scheduler.step(0.8)
        scheduler.step(0.7)
        assert optimizer.param_groups[0]["lr"] == 0.1

    def test_lr_multiple_reductions(self, optimizer):
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
        )
        for i in range(10):
            scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] == 0.1 * (0.5 ** (9 // 2))

    def test_min_lr_clipping(self, optimizer):
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, min_lr=0.01
        )
        for i in range(20):
            scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] >= 0.01

    def test_max_mode(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        scheduler.step(0.5)
        scheduler.step(0.5)
        scheduler.step(0.5)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(0.5)
        after_lr = optimizer.param_groups[0]["lr"]
        assert after_lr < before_lr

    def test_lr_property(self, optimizer):
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        assert scheduler.lr == [0.1]

    def test_state_dict_roundtrip(self, optimizer):
        scheduler1 = ReduceLROnPlateau(optimizer, mode="min", patience=3)
        scheduler1.step(1.0)
        scheduler1.step(0.9)
        state = scheduler1.state_dict()

        optimizer2 = torch.optim.SGD(torch.nn.Linear(10, 10).parameters(), lr=0.1)
        scheduler2 = ReduceLROnPlateau(optimizer2, mode="min", patience=3)
        scheduler2.load_state_dict(state)
        assert scheduler2.best == scheduler1.best
        assert scheduler2.num_bad_epochs == scheduler1.num_bad_epochs

    def test_invalid_mode_raises(self, optimizer):
        with pytest.raises(ValueError):
            ReduceLROnPlateau(optimizer, mode="invalid")

    def test_invalid_threshold_mode_raises(self, optimizer):
        with pytest.raises(ValueError):
            ReduceLROnPlateau(optimizer, mode="min", threshold_mode="invalid")
