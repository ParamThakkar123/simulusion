import torch
from simulusion.utils.jepa_utils import (
    trunc_normal_,
    repeat_interleave_batch,
    WarmupCosineSchedule,
    CosineWDSchedule,
    AverageMeter,
    grad_logger,
)


class TestJepaUtils:
    def test_trunc_normal_(self):
        tensor = torch.zeros(10, 10)
        result = trunc_normal_(tensor, mean=0.5, std=0.1, a=0.0, b=1.0)

        assert result.shape == (10, 10)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_trunc_normal_default_params(self):
        tensor = torch.zeros(5, 5)
        result = trunc_normal_(tensor)

        assert result.shape == (5, 5)
        assert result.min() >= -2.0
        assert result.max() <= 2.0

    def test_repeat_interleave_batch(self):
        x = torch.randn(12, 8)
        B = 4
        repeat = 3

        result = repeat_interleave_batch(x, B, repeat)

        assert result.shape == (36, 8)

    def test_repeat_interleave_batch_small(self):
        x = torch.randn(4, 2)
        B = 2
        repeat = 2

        result = repeat_interleave_batch(x, B, repeat)

        assert result.shape == (8, 2)


class TestWarmupCosineSchedule:
    def test_init(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = WarmupCosineSchedule(
            optimizer=optimizer, warmup_steps=100, start_lr=0.0, ref_lr=0.1, T_max=1000
        )

        assert scheduler.start_lr == 0.0
        assert scheduler.ref_lr == 0.1

    def test_step(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = WarmupCosineSchedule(
            optimizer=optimizer, warmup_steps=100, start_lr=0.0, ref_lr=0.1, T_max=1000
        )

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]

        assert lr_after >= lr_before

    def test_step_warmup_complete(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = WarmupCosineSchedule(
            optimizer=optimizer, warmup_steps=10, start_lr=0.0, ref_lr=0.1, T_max=1000
        )

        for _ in range(20):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        assert lr > 0.0

    def test_final_lr(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=10,
            start_lr=0.0,
            ref_lr=0.1,
            T_max=100,
            final_lr=0.001,
        )

        for _ in range(100):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        assert lr >= 0.0


class TestCosineWDSchedule:
    def test_init(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = CosineWDSchedule(
            optimizer=optimizer, ref_wd=0.1, T_max=1000, final_wd=0.0
        )

        assert scheduler.ref_wd == 0.1

    def test_step(self):
        optimizer = torch.optim.Adam([torch.zeros(1)], lr=0.001)

        scheduler = CosineWDSchedule(
            optimizer=optimizer, ref_wd=0.1, T_max=1000, final_wd=0.0
        )

        scheduler.step()
        wd_after = optimizer.param_groups[0]["weight_decay"]

        assert wd_after >= 0.0

    def test_step_with_wd_exclude(self):
        optimizer = torch.optim.Adam(
            [{"params": [torch.zeros(1)], "weight_decay": 0.1, "WD_exclude": True}]
        )

        scheduler = CosineWDSchedule(
            optimizer=optimizer, ref_wd=0.1, T_max=1000, final_wd=0.0
        )

        scheduler.step()

        assert optimizer.param_groups[0]["weight_decay"] == 0.1


class TestAverageMeter:
    def test_init(self):
        meter = AverageMeter()

        assert meter.val == 0
        assert meter.avg == 0

    def test_update(self):
        meter = AverageMeter()

        meter.update(5.0)

        assert meter.val == 5.0
        assert meter.avg == 5.0

    def test_update_multiple(self):
        meter = AverageMeter()

        meter.update(5.0, n=2)
        meter.update(10.0, n=2)

        assert meter.avg == 7.5

    def test_reset(self):
        meter = AverageMeter()

        meter.update(5.0)
        meter.reset()

        assert meter.val == 0
        assert meter.avg == 0


class TestGradLogger:
    def test_grad_logger(self):
        model = torch.nn.Linear(10, 5)

        x = torch.randn(2, 10)
        y = model(x)
        y.sum().backward()

        stats = grad_logger(model.named_parameters())

        assert stats.val >= 0.0

    def test_grad_logger_no_grad(self):
        model = torch.nn.Linear(10, 5)

        stats = grad_logger(model.named_parameters())

        assert stats.val == 0
        assert stats.first_layer == 0.0
        assert stats.last_layer == 0.0
