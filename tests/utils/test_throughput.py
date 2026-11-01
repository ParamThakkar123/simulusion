"""Tests for the throughput meter."""

import time

import pytest

torch = pytest.importorskip("torch")

from simulusion.utils.throughput import (  # noqa: E402
    ThroughputMeter,
    measure_steps,
    tensor_nbytes,
)


class TestTensorNbytes:
    def test_counts_dtype_width(self):
        assert tensor_nbytes(torch.zeros(10, dtype=torch.uint8)) == 10
        assert tensor_nbytes(torch.zeros(10, dtype=torch.float32)) == 40

    def test_charges_a_view_for_its_own_elements_not_the_base(self):
        base = torch.zeros(1000, dtype=torch.float32)
        assert tensor_nbytes(base[:10]) == 40

    def test_walks_containers_and_ignores_non_tensors(self):
        payload = {"a": torch.zeros(4, dtype=torch.uint8), "b": ["x", 3, None]}
        assert tensor_nbytes(payload, [torch.zeros(2, dtype=torch.uint8)]) == 6

    def test_uint8_batch_is_a_quarter_of_the_float32_one(self):
        # The property the observation-transfer path relies on.
        shape = (4, 3, 8, 8)
        assert tensor_nbytes(
            torch.zeros(shape, dtype=torch.float32)
        ) == 4 * tensor_nbytes(torch.zeros(shape, dtype=torch.uint8))


class TestThroughputMeter:
    def test_counts_steps_and_bytes(self):
        meter = ThroughputMeter()
        for _ in range(3):
            meter.record_transfer(torch.zeros(256, dtype=torch.uint8))
            meter.step()
        stats = meter.stats
        assert stats["steps"] == 3
        assert meter.total_bytes == 768
        assert stats["mib_to_device_per_step"] == pytest.approx(256 / 1024**2)

    def test_pending_bytes_are_attributed_to_one_step_only(self):
        meter = ThroughputMeter()
        meter.record_transfer(torch.zeros(100, dtype=torch.uint8))
        meter.step()
        meter.step()  # nothing transferred this step
        assert meter.total_bytes == 100
        assert meter._bytes[-1] == 0

    def test_rate_reflects_elapsed_time(self):
        meter = ThroughputMeter()
        for _ in range(2):
            time.sleep(0.05)
            meter.step()
        assert 0 < meter.stats["recent_steps_per_s"] < 100
        assert meter.stats["ms_per_step"] >= 50

    def test_reset_clears_history(self):
        meter = ThroughputMeter()
        meter.record_transfer(torch.zeros(64, dtype=torch.uint8))
        meter.step()
        meter.reset()
        assert meter.total_steps == 0
        assert meter.total_bytes == 0
        assert meter.stats["mib_to_device_per_step"] == 0.0

    def test_stats_are_safe_before_any_step(self):
        stats = ThroughputMeter().stats
        assert stats["steps"] == 0
        assert stats["recent_steps_per_s"] == 0.0
        assert stats["ms_per_step"] == 0.0

    def test_summary_is_a_single_line(self):
        meter = ThroughputMeter()
        meter.step()
        assert "\n" not in meter.summary()
        assert "steps/s" in meter.summary()


class TestMeasureSteps:
    def test_warmup_steps_are_excluded(self):
        calls = []
        stats = measure_steps(lambda: calls.append(1), iterations=5, warmup=3)
        assert len(calls) == 8  # warmup + measured
        assert stats["steps"] == 5
