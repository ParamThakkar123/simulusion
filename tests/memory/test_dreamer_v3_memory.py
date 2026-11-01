"""Tests for the DreamerV3 replay buffer."""

import numpy as np
import pytest

from simulusion.memory.dreamer_v3_memory import DreamerV3ReplayBuffer


def make_buffer(size=64, seq_len=4, batch_size=3, online_fraction=0.0):
    return DreamerV3ReplayBuffer(
        size=size,
        obs_shape=(3, 8, 8),
        action_size=2,
        seq_len=seq_len,
        batch_size=batch_size,
        online_fraction=online_fraction,
    )


def fill(buffer, steps, episode_length=None):
    for step in range(steps):
        done = episode_length is not None and (step + 1) % episode_length == 0
        buffer.add(
            {"image": np.full((3, 8, 8), step % 256, dtype=np.uint8)},
            np.array([step, -step], dtype=np.float32),
            float(step),
            done,
        )


class TestConstruction:
    def test_rejects_short_sequences(self):
        with pytest.raises(ValueError, match="seq_len"):
            make_buffer(seq_len=1)

    def test_rejects_invalid_online_fraction(self):
        with pytest.raises(ValueError, match="online_fraction"):
            DreamerV3ReplayBuffer(
                8, (3,), 1, seq_len=2, batch_size=1, online_fraction=1.5
            )

    def test_starts_empty(self):
        buffer = make_buffer()
        assert len(buffer) == 0
        assert not buffer.can_sample


class TestWriting:
    def test_tracks_steps_and_episodes(self):
        buffer = make_buffer()
        fill(buffer, 20, episode_length=5)
        assert buffer.steps == 20
        assert buffer.episodes == 4

    def test_accepts_plain_arrays_and_obs_dicts(self):
        buffer = make_buffer()
        buffer.add(np.zeros((3, 8, 8), np.uint8), np.zeros(2, np.float32), 0.0, False)
        buffer.add({"obs": np.zeros((3, 8, 8), np.uint8)}, np.zeros(2), 0.0, False)
        assert len(buffer) == 2

    def test_rejects_unknown_observation_keys(self):
        buffer = make_buffer()
        with pytest.raises(KeyError, match="image"):
            buffer.add({"pixels": np.zeros((3, 8, 8))}, np.zeros(2), 0.0, False)

    def test_first_transition_is_marked_as_an_episode_start(self):
        buffer = make_buffer()
        fill(buffer, 3)
        assert buffer.is_first[0] == 1.0
        assert buffer.is_first[1] == 0.0

    def test_is_first_marks_the_step_after_a_termination(self):
        buffer = make_buffer()
        fill(buffer, 10, episode_length=4)
        # Episode ends at index 3, so index 4 begins a new episode.
        assert buffer.is_terminal[3] == 1.0
        assert buffer.is_first[4] == 1.0
        assert buffer.is_first[3] == 0.0

    def test_time_limit_truncation_is_not_a_termination(self):
        buffer = make_buffer()
        buffer.add(
            {"image": np.zeros((3, 8, 8), np.uint8)},
            np.zeros(2, np.float32),
            1.0,
            done=True,
            is_terminal=False,
        )
        assert buffer.is_terminal[0] == 0.0
        # The episode still ends, so the next step is a fresh start.
        buffer.add({"image": np.zeros((3, 8, 8), np.uint8)}, np.zeros(2), 0.0, False)
        assert buffer.is_first[1] == 1.0

    def test_wraps_around_when_full(self):
        buffer = make_buffer(size=10)
        fill(buffer, 25)
        assert buffer.full
        assert len(buffer) == 10
        assert buffer.steps == 25


class TestSampling:
    def test_raises_before_enough_data(self):
        buffer = make_buffer(seq_len=8)
        fill(buffer, 4)
        with pytest.raises(RuntimeError, match="not"):
            buffer.sample()

    def test_batch_shapes(self):
        buffer = make_buffer(seq_len=5, batch_size=3)
        fill(buffer, 40)
        obs, actions, rewards, terminal, first = buffer.sample()
        assert obs.shape == (5, 3, 3, 8, 8)
        assert actions.shape == (5, 3, 2)
        assert rewards.shape == (5, 3)
        assert terminal.shape == (5, 3)
        assert first.shape == (5, 3)

    def test_sequences_are_contiguous_in_time(self):
        buffer = make_buffer(seq_len=4, batch_size=2)
        fill(buffer, 40)
        _, _, rewards, _, _ = buffer.sample()
        # Rewards were written as the step index, so each column must increase
        # by exactly one per timestep.
        deltas = np.diff(rewards, axis=0)
        assert np.all(deltas == 1.0)

    def test_sequences_may_span_episode_boundaries(self):
        # DreamerV3 relies on `is_first` rather than rejecting these windows.
        buffer = make_buffer(seq_len=6, batch_size=8)
        fill(buffer, 60, episode_length=3)
        _, _, _, _, first = buffer.sample()
        assert first.sum() > 0

    def test_samples_stay_within_written_data(self):
        buffer = make_buffer(size=64, seq_len=4, batch_size=6)
        fill(buffer, 20)
        _, _, rewards, _, _ = buffer.sample()
        assert rewards.max() < 20
        assert rewards.min() >= 0

    def test_full_buffer_never_reads_the_write_head_gap(self):
        buffer = make_buffer(size=16, seq_len=4, batch_size=8)
        fill(buffer, 100)
        _, _, rewards, _, _ = buffer.sample()
        # Contiguity is the observable signal that no window wrapped over the
        # boundary between newest and oldest data.
        assert np.all(np.diff(rewards, axis=0) == 1.0)


class TestOnlineQueue:
    def test_online_samples_are_preferred_when_available(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=4, online_fraction=1.0)
        fill(buffer, 40)
        _, _, rewards, _, _ = buffer.sample()
        # The queue hands out the oldest unconsumed non-overlapping windows,
        # which start at multiples of seq_len.
        assert set(rewards[0].tolist()) == {0.0, 4.0, 8.0, 12.0}

    def test_queue_is_consumed_across_batches(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=2, online_fraction=1.0)
        fill(buffer, 40)
        first_batch = buffer.sample()[2][0]
        second_batch = buffer.sample()[2][0]
        assert not set(first_batch.tolist()) & set(second_batch.tolist())

    def test_falls_back_to_uniform_when_the_queue_is_empty(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=6, online_fraction=1.0)
        fill(buffer, 12)
        obs, _, _, _, _ = buffer.sample()
        assert obs.shape[1] == 6

    def test_zero_online_fraction_uses_only_uniform_sampling(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=4, online_fraction=0.0)
        fill(buffer, 60)
        starts = {buffer.sample()[2][0].tolist()[0] for _ in range(10)}
        assert len(starts) > 1


class TestObservationDtypes:
    def test_float_observations_are_preserved(self):
        buffer = DreamerV3ReplayBuffer(
            size=32,
            obs_shape=(4,),
            action_size=1,
            seq_len=3,
            batch_size=2,
            obs_dtype=np.float32,
        )
        for step in range(10):
            buffer.add(
                np.full(4, step * 0.5, dtype=np.float32),
                np.zeros(1, np.float32),
                0.0,
                False,
            )
        obs, _, _, _, _ = buffer.sample()
        assert obs.dtype == np.float32
        assert np.any(obs % 1.0 != 0.0)
