"""Tests for the DreamerV3 replay buffer."""

import numpy as np
import pytest
import torch

from simulusion.memory.dreamer_v3_memory import DreamerV3ReplayBuffer

OBS = (3, 8, 8)


def make_buffer(size=64, seq_len=4, batch_size=3, online_fraction=0.0, **kwargs):
    return DreamerV3ReplayBuffer(
        size=size,
        obs_shape=OBS,
        action_size=2,
        seq_len=seq_len,
        batch_size=batch_size,
        online_fraction=online_fraction,
        **kwargs,
    )


def obs(value):
    return {"image": np.full(OBS, value % 256, dtype=np.uint8)}


def fill(buffer, records, episode_length=None, terminal=True):
    """Write ``records`` records; record ``i`` has observation value ``i``.

    With ``episode_length``, each episode is that many records and its last
    record ends it (as a termination, or a truncation if not ``terminal``).
    """
    position = 0
    for step in range(records):
        last = episode_length is not None and position == episode_length - 1
        buffer.add(
            obs(step),
            np.array([step, -step], dtype=np.float32),
            float(step),
            is_first=position == 0,
            is_last=last,
            is_terminal=last and terminal,
        )
        position = 0 if last else position + 1


def contiguous(indices, size):
    return np.all(np.diff(indices, axis=0) % size == 1)


class TestConstruction:
    def test_rejects_short_sequences(self):
        with pytest.raises(ValueError, match="seq_len"):
            make_buffer(seq_len=1)

    def test_rejects_invalid_online_fraction(self):
        with pytest.raises(ValueError, match="online_fraction"):
            DreamerV3ReplayBuffer(
                8, (3,), 1, seq_len=2, batch_size=1, online_fraction=1.5
            )

    def test_latent_storage_needs_the_latent_shape(self):
        with pytest.raises(ValueError, match="latent"):
            make_buffer(deter_size=8)

    def test_starts_empty(self):
        buffer = make_buffer()
        assert len(buffer) == 0
        assert not buffer.can_sample

    def test_bytes_per_record_counts_latents(self):
        plain = DreamerV3ReplayBuffer.bytes_per_record(OBS, np.uint8, 2)
        with_latents = DreamerV3ReplayBuffer.bytes_per_record(
            OBS, np.uint8, 2, deter_size=16, latent_dim=4
        )
        assert with_latents - plain == 16 * 2 + 4


class TestWriting:
    def test_tracks_transitions_and_episodes(self):
        buffer = make_buffer()
        fill(buffer, 20, episode_length=5)
        # Each 5-record episode is 4 transitions (its first record has none).
        assert buffer.steps == 16
        assert buffer.episodes == 4
        assert buffer.records == 20

    def test_accepts_plain_arrays_and_obs_dicts(self):
        buffer = make_buffer()
        buffer.add(np.zeros(OBS, np.uint8), np.zeros(2), 0.0, is_first=True)
        buffer.add({"obs": np.zeros(OBS, np.uint8)}, np.zeros(2), 0.0, is_first=False)
        assert len(buffer) == 2

    def test_rejects_unknown_observation_keys(self):
        buffer = make_buffer()
        with pytest.raises(KeyError, match="image"):
            buffer.add({"pixels": np.zeros(OBS)}, np.zeros(2), 0.0, is_first=True)

    def test_first_record_must_start_an_episode(self):
        buffer = make_buffer()
        with pytest.raises(ValueError, match="is_first"):
            buffer.add(obs(0), np.zeros(2), 0.0, is_first=False)

    def test_a_terminal_must_also_be_last(self):
        buffer = make_buffer()
        with pytest.raises(ValueError, match="last"):
            buffer.add(obs(0), np.zeros(2), 0.0, is_first=True, is_terminal=True)

    def test_episode_starts_store_no_action_or_reward(self):
        buffer = make_buffer()
        buffer.add(obs(0), np.ones(2), 5.0, is_first=True)
        assert np.all(buffer.prev_actions[0] == 0.0)
        assert buffer.rewards[0] == 0.0

    def test_records_store_the_action_and_reward_that_led_to_them(self):
        buffer = make_buffer()
        fill(buffer, 3)
        assert buffer.rewards[1] == 1.0
        assert buffer.prev_actions[2].tolist() == [2.0, -2.0]

    def test_a_record_after_an_episode_end_must_be_first(self):
        buffer = make_buffer()
        fill(buffer, 4, episode_length=4)
        with pytest.raises(ValueError, match="is_first"):
            buffer.add(obs(4), np.zeros(2), 0.0, is_first=False)

    def test_terminal_and_truncation_both_end_the_episode(self):
        buffer = make_buffer()
        fill(buffer, 4, episode_length=4, terminal=False)
        assert buffer.is_last[3] == 1.0
        assert buffer.is_terminal[3] == 0.0

    def test_starting_a_new_episode_closes_the_open_one(self):
        buffer = make_buffer()
        fill(buffer, 3)
        buffer.add(obs(9), np.zeros(2), 0.0, is_first=True)
        assert buffer.is_last[2] == 1.0
        assert buffer.is_terminal[2] == 0.0
        assert buffer.episodes == 1

    def test_end_episode_marks_a_truncation_once(self):
        buffer = make_buffer()
        fill(buffer, 3)
        assert buffer.episode_open
        buffer.end_episode()
        buffer.end_episode()
        assert buffer.is_last[2] == 1.0
        assert buffer.episodes == 1
        assert not buffer.episode_open

    def test_wraps_around_when_full(self):
        buffer = make_buffer(size=10)
        fill(buffer, 25)
        assert buffer.full
        assert len(buffer) == 10


class TestSampling:
    def test_raises_before_enough_data(self):
        buffer = make_buffer(seq_len=8)
        fill(buffer, 4)
        with pytest.raises(RuntimeError, match="not"):
            buffer.sample()

    def test_batch_shapes(self):
        buffer = make_buffer(seq_len=5, batch_size=3)
        fill(buffer, 40)
        batch = buffer.sample()
        assert batch["observations"].shape == (5, 3, *OBS)
        assert batch["prev_actions"].shape == (5, 3, 2)
        for key in ("rewards", "is_first", "is_last", "is_terminal", "indices"):
            assert batch[key].shape == (5, 3)
        assert batch["init_valid"].shape == (3,)

    def test_sequences_are_contiguous_in_time(self):
        buffer = make_buffer(seq_len=4, batch_size=2)
        fill(buffer, 40)
        batch = buffer.sample()
        assert contiguous(batch["indices"], buffer.size)
        assert np.all(np.diff(batch["observations"][:, :, 0, 0, 0], axis=0) == 1)

    def test_sequences_may_span_episode_boundaries(self):
        buffer = make_buffer(seq_len=6, batch_size=8)
        fill(buffer, 60, episode_length=3)
        assert buffer.sample()["is_first"].sum() > 0

    def test_samples_stay_within_written_data(self):
        buffer = make_buffer(size=64, seq_len=4, batch_size=6)
        fill(buffer, 20)
        indices = buffer.sample()["indices"]
        assert indices.max() < 20

    def test_full_buffer_never_reads_across_the_write_head(self):
        buffer = make_buffer(size=16, seq_len=4, batch_size=8)
        fill(buffer, 100)
        for _ in range(20):
            values = buffer.sample()["observations"][:, :, 0, 0, 0].astype(int)
            # Observation values were written in order, so a window that
            # wrapped from the newest to the oldest record would jump back.
            assert np.all(np.diff(values, axis=0) == 1)


class TestOnlineQueue:
    def test_online_samples_fill_the_batch_first(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=4, online_fraction=1.0)
        fill(buffer, 40)
        starts = buffer.sample()["indices"][0]
        # The oldest unconsumed non-overlapping chunks start at multiples of 4.
        assert sorted(starts.tolist()) == [0, 4, 8, 12]

    def test_queue_is_consumed_across_batches(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=2, online_fraction=1.0)
        fill(buffer, 40)
        first_batch = set(buffer.sample()["indices"][0].tolist())
        second_batch = set(buffer.sample()["indices"][0].tolist())
        assert not first_batch & second_batch

    def test_falls_back_to_uniform_when_the_queue_is_empty(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=6, online_fraction=1.0)
        fill(buffer, 12)
        assert buffer.sample()["observations"].shape[1] == 6

    def test_overwritten_chunks_are_skipped(self):
        buffer = make_buffer(size=16, seq_len=4, batch_size=2, online_fraction=1.0)
        fill(buffer, 40)
        # Records 0..23 are gone; queued chunks pointing at them must not be
        # handed out, and the surviving chunks are complete windows.
        for _ in range(4):
            batch = buffer.sample()
            values = batch["observations"][0, :, 0, 0, 0].astype(int)
            assert np.all(values >= 24 % 256)
            assert contiguous(batch["indices"], buffer.size)

    def test_zero_online_fraction_uses_only_uniform_sampling(self):
        buffer = make_buffer(size=256, seq_len=4, batch_size=4, online_fraction=0.0)
        fill(buffer, 60)
        starts = {buffer.sample()["indices"][0, 0] for _ in range(10)}
        assert len(starts) > 1


class TestLatents:
    def make(self):
        return make_buffer(
            size=64,
            seq_len=4,
            batch_size=4,
            deter_size=6,
            latent_dim=3,
            latent_classes=5,
        )

    @staticmethod
    def latent(value, index):
        stoch = torch.zeros(3, 5)
        stoch[torch.arange(3), index] = 1.0
        return {"deter": torch.full((6,), float(value)), "stoch": stoch}

    def test_stored_state_of_the_previous_record_starts_the_sequence(self):
        buffer = self.make()
        for step in range(30):
            buffer.add(
                obs(step),
                np.zeros(2),
                0.0,
                is_first=step == 0,
                latent=self.latent(step, step % 5),
            )
        for _ in range(10):
            batch = buffer.sample()
            for row in range(4):
                start = int(batch["indices"][0, row])
                if start == 0:
                    assert batch["init_valid"][row] == 0  # episode start
                    continue
                assert batch["init_valid"][row] == 1
                assert np.allclose(batch["init_deter"][row], start - 1)
                assert (
                    batch["init_stoch"][row].argmax(-1).tolist()
                    == [(start - 1) % 5] * 3
                )

    def test_records_without_a_stored_state_start_from_zeros(self):
        buffer = self.make()
        fill(buffer, 30)
        batch = buffer.sample()
        assert np.all(batch["init_valid"] == 0)
        assert np.all(batch["init_deter"] == 0)

    def test_update_latents_writes_fresh_states_back(self):
        buffer = self.make()
        fill(buffer, 30)
        batch = buffer.sample()
        seq_len, batch_size = batch["indices"].shape
        deter = torch.full((seq_len, batch_size, 6), 2.5)
        stoch = torch.zeros(seq_len, batch_size, 3, 5)
        stoch[..., 4] = 1.0
        buffer.update_latents(batch["indices"], deter, stoch)
        written = batch["indices"].reshape(-1)
        assert buffer.has_latent[written].all()
        assert np.allclose(buffer.deter[written], 2.5)
        assert np.all(buffer.stoch[written] == 4)

    def test_the_oldest_record_has_no_usable_predecessor(self):
        buffer = self.make()
        for step in range(70):
            buffer.add(
                obs(step),
                np.zeros(2),
                0.0,
                is_first=step == 0,
                latent=self.latent(step, 0),
            )
        oldest = buffer.idx
        _, _, valid = buffer._initial_state(np.array([oldest]))
        assert valid[0] == 0


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
                is_first=step == 0,
            )
        batch = buffer.sample()
        assert batch["observations"].dtype == np.float32
        assert np.any(batch["observations"] % 1.0 != 0.0)
