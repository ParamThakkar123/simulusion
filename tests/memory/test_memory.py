import pytest
import numpy as np
import torch
from simulusion.memory.planet_memory import Episode, Memory
from simulusion.memory.iris_memory import IRISReplayBuffer, IRISOnPolicyBuffer


class TestEpisode:
    def test_append_and_terminate(self):
        episode = Episode()

        obs = torch.from_numpy(np.random.rand(3, 64, 64).astype(np.float32))
        act = torch.tensor([0.5])
        reward = 1.0
        terminal = False

        episode.append(obs, act, reward, terminal)
        assert episode.size == 1

        obs2 = torch.from_numpy(np.random.rand(3, 64, 64).astype(np.float32))
        episode.terminate(obs2)

        assert isinstance(episode.x, np.ndarray)
        assert isinstance(episode.u, np.ndarray)
        assert isinstance(episode.r, np.ndarray)
        assert isinstance(episode.t, np.ndarray)
        assert episode.x.shape[0] == 2

    def test_postprocess_fn(self):
        def postprocess(x):
            return x * 2

        episode = Episode(postprocess_fn=postprocess)
        obs = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32))
        act = torch.tensor([0.5])

        episode.append(obs, act, 1.0, False)

        assert episode.x[0].sum() == 2 * 3 * 64 * 64


class TestMemory:
    def test_init_with_size(self):
        mem = Memory(size=10)
        assert mem.episodes.maxlen == 10

    def test_init_without_size(self):
        mem = Memory()
        assert mem.episodes.maxlen is None

    def test_append_single_episode(self):
        mem = Memory(size=10)
        episode = Episode()

        for i in range(5):
            obs = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * i)
            act = torch.tensor([float(i)])
            episode.append(obs, act, float(i), i == 4)

        obs_term = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * 5)
        episode.terminate(obs_term)

        mem.append(episode)

        assert len(mem.episodes) == 1

    def test_append_list_of_episodes(self):
        mem = Memory(size=10)

        episode1 = Episode()
        episode1.append(torch.ones((3, 64, 64)), torch.tensor([1.0]), 1.0, False)
        episode1.terminate(torch.ones((3, 64, 64)))

        episode2 = Episode()
        episode2.append(torch.ones((3, 64, 64)) * 2, torch.tensor([2.0]), 2.0, False)
        episode2.terminate(torch.ones((3, 64, 64)) * 2)

        mem.append([episode1, episode2])

        assert len(mem.episodes) == 2

    def test_append_invalid_type(self):
        mem = Memory(size=10)

        with pytest.raises(ValueError):
            mem.append("not an episode")

    def test_sample_empty_memory(self):
        mem = Memory(size=10)

        with pytest.raises(ValueError, match="Memory is empty"):
            mem.sample(2)

    def test_sample_no_valid_episodes(self):
        mem = Memory(size=10)
        episode = Episode()

        episode.append(torch.ones((3, 64, 64)), torch.tensor([1.0]), 1.0, True)
        episode.terminate(torch.ones((3, 64, 64)))

        mem.append(episode)

        with pytest.raises(ValueError, match="No episodes with length"):
            mem.sample(2, tracelen=10)

    def test_sample_basic(self):
        mem = Memory(size=10)

        for _ in range(3):
            episode = Episode()
            for i in range(10):
                obs = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * i)
                act = torch.tensor([float(i)])
                episode.append(obs, act, float(i), i == 9)
            obs_term = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * 10)
            episode.terminate(obs_term)
            mem.append(episode)

        result, lengths = mem.sample(batch_size=2, tracelen=3)

        x, u, r, t = result
        assert x.shape[0] == 2
        assert x.shape[1] == 4
        assert u.shape[0] == 2
        assert u.shape[1] == 3
        assert lengths.shape[0] == 2

    def test_sample_time_first(self):
        mem = Memory(size=10)

        for _ in range(3):
            episode = Episode()
            for i in range(10):
                obs = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * i)
                act = torch.tensor([float(i)])
                episode.append(obs, act, float(i), i == 9)
            obs_term = torch.from_numpy(np.ones((3, 64, 64), dtype=np.float32) * 10)
            episode.terminate(obs_term)
            mem.append(episode)

        result, lengths = mem.sample(batch_size=2, tracelen=3, time_first=True)

        x, u, r, t = result
        assert x.shape[0] == 4
        assert x.shape[1] == 2


class TestIRISReplayBuffer:
    def test_init(self):
        buffer = IRISReplayBuffer(
            size=1000, obs_shape=(3, 64, 64), action_size=2, seq_len=20, batch_size=32
        )

        assert buffer.size == 1000
        assert buffer.obs_shape == (3, 64, 64)
        assert buffer.action_size == 2
        assert buffer.seq_len == 20
        assert buffer.batch_size == 32

    def test_add(self):
        buffer = IRISReplayBuffer(size=100, obs_shape=(3, 64, 64), action_size=2)

        obs = np.ones((3, 64, 64), dtype=np.uint8)
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, False)

        assert buffer.steps == 1
        assert buffer.episodes == 0

    def test_add_terminal(self):
        buffer = IRISReplayBuffer(size=100, obs_shape=(3, 64, 64), action_size=2)

        obs = np.ones((3, 64, 64), dtype=np.uint8)
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, True)

        assert buffer.steps == 1
        assert buffer.episodes == 1

    def test_len(self):
        buffer = IRISReplayBuffer(size=100, obs_shape=(3, 64, 64), action_size=2)

        for i in range(50):
            obs = np.ones((3, 64, 64), dtype=np.uint8) * i
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 49)

        assert len(buffer) == 50

    def test_len_full(self):
        buffer = IRISReplayBuffer(size=100, obs_shape=(3, 64, 64), action_size=2)

        for i in range(150):
            obs = np.ones((3, 64, 64), dtype=np.uint8) * (i % 100)
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 149)

        assert len(buffer) == 100

    def test_sample_sequence(self):
        buffer = IRISReplayBuffer(
            size=100, obs_shape=(3, 64, 64), action_size=2, seq_len=5, batch_size=4
        )

        for i in range(50):
            obs = np.ones((3, 64, 64), dtype=np.uint8) * i
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 49)

        obs_batch, act_batch, rew_batch, term_batch = buffer.sample_sequence()

        assert obs_batch.shape == (4, 6, 3, 64, 64)
        assert act_batch.shape == (4, 5, 2)
        assert rew_batch.shape == (4, 5)
        assert term_batch.shape == (4, 5)

    def test_sample_single(self):
        buffer = IRISReplayBuffer(size=100, obs_shape=(3, 64, 64), action_size=2)

        for i in range(10):
            obs = np.ones((3, 64, 64), dtype=np.uint8) * i
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 9)

        obs, act, rew, term = buffer.sample_single()

        assert obs.shape == (3, 64, 64)
        assert act.shape == (2,)
        assert isinstance(rew, (float, np.floating))
        assert isinstance(term, (float, np.floating))

    def test_buffer_capacity(self):
        buffer = IRISReplayBuffer(size=500, obs_shape=(3, 64, 64), action_size=2)

        assert buffer.buffer_capacity == 500


class TestIRISOnPolicyBuffer:
    def test_init(self):
        buffer = IRISOnPolicyBuffer(max_steps=1000)

        assert buffer.max_steps == 1000
        assert len(buffer) == 0

    def test_add(self):
        buffer = IRISOnPolicyBuffer()

        obs = np.ones((3, 64, 64), dtype=np.uint8)
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, False)

        assert len(buffer) == 1

    def test_clear(self):
        buffer = IRISOnPolicyBuffer()

        obs = np.ones((3, 64, 64), dtype=np.uint8)
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, False)
        buffer.add(obs, action, 2.0, True)

        buffer.clear()

        assert len(buffer) == 0

    def test_get_arrays(self):
        buffer = IRISOnPolicyBuffer()

        obs1 = np.ones((3, 64, 64), dtype=np.uint8)
        obs2 = np.ones((3, 64, 64), dtype=np.uint8) * 2
        action = np.array([0.5, 0.5])

        buffer.add(obs1, action, 1.0, False)
        buffer.add(obs2, action, 2.0, True)

        observations, actions, rewards, terminals = buffer.get_arrays()

        assert observations.shape == (2, 3, 64, 64)
        assert actions.shape == (2, 2)
        assert rewards.shape == (2,)
        assert terminals.shape == (2,)

    def test_len(self):
        buffer = IRISOnPolicyBuffer()

        for i in range(10):
            obs = np.ones((3, 64, 64), dtype=np.uint8) * i
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 9)

        assert len(buffer) == 10


class TestDreamerReplayBuffer:
    def test_init(self):
        from simulusion.memory.dreamer_memory import ReplayBuffer

        buffer = ReplayBuffer(
            size=1000, obs_shape=(3, 64, 64), action_size=2, seq_len=10, batch_size=32
        )

        assert buffer.size == 1000
        assert buffer.obs_shape == (3, 64, 64)

    def test_add(self):
        from simulusion.memory.dreamer_memory import ReplayBuffer

        buffer = ReplayBuffer(
            size=100, obs_shape=(3, 64, 64), action_size=2, seq_len=10, batch_size=32
        )

        obs = {"image": np.ones((3, 64, 64), dtype=np.uint8)}
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, False)

        assert buffer.steps == 1
        assert buffer.episodes == 0

    def test_add_terminal(self):
        from simulusion.memory.dreamer_memory import ReplayBuffer

        buffer = ReplayBuffer(
            size=100, obs_shape=(3, 64, 64), action_size=2, seq_len=10, batch_size=32
        )

        obs = {"image": np.ones((3, 64, 64), dtype=np.uint8)}
        action = np.array([0.5, 0.5])

        buffer.add(obs, action, 1.0, True)

        assert buffer.steps == 1
        assert buffer.episodes == 1

    def test_sample(self):
        from simulusion.memory.dreamer_memory import ReplayBuffer

        buffer = ReplayBuffer(
            size=100, obs_shape=(3, 64, 64), action_size=2, seq_len=5, batch_size=4
        )

        for i in range(50):
            obs = {"image": np.ones((3, 64, 64), dtype=np.uint8) * i}
            action = np.array([float(i), 0.0])
            buffer.add(obs, action, float(i), i == 49)

        obs_batch, act_batch, rew_batch, term_batch = buffer.sample()

        assert obs_batch.shape[0] == 5
        assert obs_batch.shape[1] == 4
