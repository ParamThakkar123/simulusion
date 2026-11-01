# Memory & Replay Buffers

Simulusion provides replay buffers and episodic memory for each agent family.
All buffers store environment interaction data and support sequence sampling
for world-model training.

```{contents} Contents
:depth: 3
```

## Which buffer should I use?

| Agent | Buffer class | Location | Storage |
|---|---|---|---|
| Dreamer (V1/V2/V3) | `ReplayBuffer` | `simulusion.memory.dreamer_memory` | Ring buffer of individual transitions |
| PlaNet / RSSM | `Memory` + `Episode` | `simulusion.memory.planet_memory` | Deque of complete episodes |
| IRIS | `IRISReplayBuffer` + `IRISOnPolicyBuffer` | `simulusion.memory.iris_memory` | Ring buffer of individual transitions |
| DIAMOND | `ReplayBuffer` + `SequenceDataset` | `simulusion.datasets.diamond_dataset` | Ring buffer with next-observation + PyTorch Dataset wrapper |

All buffers are accessible from the top-level package:

```python
import simulusion

buffer = simulusion.ReplayBuffer(size=100000, obs_shape=(3, 64, 64), action_size=6)
```

## Dreamer `ReplayBuffer`

The primary buffer for Dreamer training. Stores image observations as uint8
to save memory and samples **contiguous sequences** for temporal world-model
learning.

```python
from simulusion import ReplayBuffer

buffer = ReplayBuffer(
    size=100000,  # max transitions before FIFO eviction
    obs_shape=(3, 64, 64),  # C, H, W
    action_size=6,  # continuous action dimension
    seq_len=50,  # sequence length per sample
    batch_size=50,  # parallel sequences per batch
)

# Add a transition during environment interaction
buffer.add(obs, action, reward, done)
#   obs:    dict with key "image" containing (C, H, W) uint8
#   action: (action_size,) float32
#   reward: scalar float
#   done:   1.0 if terminal, 0.0 otherwise

# Sample a training batch
obs_batch, act_batch, rew_batch, term_batch = buffer.sample()
#   obs_batch:  (seq_len, batch, C, H, W) uint8
#   act_batch:  (seq_len, batch, action_size) float32
#   rew_batch:  (seq_len, batch) float32
#   term_batch: (seq_len, batch) float32
```

### Important details

| Aspect | Detail |
|---|---|
| **Sequence boundary validation** | Sampling skips indices that would cause a sequence to cross an episode boundary (detected via terminal flags). Prevents the model from learning impossible transitions. |
| **Memory efficient** | uint8 images use 1 byte per pixel vs 4 bytes for float32. A 100k-buffer of 3×64×64 images uses ~1.2 GB. |
| **Time-major format** | Returned arrays are `(seq_len, batch, ...)` — the format expected by RSSM and Dreamer's recurrent training loop. |

## PlaNet `Memory` and `Episode`

PlaNet stores **complete episodes** rather than individual transitions.
Each episode is captured by an `Episode` object, and a collection of episodes
is managed by `Memory`.

```python
from simulusion import Memory, Episode

memory = Memory(size=100)  # keep at most 100 episodes

# Record an episode
episode = Episode()
episode.append(obs, action, reward, done)
episode.append(obs, action, reward, done)
episode.terminate(final_obs)  # converts lists to numpy arrays

memory.append([episode])

# Sample sub-sequences for training
sequences, lengths = memory.sample(batch_size=32, tracelen=50)
#   sequences: [observations, actions, rewards, terminals]
#     observations: (batch, tracelen+1, C, H, W) or (tracelen+1, batch, ...) with time_first=True
#     actions:      (batch, tracelen, action_dim)
#     rewards:      (batch, tracelen)
#     terminals:    (batch, tracelen)
#   lengths:  (batch,) original episode lengths
```

### Key differences from Dreamer's buffer

| Feature | Dreamer `ReplayBuffer` | PlaNet `Memory` |
|---|---|---|
| Storage unit | Individual transitions | Complete episodes |
| Sampling | Random subsequences from any point | Subsequences from randomly selected episodes |
| Eviction | FIFO per-transition | FIFO per-episode |
| OOM protection | Fixed-size preallocation | Memory estimation with 200 MiB threshold |
| Time-major | Always `(T, B, ...)` | Optional via `time_first=True` |

## IRIS buffers

IRIS uses two buffers: a ring buffer for long-term storage and an on-policy
buffer for collecting the current episode.

```python
from simulusion import IRISReplayBuffer, IRISOnPolicyBuffer

# Main replay buffer
buffer = IRISReplayBuffer(
    size=50000,
    obs_shape=(3, 64, 64),
    action_size=6,
    seq_len=20,
    batch_size=64,
)

buffer.add(obs, action, reward, done)
#   obs:    (C, H, W) uint8
#   action: (action_size,) float32
#   reward: scalar float
#   done:   bool

# Sample sequences (one extra obs frame for next-frame prediction)
obs_batch, act_batch, rew_batch, term_batch = buffer.sample_sequence()
#   obs_batch:  (batch, seq_len+1, C, H, W)
#   act_batch:  (batch, seq_len, action_size)
#   rew_batch:  (batch, seq_len)
#   term_batch: (batch, seq_len)

# Single-transition sampling
obs, act, rew, done = buffer.sample_single()

# On-policy buffer for episode collection
on_policy = IRISOnPolicyBuffer(max_steps=1000)

while not done:
    on_policy.add(obs, action, reward, done)

# Transfer to main buffer
for i in range(len(on_policy)):
    buffer.add(
        on_policy.observations[i],
        on_policy.actions[i],
        on_policy.rewards[i],
        on_policy.terminals[i],
    )
on_policy.clear()
```

## DIAMOND `ReplayBuffer` and `SequenceDataset`

The DIAMOND buffer works with a PyTorch `Dataset` wrapper for integration
with DataLoader-based training loops.

```python
from simulusion.datasets.diamond_dataset import ReplayBuffer, SequenceDataset

buffer = ReplayBuffer(
    capacity=100000,
    obs_shape=(64, 64, 3),  # H, W, C (native format)
    action_dim=4,
)

buffer.add(obs, action, reward, done, next_obs)
# Expects HWC observations (converted internally to CHW for training)

# Wrap in PyTorch Dataset
dataset = SequenceDataset(
    replay_buffer=buffer,
    sequence_length=5,
    burn_in=4,
)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch keys: obs_seq, action_seq, rewards, dones, next_obs
    pass

# Checkpointing
state = buffer.state_dict()
torch.save(state, "buffer.pt")

# Restore
state = torch.load("buffer.pt", weights_only=True)
buffer.load_state_dict(state)
```

### Comparison of `state_dict` / `load_state_dict`

Only the DIAMOND `ReplayBuffer` in `datasets/diamond_dataset.py` supports
checkpointing via `state_dict()` / `load_state_dict()`. The Dreamer and IRIS
ring buffers do not — they must be repopulated by re-running environment
interactions.

## Common patterns

### Recording during environment interaction

```python
buffer = ReplayBuffer(size=100000, obs_shape=(3, 64, 64), action_size=6)
obs, _ = env.reset()

for step in range(total_steps):
    action = policy(obs)
    next_obs, reward, done, truncated, _ = env.step(action)
    buffer.add({"image": obs}, action, reward, float(done or truncated))
    obs = next_obs
```

### Training loop with sequence sampling

```python
while step < total_steps:
    # Collect experience
    obs = env.reset()
    for _ in range(collect_steps):
        action = policy(obs)
        next_obs, reward, done, _ = env.step(action)
        buffer.add({"image": obs}, action, reward, float(done))
        obs = next_obs

    # Train on sampled sequences
    for _ in range(update_steps):
        obs_batch, act_batch, rew_batch, term_batch = buffer.sample()
        loss = world_model_train_step(obs_batch, act_batch, rew_batch, term_batch)
```

### Episodic memory for planning

```python
memory = Memory(size=50)
episode = Episode(postprocess_fn=lambda x: x / 255.0)

obs, _ = env.reset()
episode.append(obs, torch.zeros(action_dim), 0.0, False)

for _ in range(max_steps):
    action = plan(episode)
    obs, reward, done, _ = env.step(action)
    episode.append(obs, action, reward, done)
    if done:
        episode.terminate(obs)
        memory.append([episode])
        break

# Sample for training
sequences, lengths = memory.sample(batch_size=16, tracelen=50, time_first=True)
```

## See Also

- {doc}`training_guide` — how replay buffers integrate with training loops
- {doc}`dreamer` — uses Dreamer ReplayBuffer
- {doc}`iris` — uses IRISReplayBuffer + IRISOnPolicyBuffer
