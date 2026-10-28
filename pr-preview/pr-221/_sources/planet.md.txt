# PlaNet

**PlaNet** (Deep **Pl**anning **Net**work) is a latent-dynamics model that learns
a compact state-space representation from image observations and uses online
planning rather than a separately trained actor-critic policy. At each step,
PlaNet runs a **cross-entropy method** (CEM) planner inside the learned
dynamics model to select the best action.

PlaNet introduced the **RSSM** (Recurrent State-Space Model) that later became
the backbone of Dreamer and other latent-dynamics agents.

## See it in action

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/planet_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: the real environment. Right: PlaNet's latent model predicting open loop from the actions its CEM planner chooses.</figcaption>
</figure>
```

Trained from scratch in one to two hours on a 4 GB laptop GPU with
`python demos/planet_demo.py train`, then recorded with `python demos/planet_demo.py record`.
More clips are in the {ref}`gallery <gallery-planet>`.

```{contents} Contents
```

## Architecture

PlaNet's learned world model has four components:

| Component | Description |
|---|---|
| **CNN Encoder** | 4-layer conv net (`64×64` → `1024-d` embedding) |
| **RSSM** | GRU-based recurrent state-space model with deterministic state `h_t` and stochastic latent `s_t` |
| **CNN Decoder** | 4-layer transposed-conv decoder (reconstructs image from `(h_t, s_t)`) |
| **Reward predictor** | 3-layer MLP that predicts reward from `(h_t, s_t)` |

### RSSM latent dynamics

The RSSM maintains two state variables:

- **Deterministic state** `h_t` — a GRU hidden state that captures temporal
  context across the full sequence.
- **Stochastic latent** `s_t` — a diagonal Gaussian `N(μ_t, σ_t)` representing
  the uncertainty about the current observation.

At each timestep the model:

1. Encodes the observation: `e_t = enc(o_t)`
2. Updates the deterministic state: `h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1})`
3. Computes the **prior**: `p(s_t | h_t)` — predicts the latent without seeing the observation
4. Computes the **posterior**: `q(s_t | h_t, e_t)` — infers the latent after seeing the observation

During planning, the **prior** is used to roll out imagined trajectories. The
**posterior** is only used during training to provide a target.

## Planning with CEM

PlaNet does not train a separate policy network. Instead it uses the
**cross-entropy method** at every environment step:

1. Sample `K` candidate action sequences from a diagonal Gaussian.
2. Roll out each candidate through the RSSM prior for `H` steps.
3. Score each rollout by the sum of predicted rewards.
4. Refit the Gaussian to the top `N` candidates.
5. Repeat for `I` iterations, then execute the first action of the best sequence.

| Parameter | Default | Description |
|---|---|---|
| `planning_horizon` | 20 | Number of imagined steps per candidate |
| `num_candidates` | 1000 | Candidate action sequences sampled per iteration |
| `num_iterations` | 10 | CEM re-fitting iterations |
| `top_candidates` | 100 | Elite candidates kept for re-fitting |

## Loss function

PlaNet uses the **standard variational bound** (single-step predictions):

```{math}
\mathcal{L} = \beta \cdot \text{KL}\big(q(s_t \mid h_t, e_t) \;\|\; p(s_t \mid h_t)\big)
+ \|o_t - \hat{o}_t\|^2
+ \text{MSE}(r_t, \hat{r}_t)
```

| Term | Description |
|---|---|
| **KL divergence** | Regularizes the posterior toward the prior, with free nats = 3.0 |
| **Reconstruction (MSE)** | Pixel-level image reconstruction loss |
| **Reward prediction (MSE)** | Learns to predict rewards from the latent state |

The KL term uses `free_nats = 3.0` — the KL is clamped so that once the
posterior and prior are within 3 nats, no further gradient pressure is applied.

## Memory

PlaNet stores **complete episodes** rather than individual transitions:

```python
from simulusion import Memory, Episode

memory = Memory(size=100)  # keep at most 100 episodes
episode = Episode()
episode.append(obs, action, reward, done)
episode.terminate(final_obs)
memory.append([episode])

# Sample random subsequences for training
sequences, lengths = memory.sample(batch_size=32, tracelen=50)
```

| Feature | Detail |
|---|---|
| **Storage unit** | Complete episodes (not individual transitions) |
| **Sampling** | Randomly select episodes, then contiguous subsequences |
| **Eviction** | FIFO per-episode (deque-based) |
| **Time-major** | Optional via `time_first=True` |

## Usage in Simulusion

### Direct construction

```python
import simulusion

agent = simulusion.create_model("planet", env="CartPole-v1")
agent.train(epochs=100, steps_per_epoch=150)
```

PlaNet takes parameters directly in its constructor (no separate config class):

```python
from simulusion import Planet

agent = Planet(
    env="CartPole-v1",
    bit_depth=5,  # image bit depth for preprocessing
    state_size=200,  # deterministic GRU state dimension
    latent_size=30,  # stochastic latent dimension
    embedding_size=1024,  # encoder output dimension
    memory_size=100,  # number of episodes to keep
    action_repeats=1,
    max_episode_steps=1000,
    headless=False,  # set True for headless servers
)
```

### Training options

```python
results_dir = agent.train(
    epochs=100,
    steps_per_epoch=150,
    batch_size=32,
    H=50,  # sequence length for training
    beta=1.0,  # KL weight
    save_every=25,
    record_grads=False,
    scheduler_type="step",  # LR scheduler: "step", "cosine", "exponential", "plateau", None
    scheduler_kwargs={"step_size": 50, "gamma": 0.5},
)
```

### Warmup

Collect random episodes before training starts:

```python
agent.warmup(n_episodes=5, random_policy=True)
```

If not called explicitly, one warmup episode is collected automatically at the
start of `train()`.

### CEM Planner

The CEM planner is embedded inside the `Planet` agent and configured through the
`policy_cfg` dict:

```python
agent = Planet(
    env="CartPole-v1",
    policy_cfg={
        "planning_horizon": 20,
        "num_candidates": 1000,
        "num_iterations": 10,
        "top_candidates": 100,
    },
)
```

## See Also

- {doc}`dreamer` — successor to PlaNet; replaces CEM with a trained actor-critic
- {doc}`iris` — discrete world model that also uses the CEM-style planning
- {doc}`memory_guide` — PlaNet Memory and Episode in detail
- {doc}`world_models_guide` — conceptual overview of latent-dynamics models
- {doc}`vision_guide` — CNNEncoder and CNNDecoder used by PlaNet
