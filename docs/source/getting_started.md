# Getting Started

This page takes you from installation to a trained world model you can watch,
then shows where to go for each model family.

## 1. Install

```bash
pip install "simulusion[gym]"
```

The package installs as `simulusion` and imports as `simulusion`. The `gym`
extra brings Gymnasium with its classic-control and Atari environments. Other
extras add DeepMind Control (`dmc`), logging (`ml`), MuJoCo, Brax and more; see
{doc}`installation`.

Simulusion does not pin a PyTorch build. For GPU training, install the CUDA wheel
that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/)
first, and check it with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Train your first world model

Dreamer learns a model of the environment from raw pixels and trains its policy
entirely inside that model:

```python
import simulusion

agent = simulusion.create_model(
    "dreamer",
    env="Pendulum-v1",
    env_backend="gym",
    total_steps=20_000,
)
agent.train()
```

Training prints a line per iteration and writes everything to one run folder,
`runs/Pendulum-v1_Dreamerv1_<name>_<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | the full configuration of the run |
| `metrics.jsonl` | one JSON line per iteration: returns, losses, throughput |
| `ckpts/<step>_ckpt.pt` | checkpoints, the final one always written |

`create_model` accepts any field of the model's config as a keyword argument.
To see them all, build the config first:

```python
cfg = simulusion.create_config("dreamer", env="Pendulum-v1", env_backend="gym")
print(cfg)
```

## 3. Watch what it learned

The [demo scripts](https://github.com/ParamThakkar123/simulusion/tree/main/demos)
turn a run into video. For Dreamer, `record` shows the real environment beside
the world model's prediction of it, plus the policy acting:

```bash
python demos/dreamer_demo.py record --run runs/Pendulum-v1_Dreamerv1_<name>_<timestamp>
```

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/dreamer_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>What <code>record</code> produces, here for a Dreamer trained on cartpole swing-up. Left: the real environment. Right: the world model, observing 5 frames and then imagining the rest.</figcaption>
</figure>
```

For a live session instead of a recording, `simulusion play --model dreamer -c
<checkpoint>` opens a window where you can take over the controls (needs a
desktop session).

## 4. Choose a model

Every model is built with `simulusion.create_model(name, ...)`. The names are those
returned by `simulusion.list_models()`:

| Model | `create_model` name | What it does | Demo |
|---|---|---|---|
| {doc}`Dreamer <dreamer>` | `dreamer`, `dreamer-v1`, `dreamer-v2` | Learns latent dynamics from pixels; trains an actor-critic in imagination | {ref}`video <gallery-dreamer>` |
| {doc}`PlaNet <planet>` | `planet` | Plans each action with CEM inside a learned latent model | {ref}`video <gallery-planet>` |
| {doc}`DIAMOND <diamond>` | `diamond` | Diffusion world model of Atari; agent trained on generated frames | {ref}`video <gallery-diamond>` |
| {doc}`IRIS <iris>` | `iris` | Discrete-token autoencoder + Transformer world model | {ref}`video <gallery-iris>` |
| {doc}`Genie <genie>` | `genie`, `genie-small`, `genie-large` | Playable world model learned from unlabelled video | {ref}`video <gallery-genie>` |
| {doc}`DiT <dit>` | `dit` | Diffusion Transformer for image generation | {ref}`video <gallery-dit>` |
| {doc}`I-JEPA <jepa>` | `jepa` | Self-supervised image representations, predicted in latent space | {ref}`images <gallery-i-jepa>` |
| {doc}`ModularRSSM <modular_rssm_guide>` | `modular-rssm` | A world model assembled from swappable encoder, backbone and decoder | {ref}`video <gallery-modular-rssm>` |

To run any of them on a different environment, pass `env=` and
`env_backend=`. The same backend names work with `simulusion.make_env()`:

| Backend | Environments |
|---|---|
| `gym` | Gymnasium IDs (classic control, Box2D, Atari), or an existing env instance |
| `dmc` | DeepMind Control Suite tasks, written `domain-task` (e.g. `walker-walk`) |
| `mujoco` | Gymnasium MuJoCo tasks or native MJCF/MJB models |
| `robotics` | IDs registered by Gymnasium Robotics |
| `brax` | JAX/Brax continuous-control environments |
| `procgen` | Procgen games such as `coinrun` (manual install) |
| `dmlab` | DeepMind Lab 3D navigation tasks |
| `unity_mlagents` | Unity ML-Agents executables |

## 5. Log to Weights & Biases or TensorBoard

Both are off by default. Install the `ml` extra, then set them on the config:

```python
cfg = simulusion.create_config("dreamer", env="Pendulum-v1", env_backend="gym")
cfg.enable_tensorboard = True   # then: tensorboard --logdir runs
cfg.enable_wandb = True         # needs WANDB_API_KEY in the environment
cfg.wandb_project = "simulusion"
agent = simulusion.create_model("dreamer", config=cfg)
```

## Next steps

- The {doc}`gallery` shows every model trained on a laptop GPU, with the
  commands to reproduce each clip.
- {doc}`training_guide` covers budgets, checkpointing, resuming and early
  stopping.
- {doc}`inference_guide` and {doc}`deployment_guide` cover fast inference and
  exporting to ONNX, TorchScript and TensorRT.
- {doc}`world_models_guide` explains the ideas the models share.
