# Simulusion demos

One script per model. Each trains the model from scratch on a single GPU in one
to two hours and then records what it learned as video, images and a learning
curve. The results are shown in the
[documentation gallery](https://paramthakkar123.github.io/simulusion/gallery.html).

| Model | Script | Environment / data | Default budget | Records |
|---|---|---|---|---|
| Dreamer | `dreamer_demo.py` | DMC cartpole swing-up | 60k env steps | real vs imagined, policy, curve |
| DreamerV2 | `dreamer_demo.py --algo dreamer-v2` | DMC cartpole swing-up | 60k env steps | real vs imagined, policy, curve |
| PlaNet | `planet_demo.py` | DMC cartpole swing-up | 75 min | real vs imagined, CEM planner, curve |
| ModularRSSM | `modular_rssm_demo.py` | DMC cartpole swing-up, random actions | 25 min per backbone | real vs GRU vs LSTM, curve |
| DIAMOND | `diamond_demo.py` | Atari Breakout | 21 epochs | real vs diffusion dream, agent in the dream, curve |
| IRIS | `iris_demo.py` | Atari Pong | 100 min | token reconstructions, imagination, policy, curve |
| Genie | `genie_demo.py` | TinyWorlds Sonic video | 6k steps | replay from inferred actions, all latent actions, tokens, curve |
| DiT | `dit_demo.py` | CIFAR-10 | 40 epochs | class grid, denoising video |
| I-JEPA | `jepa_demo.py` | CIFAR-10 | 30 epochs | nearest neighbours, patch-feature PCA, kNN accuracy |

The budgets were sized on an RTX 3050 Laptop GPU with 4 GB of memory and 16 GB
of RAM.

## Running a demo

Every script has the same two commands:

```bash
python demos/dreamer_demo.py train     # writes demos/runs/dreamer/
python demos/dreamer_demo.py record    # writes demos/runs/dreamer/media/
```

`--help` on either command lists its options, for example a different
environment, budget or output folder. Run from a clone, the scripts use the
repository's `simulusion`; otherwise they use the installed package.

To train and record everything, one demo after another, and copy the results
into the docs:

```bash
python demos/run_all.py                  # every demo not recorded yet
python demos/run_all.py dreamer genie    # only these
python demos/run_all.py --record-only    # re-record from existing runs
python demos/build_gallery.py            # compress media into docs/source/_static/gallery
```

`run_all.py` is safe to stop and rerun. Finished demos are skipped, and the
Dreamer demos continue from their newest checkpoint (`dreamer_demo.py train
--resume`): weights and optimizers are restored, the replay buffer refills from
fresh experience, and the learning curve is stitched across the restart. Other
demos that were interrupted start training again.

## Requirements

```bash
pip install "simulusion[gym,dmc,viz]"   # Atari, DeepMind Control, H.264 encoder
pip install imageio matplotlib omegaconf  # video/plots, and the Genie trainer's CLI
```

- **PyTorch with CUDA.** Simulusion does not pin a PyTorch build; install the wheel
  for your driver from [pytorch.org](https://pytorch.org/get-started/locally/).
  Check that it sees the GPU with
  `python -c "import torch; print(torch.cuda.is_available())"`.
- **CIFAR-10** for DiT and I-JEPA is read from `data/`. Download it once with
  `python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"`.
- **TinyWorlds** data for Genie downloads automatically from the Hugging Face
  Hub on first use (about 250 MB for Sonic).

## What the clips show

The world-model clips put the real environment on the left and the model on the
right. The model observes a few real frames, then its observations are cut off
and every later frame is its own prediction, given the same actions that are
taken in the real environment. Where the halves drift apart is where the model
is wrong. DIAMOND's `play_in_dream.mp4` and IRIS's `dream.mp4` go one step
further: the agent acts on the model's generated frames, with no emulator
involved at all.

## Playing in the browser

`export_web_diamond.py` turns a DIAMOND checkpoint into files a web page can
run with ONNX Runtime Web: one graph per dream frame (the whole three-step
sampler), one per policy step, and a few real opening frames to start dreams
from. Both graphs are checked against the eager model before the script exits.

```bash
python demos/export_web_diamond.py -c demos/runs/diamond/continue_21/checkpoint_30.pt \
    --out ../TorchWMUI/public/play/diamond
```

The export pads convs whose input width is not a multiple of 4 with zero
channels: ONNX Runtime Web 1.30's WebGPU Conv got DIAMOND's 15-channel input
conv wrong (3, 12 and 16 channels were exact).

## Notes from building these

- **Run one demo at a time.** Dreamer's replay buffer and the Atari replays hold
  gigabytes of RAM, and each demo uses most of a 4 GB GPU. Two at once ran the
  machine out of memory.
- **Watch GPU memory on Windows.** When a model needs more VRAM than the card
  has, the driver silently spills into system memory and training slows by
  roughly 10x instead of failing. Genie at batch 4 x 16 frames needs about
  5 GB and ran at 7 s/step; the demo uses batch 3 with mixed precision
  (about 3.2 GB, 0.9 s/step).
- **DataLoader workers.** On Windows each worker is a new process that
  re-imports torch and receives a pickled copy of the dataset. The DIAMOND and
  I-JEPA demos use `num_workers=0`; with workers, DIAMOND copied its replay
  buffer into each one and I-JEPA stalled at start-up.
- **Interval alignment in Dreamer.** `global_step` advances by `action_repeat`
  per environment step, so checkpoint and evaluation intervals only fire if they
  are multiples of it. `dreamer_demo.py` sets compatible values.

## Older scripts

`train_demo.py`, `record_diamond.py`, `record_iris.py`, `record_dit.py`,
`record_genie.py` and `record_jepa.py` predate the per-model demos above, which
supersede them. They are kept for now, but several of them build models from
default configs rather than the checkpoint's own, so prefer the scripts above.
