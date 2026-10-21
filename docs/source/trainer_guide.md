# One Trainer for Every Model

Every family in Simulusion trains on the same loop,
{py:class}`simulusion.training.Trainer`, from the same kind of file: a YAML
**run config** that names a **task** and fills its sections. The trainer owns
what is common to all of them: devices and data parallelism (through
[Hugging Face Accelerate](https://huggingface.co/docs/accelerate)), one
precision setting, seeding, the step budget, logging, evaluation, early
stopping, checkpoints and resuming, and exporting the trained world model.

```{contents} Contents
:depth: 2
```

## Train from the command line

```bash
# A shipped config by name, with key=value overrides for any field.
simulusion train --config dreamer model.env=cheetah-run trainer.precision=bf16

# Or name the task and set only what you need; the rest keeps its defaults.
simulusion train genie data.path=data/clips trainer.max_steps=20000

# See the fully resolved config without training.
simulusion train --config iris --print-config
```

Nested keys use dots (`model.lr=3e-4`, `trainer.eval_every=1000`). A key that
does not exist is an error that names its full path, so a typo fails before
the run starts instead of being ignored. `simulusion train --legacy MODEL`
still reaches each family's older entrypoint.

## Train from Python

```python
import simulusion

result = simulusion.run_training(
    "simulusion/configs/runs/genie.yaml",
    ["data.path=data/clips", "trainer.max_steps=500"],
)
model = simulusion.load_world_model(result.world_model_path)
```

or build the pieces yourself:

```python
from simulusion.training import Trainer, TrainerConfig
from simulusion.training.tasks import GenieTask
from simulusion.training.tasks.genie import GenieTaskConfig

config = GenieTaskConfig()
config.data.path = "data/clips"
result = Trainer(TrainerConfig(max_steps=500, precision="bf16")).fit(GenieTask(config))
```

## Tasks

A task is one family's training: its networks, optimizers, where batches come
from and what one unit of progress is. The trainer's budget
(`trainer.max_steps`) counts that unit.

| Task | Trains | Data | Progress | Data parallel |
|---|---|---|---|---|
| `dreamer`, `dreamer-v1`, `dreamer-v2` | world model + actor + critic | environment | env steps | no |
| `dreamer-wm` | Dreamer's world model alone | episode folder | optimizer steps | yes |
| `planet` | RSSM, acting by CEM planning | environment | epochs | no |
| `iris` | tokenizer + transformer + actor-critic (paper settings) | environment | epochs | no |
| `iris-wm` | IRIS's world model alone | episode folder | optimizer steps | yes |
| `diamond` | denoiser + reward model + actor-critic | environment | epochs | no |
| `diamond-wm` | DIAMOND's world model alone | episode folder | optimizer steps | yes |
| `genie` | tokenizer + latent actions + dynamics | episode folder / videos | optimizer steps | yes |
| `vae`, `mdrnn` | World Models' V, then M | episode folder | optimizer steps | yes |
| `controller` | World Models' C, by CMA-ES (real env or dream) | world model | generations | no |
| `dit` | DiT image diffusion | image dataset | optimizer steps | yes |
| `jepa` | I-JEPA image representations | image dataset | optimizer steps | yes |

The `-wm` tasks train a world model **without its agent**: no environment and
no policy, only recorded episodes (your own, or converted from LeRobot; see
{doc}`episodes_guide`). What comes out is a
{py:class}`~simulusion.world_model.WorldModel`, usable by any agent that
speaks the protocol, such as {py:class}`simulusion.agents.PlanningAgent`:

```python
from simulusion.agents import CEMPlanner, PlanningAgent, run_episode

agent = PlanningAgent(
    simulusion.load_world_model("runs/dreamer-wm_20261020-120000/world_model"),
    CEMPlanner(horizon=12, candidates=500),
)
episode_return = run_episode(env, agent)
```

## Run configs

```yaml
task: genie
trainer:              # TrainerConfig: shared by every task
  max_steps: 20000
  precision: auto     # fp32 | bf16 | fp16 | auto
  eval_every: 1000
  checkpoint_every: 2000
  early_stopping_metric: val_recon_loss
model:                # the family's own config (GenieConfig here)
  num_frames: 16
data:                 # where batches come from
  path: data/clips
  val_fraction: 0.05
```

Shipped configs for every task live in `simulusion/configs/runs/`;
`--config <name>` finds them by name.

### Trainer settings

| Field | Meaning |
|---|---|
| `max_steps` | Budget in the task's progress unit; the task's default when unset. |
| `seed`, `deterministic` | Seeds Python, NumPy and PyTorch (offset per process); `deterministic` asks for deterministic kernels. |
| `precision` | `fp32`, `bf16`, `fp16` (loss-scaled) or `auto` (bf16 where supported, fp16 on older GPUs, fp32 on CPU). Replaces the per-family `use_amp` flags for runs on the trainer. |
| `device` | `null` lets Accelerate choose; `cpu` forces the CPU. |
| `grad_accumulation` | Batches per optimizer step (dataset-driven tasks). |
| `log_every`, `eval_every`, `checkpoint_every` | Intervals in progress units. Progress may jump (env steps), so each fires when a multiple is crossed. |
| `keep_checkpoints` | Periodic checkpoints kept; the final one always is. |
| `out_dir`, `run_name` | Run directory: `out_dir/run_name`, timestamped when `run_name` is unset. |
| `resume` | A checkpoint, a run directory, or `auto` (the newest checkpoint of `out_dir/run_name`). |
| `early_stopping_metric`, `early_stopping_mode`, `patience`, `min_delta` | Stop when an evaluation metric stops improving. |
| `console` | Also print metrics to the console (they always go to `metrics.jsonl`). |
| `export_world_model` | Write the world model to `<run>/world_model` at the end. |

## What a run leaves behind

```text
runs/genie_20261020-120000/
    config.yaml            # the fully resolved config: `--config` it to repeat the run
    run.json               # git commit and dirty flag, package versions, GPUs, command line
    metrics.jsonl          # every logged metric (train/..., eval/...)
    checkpoints/
        step_00002000/     # resumable: trainer.resume=<this dir> or resume=auto
    world_model/           # load_world_model(...) reads it
    model/                 # tasks that are not world models (dit, jepa, vae) export here
```

Checkpoints are written to a staging directory and renamed when complete, so
an interrupted write never looks like a checkpoint. They hold the task's
weights, optimizer and loss-scaler states, the loop position and RNG states
(resuming a single-process run is bit-exact), and for online tasks the replay
buffer, so a resumed run trains on the same data. Ctrl+C writes one before
exiting.

## Several processes

Data-parallel tasks run unchanged on several processes, under any launcher
that sets the standard `torch.distributed` environment (`RANK`,
`WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`); Accelerate picks it
up. Batches are sharded across processes and metrics averaged over them.
Modules called through `forward()` are wrapped in DDP; world models driven
through other methods (Dreamer's `rssm.observe_rollout`, IRIS's tokenizer)
start from process 0's weights and average their gradients before every step
instead. Vector-quantizer codebooks, which change outside the optimizer, are
re-broadcast from process 0 after each step. Tasks that collect from an
environment into a replay buffer are single-process and refuse to start
under data parallelism.

This is tested with two CPU processes (gloo): every rank ends with identical
weights. Multi-GPU runs use the same code path but have not been tested yet.
Sharded training (FSDP, DeepSpeed) is refused: gradient clipping and
checkpoints assume every process holds the whole model.

## Writing a task

```python
from dataclasses import dataclass, field

import torch

from simulusion.training import TrainingTask, register_task
from simulusion.training.tasks.common import SequenceDataConfig, sequence_loaders


@dataclass
class MyConfig:
    lr: float = 3e-4
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)


@register_task("my-model")
class MyTask(TrainingTask):
    config_cls = MyConfig  # its fields are the run config's sections
    distributed = True

    def setup(self, trainer):
        self.model = trainer.prepare_module(MyModel())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.update = trainer.gradient_update(
            optimizer, self.model.parameters(), max_grad_norm=1.0, modules=[self.model]
        )
        self.batches, self.val, _ = sequence_loaders(
            self.config.data, trainer, sequence_length=8, image_size=64
        )

    def train_step(self):
        batch = next(self.batches)
        with self.trainer.precision.autocast():
            loss = self.model(batch["frames"])
        self.update(loss)
        return {"loss": loss.item()}

    def state_dict(self):
        return {"model": self.trainer.unwrap(self.model).state_dict()}

    def load_state_dict(self, state):
        self.trainer.unwrap(self.model).load_state_dict(state["model"])
```

Optional hooks: `progress()` (a budget in env steps or epochs),
`evaluate()`, `final_evaluation()`, `world_model()` (export through the
protocol), `export(directory)` (other models), `save_checkpoint()` /
`load_checkpoint()` (state too large for one file, such as a replay buffer)
and `close()`.

## The older entrypoints

`DreamerAgent.train()`, `IRISTrainer.train()`, `DiamondAgent.train()`,
`train_dit`, `train_jepa`, `train_convae` and the rest still work and use the
same update code (and the same `GradientUpdate`), so their results are
unchanged. New features (resume, data parallelism, run directories) land on
the trainer.
