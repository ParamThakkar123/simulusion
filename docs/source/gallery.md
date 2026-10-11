# Gallery

What every model in Simulusion looks like in action. Each clip below was produced by
the scripts in [`demos/`](https://github.com/ParamThakkar123/simulusion/tree/main/demos),
trained from scratch on a single **laptop GPU (RTX 3050, 4 GB)** for one to two
hours. They show what the library does in a short budget, not the paper results,
which need far more compute. Every section ends with the commands that
reproduce it.

```{raw} html
<div class="demo-grid">
  <a class="demo-card" href="#gallery-dreamer">
    <video src="_static/gallery/dreamer_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">Dreamer</span><span class="demo-card-text">Learns a world model from pixels and trains its policy inside it.</span></span>
  </a>
  <a class="demo-card" href="#gallery-planet">
    <video src="_static/gallery/planet_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">PlaNet</span><span class="demo-card-text">Plans every action by searching inside its learned latent model.</span></span>
  </a>
  <a class="demo-card" href="#gallery-diamond">
    <video src="_static/gallery/diamond_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">DIAMOND</span><span class="demo-card-text">A diffusion model that generates the next Atari frame.</span></span>
  </a>
  <a class="demo-card" href="#gallery-iris">
    <video src="_static/gallery/iris_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">IRIS</span><span class="demo-card-text">A Transformer that imagines Atari as sequences of discrete tokens.</span></span>
  </a>
  <a class="demo-card" href="#gallery-genie">
    <video src="_static/gallery/genie_replay.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">Genie</span><span class="demo-card-text">A playable world model learned from unlabelled gameplay video.</span></span>
  </a>
  <a class="demo-card" href="#gallery-dit">
    <video src="_static/gallery/dit_denoising.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">DiT</span><span class="demo-card-text">A class-conditional Diffusion Transformer denoising CIFAR-10.</span></span>
  </a>
  <a class="demo-card" href="#gallery-modular-rssm">
    <video src="_static/gallery/modular_rssm_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <span class="demo-card-body"><span class="demo-card-title">ModularRSSM</span><span class="demo-card-text">Swap a world model's backbone and compare the predictions.</span></span>
  </a>
  <a class="demo-card" href="#gallery-i-jepa">
    <img src="_static/gallery/jepa_neighbours.png" alt="I-JEPA nearest neighbours" loading="lazy">
    <span class="demo-card-body"><span class="demo-card-title">I-JEPA</span><span class="demo-card-text">Self-supervised image features, learned without pixels or labels.</span></span>
  </a>
</div>
```

## How to read the world-model clips

Most clips put the **real environment on the left** and the **world model on the
right**. The model is shown a few real frames ("observed"), then its
observations are cut off and it predicts every later frame on its own
("imagined t+k"), given the same actions the agent takes in the real
environment. Nothing on the right is copied from the left: wherever the two
halves drift apart is exactly where the model is wrong.

---

(gallery-dreamer)=
## Dreamer

Dreamer learns a recurrent state-space model of the environment from pixels,
then trains an actor-critic purely on imagined trajectories. Here it is on the
DeepMind Control *cartpole swing-up* task. In 60k environment steps (about
95 minutes) its evaluation return rose from about 75 to 493, out of a maximum of
roughly 880: the policy swings the pole up but cannot yet balance it. The world
model tracks the real cart for about ten imagined steps before it drifts.

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/dreamer_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: the real environment. Right: Dreamer's world model, given 5 real frames and then imagining the next 45 open loop from the policy's actions.</figcaption>
</figure>
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/dreamer_policy.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>The policy, trained only in imagination, swinging the pole up in the real environment.</figcaption>
  </figure>
  <figure class="demo-media">
    <img src="_static/gallery/dreamer_curve.png" alt="Dreamer evaluation return over training" loading="lazy">
    <figcaption>Evaluation return over training.</figcaption>
  </figure>
</div>
```

```bash
python demos/dreamer_demo.py train --env cartpole-swingup --steps 60000
python demos/dreamer_demo.py record
```

**DreamerV2** swaps in symlog two-hot reward and value heads and a balanced KL.
On the same budget its return rose from about 75 to between 165 and 250. It
trained at batch 32 rather than 50, the most that fit beside MuJoCo's renderer
on a 4 GB GPU. Train it with `create_model("dreamer-v2")`, or:

```{raw} html
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/dreamer_v2_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>DreamerV2 on the same task: real environment and open-loop imagination.</figcaption>
  </figure>
  <figure class="demo-media">
    <img src="_static/gallery/dreamer_v2_curve.png" alt="DreamerV2 evaluation return over training" loading="lazy">
    <figcaption>DreamerV2 evaluation return over training.</figcaption>
  </figure>
</div>
```

```bash
python demos/dreamer_demo.py train --algo dreamer-v2 --batch-size 32 --run demos/runs/dreamer_v2
python demos/dreamer_demo.py record --run demos/runs/dreamer_v2
```

See {doc}`dreamer` for the method and the full API.

---

(gallery-planet)=
## PlaNet

PlaNet has no policy network. At every step it searches over action sequences
with the cross-entropy method *inside* its learned latent model and executes the
first action of the best plan. Same task as Dreamer, so the two are comparable:
in 75 minutes its evaluation return rose from 76 to 435, and the recorded
planner episode below scores 388.

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/planet_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: the real environment. Right: PlaNet's latent model predicting open loop from the planner's actions.</figcaption>
</figure>
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/planet_policy.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>The CEM planner controlling the real environment.</figcaption>
  </figure>
  <figure class="demo-media">
    <img src="_static/gallery/planet_curve.png" alt="PlaNet evaluation return over training" loading="lazy">
    <figcaption>Evaluation return over training.</figcaption>
  </figure>
</div>
```

```bash
python demos/planet_demo.py train --minutes 75
python demos/planet_demo.py record
```

See {doc}`planet`.

---

(gallery-diamond)=
## DIAMOND

DIAMOND's world model is a diffusion model: it generates each next Atari frame
by denoising, conditioned on the previous four frames and the action. The agent
is trained entirely on frames the model generates. After 21 epochs (about
1.75 hours) the model renders Breakout's walls, bricks, score and paddle
convincingly, but it has not yet learned the ball, which is one or two pixels at
64x64; small, fast objects are the last thing it picks up. The agent, trained on
those frames, is correspondingly weak.

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/diamond_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: Breakout in the emulator. Right: the diffusion model, started from the same four frames and fed the same actions, generating every later frame from its own previous outputs.</figcaption>
</figure>
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/diamond_play_in_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>The agent playing entirely inside the world model; no emulator is involved.</figcaption>
  </figure>
  <figure class="demo-media">
    <img src="_static/gallery/diamond_curve.png" alt="DIAMOND denoising loss over training" loading="lazy">
    <figcaption>Denoising loss over training.</figcaption>
  </figure>
</div>
```

```bash
python demos/diamond_demo.py train --epochs 21
python demos/diamond_demo.py record
```

To drive the model yourself with the keyboard, run
`simulusion play --model diamond -c demos/runs/diamond/checkpoint_20.pt --game Breakout-v5`
and press `TAB` to switch between the real game and the dream. See {doc}`diamond`.

---

(gallery-iris)=
## IRIS

IRIS compresses every frame into 16 discrete tokens with a VQ autoencoder,
then models the game as a sequence of tokens and actions with a Transformer. The
policy learns from Transformer imagination only.

```{raw} html
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/iris_tokens.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>Each real Pong frame (left) and its reconstruction from 16 tokens (right): the only view of the game the world model gets.</figcaption>
  </figure>
  <figure class="demo-media">
    <video src="_static/gallery/iris_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>The policy acting inside the Transformer's imagination, starting from one real frame.</figcaption>
  </figure>
</div>
<figure class="demo-media">
  <img src="_static/gallery/iris_curve.png" alt="IRIS losses over training" loading="lazy">
  <figcaption>Autoencoder reconstruction loss and the Transformer's next-token loss over training.</figcaption>
</figure>
```

```bash
python demos/iris_demo.py train --minutes 600 --autoencoder-steps 200 --transformer-steps 200 --actor-critic-steps 20
python demos/iris_demo.py record
```

The demo keeps the paper's architecture and its schedule (autoencoder alone
until epoch 25, the policy from epoch 50). Batches are a quarter of the paper's
to fit a 4 GB GPU, and the policy takes 20 updates per epoch instead of 200,
because each one imagines 20 frames token by token. `--resume` continues an
interrupted run. See {doc}`iris`.

---

(gallery-genie)=
## Genie

Genie learns from gameplay video alone, with no actions recorded. A latent
action model discovers a small vocabulary of "controls" from how consecutive
frames differ, and a MaskGIT dynamics model generates the next frame given one.
Trained here for 6,000 steps (about 1.5 hours) on Sonic from the TinyWorlds
dataset.

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/genie_replay.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Left: a real clip. Right: Genie's generation from only its first frame, driven by the latent actions it inferred from the real clip.</figcaption>
</figure>
<figure class="demo-media">
  <video src="_static/gallery/genie_tokens.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Real frames and their reconstruction from Genie's video tokens.</figcaption>
</figure>
```

At this budget Genie reproduces the scene layout, but the scene stays largely
still, and its 8 latent actions are not yet distinct controls: played from the
same frame, they produce nearly the same video. Learning separable actions is
the part of Genie that needs the most training. Frames are sampled at
temperature 0.3 (`record --temperature`).

```bash
python demos/genie_demo.py train --steps 6000
python demos/genie_demo.py record
```

See {doc}`genie`.

---

(gallery-dit)=
## DiT

A Diffusion Transformer (DiT-S/4, 33M parameters) trained class-conditionally
on CIFAR-10 for 40 epochs, about two hours, and sampled with classifier-free
guidance at scale 4.

```{raw} html
<div class="demo-pair">
  <figure class="demo-media">
    <video src="_static/gallery/dit_denoising.mp4" autoplay loop muted playsinline preload="metadata"></video>
    <figcaption>Denoising from pure Gaussian noise over 1000 DDPM steps; one row per class.</figcaption>
  </figure>
  <figure class="demo-media">
    <img src="_static/gallery/dit_samples.png" alt="DiT samples, one row per CIFAR-10 class" loading="lazy">
    <figcaption>Final samples, one row per CIFAR-10 class.</figcaption>
  </figure>
</div>
```

```bash
python demos/dit_demo.py train --epochs 40 --batch-size 64
python demos/dit_demo.py record
```

See {doc}`dit`.

---

(gallery-modular-rssm)=
## ModularRSSM

`ModularRSSM` builds a world model from interchangeable parts: an encoder, a
decoder and a recurrent backbone. The demo trains the same encoder and decoder
with a GRU and with an LSTM backbone on identical data, then lets both predict
the future.

```{raw} html
<figure class="demo-media">
  <video src="_static/gallery/modular_rssm_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
  <figcaption>Real frames, then the GRU and LSTM world models predicting open loop from the same actions.</figcaption>
</figure>
<figure class="demo-media">
  <img src="_static/gallery/modular_rssm_curve.png" alt="Reconstruction error for both backbones" loading="lazy">
  <figcaption>Reconstruction error (half the summed squared error per frame) for both backbones, 25 minutes each.</figcaption>
</figure>
```

```bash
python demos/modular_rssm_demo.py train --minutes-per-backbone 25
python demos/modular_rssm_demo.py record
```

See {doc}`modular_rssm_guide`.

---

(gallery-i-jepa)=
## I-JEPA

I-JEPA learns image features without reconstructing pixels and without labels:
it predicts the *representations* of masked image blocks from a visible context
block. A ViT-Tiny trained for 30 epochs on CIFAR-10 (about an hour) shows
what that buys: classifying test images by their 20 nearest training images in
feature space scores 41.1%, against 29.6% for the same network at random
initialisation (chance is 10%).

```{raw} html
<figure class="demo-media">
  <img src="_static/gallery/jepa_neighbours.png" alt="Test images and their nearest training images in I-JEPA feature space" loading="lazy">
  <figcaption>Each row: a test image (left) and its nearest training images in the encoder's feature space. No labels were used to train the encoder or to find the neighbours.</figcaption>
</figure>
```

```bash
python demos/jepa_demo.py train --epochs 30
python demos/jepa_demo.py record
```

See {doc}`jepa`.

---

## Reproduce everything

```bash
python demos/run_all.py            # trains and records every demo, about thirteen hours on a 4 GB laptop GPU
python demos/build_gallery.py      # copies the media into docs/source/_static/gallery
```

Each demo runs on its own, one after another: several hold gigabytes of replay
data in RAM and most of a 4 GB GPU. See
[`demos/README.md`](https://github.com/ParamThakkar123/simulusion/blob/main/demos/README.md)
for details and per-demo options.
