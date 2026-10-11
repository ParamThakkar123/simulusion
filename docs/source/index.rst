Simulusion Documentation
========================

Simulusion is a modular PyTorch library for world models: agents that learn how
their environment works, then plan, act or generate inside that learned model.
Dreamer, PlaNet, DIAMOND, IRIS, Genie, DiT and I-JEPA share one API, and their
encoders, dynamics backbones and decoders are reusable parts.

.. raw:: html

   <div class="demo-grid">
     <a class="demo-card" href="gallery.html#gallery-dreamer">
       <video src="_static/gallery/dreamer_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
       <span class="demo-card-body"><span class="demo-card-title">Dreamer</span><span class="demo-card-text">Real environment beside the world model's open-loop imagination.</span></span>
     </a>
     <a class="demo-card" href="gallery.html#gallery-diamond">
       <video src="_static/gallery/diamond_dream.mp4" autoplay loop muted playsinline preload="metadata"></video>
       <span class="demo-card-body"><span class="demo-card-title">DIAMOND</span><span class="demo-card-text">Breakout generated frame by frame by a diffusion model.</span></span>
     </a>
     <a class="demo-card" href="gallery.html#gallery-genie">
       <video src="_static/gallery/genie_replay.mp4" autoplay loop muted playsinline preload="metadata"></video>
       <span class="demo-card-body"><span class="demo-card-title">Genie</span><span class="demo-card-text">A playable world learned from unlabelled video.</span></span>
     </a>
   </div>

Every model has a demo you can run on a laptop GPU. See the :doc:`gallery` for
all of them.

Install
-------

.. code-block:: bash

   pip install "simulusion[gym]"

Train an agent
--------------

.. code-block:: python

   import simulusion

   # Dreamer from pixels on Pendulum; runs on ``pip install simulusion[gym]``.
   agent = simulusion.create_model(
       "dreamer", env="Pendulum-v1", env_backend="gym", total_steps=5_000
   )
   agent.train()

   print(simulusion.list_models())  # every model create_model() can build

Continue with :doc:`getting_started` for a guided first run, or jump to a
model under *Algorithms* in the sidebar.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   getting_started
   installation
   gallery

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   public_api
   training_guide
   inference_guide
   deployment_guide
   evaluation_guide
   memory_guide
   environments_guide
   environments/index
   datasets/nuplan
   cli
   package_overview
   controllers_guide
   world_models_guide
   tutorials/world_model_env_rl_libraries
   modular_rssm_guide
   vision_guide
   datasets_guide
   losses_guide
   plugin_registry
   world_models_deep_dive

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   dreamer
   planet
   jepa
   iris
   dit
   diamond
   genie

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api_reference
   api_exports
   configs_reference
   export_guide

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   benchmarks
   run_code_in_docs
