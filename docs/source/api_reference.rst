API Reference
=============

This reference is generated from source docstrings and grouped by workflow. Use
:doc:`world_models_guide` for conceptual explanations and this page for exact
classes, functions, and module-level APIs.

Public package surface
----------------------

These modules expose the most common imports and lazy constructors.

Use ``simulusion`` for common workflows::

   import simulusion
   agent = simulusion.create_model("dreamer", env="walker-walk")

**Primary modules:** ``simulusion``, ``simulusion.models``, ``simulusion.configs``, ``simulusion.catalog``, and ``simulusion.envs``.

.. automodule:: simulusion
   :no-index:

.. automodule:: simulusion.api
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.export
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.runtime
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.precision
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.steppers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.quantize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.inference.bundle
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.world_model
   :members: as_world_model, load_world_model, list_world_model_families, world_model_class

.. automodule:: simulusion.world_model.protocol
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.dreamer
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.planet
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.iris
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.diamond
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.genie
   :members:
   :show-inheritance:

.. automodule:: simulusion.world_model.vae_mdrnn
   :members:
   :show-inheritance:

.. automodule:: simulusion.models
   :no-index:

.. automodule:: simulusion.catalog
   :members:
   :undoc-members:
   :show-inheritance:

Model catalog
-------------

Core model families
~~~~~~~~~~~~~~~~~~~

**Key classes:** ``Dreamer``, ``DreamerAgent``, ``RSSM``, ``RecurrentStateSpaceModel``, ``Planet``, ``ModularRSSM``, ``JEPAAgent``, ``VisionTransformer``, ``IRISAgent``, ``IRISTransformer``, ``IRISWorldModel``, ``Genie``, ``LatentActionModel``, and ``DynamicsModel``.

.. automodule:: simulusion.models.dreamer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.dreamer_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.mdrnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.modular_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.jepa_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.vit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.iris_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.iris_transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.latent_action_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.dynamics_model
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion and DIAMOND components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key classes:** ``DiamondAgent``, ``DDPM``, ``DiT``, ``DiffusionUNet``, ``EDMPreconditioner``, ``EulerSampler``, ``RewardTerminationModel``, and ``ActorCriticNetwork``.

DIAMOND exposes ``DiamondAgent`` from ``simulusion.training.train_diamond``; there is no separate ``DIAMONDAgent`` class name in the package.

.. automodule:: simulusion.models.diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.diffusion.DDPM
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.diffusion.DiT
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.diffusion.diamond_diffusion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.diffusion.reward_termination
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.models.diffusion.actor_critic
   :members:
   :undoc-members:
   :show-inheritance:

Vision, tokenization, and layers
--------------------------------

**Key classes:** ``ConvEncoder``, ``ConvDecoder``, ``DenseDecoder``, ``ActionDecoder``, ``CNNEncoder``, ``CNNDecoder``, ``IRISEncoder``, ``IRISDecoder``, ``DiscreteAutoencoder``, ``VectorQuantizer``, ``VectorQuantizerEMA``, ``VideoTokenizer``, ``MultiHeadSelfAttention``, and ``STTransformer``.

.. automodule:: simulusion.vision.VAE.ConvVAE
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.dreamer_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.dreamer_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.planet_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.planet_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.iris_encoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.iris_decoder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.vq_layer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.vision.video_tokenizer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.blocks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.blocks.mhsa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.blocks.st_transformer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration objects
---------------------

.. automodule:: simulusion.configs
   :no-index:

.. automodule:: simulusion.configs.wm_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.dreamer_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.jepa_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.iris_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.genie_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.dit_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.configs.diamond_config
   :members:
   :undoc-members:
   :show-inheritance:

Training entry points
---------------------

**Key classes and functions:** ``DiamondAgent``, ``train_diamond``, ``train_dreamer``, ``GenieTrainer``, ``IRISTrainer``, and related training entry points.

.. automodule:: simulusion.training
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_world_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_convvae
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_mdn_rnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_iris
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_genie
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_planet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_rssm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.train_diamond
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.training.rl_harness
   :members:
   :undoc-members:
   :show-inheritance:

Memory and controllers
----------------------

.. automodule:: simulusion.memory.dreamer_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.memory.planet_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.memory.iris_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.controller.rssm_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.controller.iris_policy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.controller.rollout_generator
   :members:
   :undoc-members:
   :show-inheritance:

Datasets, environments, and transforms
--------------------------------------

Environment adapters
~~~~~~~~~~~~~~~~~~~~

The environment APIs below mirror the dedicated environment guide pages: DMC,
DeepMind Lab, Gym/Gymnasium, Atari/ALE, Procgen, MuJoCo, Unity ML-Agents, and vectorization utilities.
DIAMOND-style Atari support is intentionally not listed as an environment
adapter because it is Atari preprocessing rather than a separate environment
family.

.. automodule:: simulusion.envs
   :no-index:

.. automodule:: simulusion.envs.dmc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.dmlab
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.gym_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.ale_atari_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.ale_atari_vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.procgen_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.mujoco_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.robotics_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.unity_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.vector_env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Atari preprocessing helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These helpers wrap Atari environments for specific training recipes. They are
not separate environment families.

.. automodule:: simulusion.envs.diamond_atari
   :members:
   :undoc-members:
   :show-inheritance:

Datasets and transforms
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: simulusion.datasets.wm_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.video_datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.tinyworlds
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.diamond_dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.cifar10
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.imagenet1k
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.datasets.nuplan
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.transforms.image
   :members:
   :undoc-members:
   :show-inheritance:

Masking and JEPA helpers
------------------------

.. automodule:: simulusion.masks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.masks.default
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.masks.multiblock
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.masks.random
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.helpers.jepa_helper
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarks and reports
----------------------

.. automodule:: simulusion.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.benchmarks.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.benchmarks.adapters
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.benchmarks.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.benchmarks.reporting
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: simulusion.losses.convae_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.losses.gmm_loss
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.train_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.dreamer_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.jepa_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.memory_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.logging_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: simulusion.utils.utils
   :members:
   :undoc-members:
   :show-inheritance:
