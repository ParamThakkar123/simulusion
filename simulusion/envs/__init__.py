from typing import Any
import logging as _logging

logger = _logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name in ("make_atari_env", "list_available_atari_envs"):
        from .ale_atari_env import make_atari_env, list_available_atari_envs

        if name == "make_atari_env":
            return make_atari_env
        return list_available_atari_envs

    if name == "make_atari_vector_env":
        from .ale_atari_vector_env import make_atari_vector_env

        return make_atari_vector_env

    if name in (
        "list_gymnasium_robotics_envs",
        "make_robotics_env",
        "register_gymnasium_robotics_envs",
    ):
        from .robotics_env import (
            list_gymnasium_robotics_envs,
            make_robotics_env,
            register_gymnasium_robotics_envs,
        )

        if name == "list_gymnasium_robotics_envs":
            return list_gymnasium_robotics_envs
        if name == "make_robotics_env":
            return make_robotics_env
        return register_gymnasium_robotics_envs

    if name in (
        "MuJoCoImageEnv",
        "MujocoEnv",
        "make_mujoco_env",
        "make_mujoco_env_from_config",
    ):
        from .mujoco_env import (
            MuJoCoImageEnv,
            make_mujoco_env,
            make_mujoco_env_from_config,
        )

        # ``MujocoEnv`` is the historical spelling kept for the public surface.
        if name in ("MuJoCoImageEnv", "MujocoEnv"):
            return MuJoCoImageEnv
        if name == "make_mujoco_env":
            return make_mujoco_env
        return make_mujoco_env_from_config

    if name in ("GymImageEnv", "make_gym_env"):
        from .gym_env import GymImageEnv, make_gym_env

        if name == "GymImageEnv":
            return GymImageEnv
        return make_gym_env

    if name in ("DreamEnv", "DreamVectorEnv"):
        from .dream import DreamEnv, DreamVectorEnv

        return DreamEnv if name == "DreamEnv" else DreamVectorEnv

    if name in ("WorldModelEnv", "make_world_model_env"):
        from .world_model_env import WorldModelEnv, make_world_model_env

        if name == "WorldModelEnv":
            return WorldModelEnv
        return make_world_model_env

    if name in (
        "PROCGEN_ENVS",
        "ProcgenImageEnv",
        "list_procgen_envs",
        "make_procgen_env",
        "normalize_procgen_env_name",
    ):
        from .procgen_env import (
            PROCGEN_ENVS,
            ProcgenImageEnv,
            list_procgen_envs,
            make_procgen_env,
            normalize_procgen_env_name,
        )

        _map = {
            "PROCGEN_ENVS": PROCGEN_ENVS,
            "ProcgenImageEnv": ProcgenImageEnv,
            "list_procgen_envs": list_procgen_envs,
            "make_procgen_env": make_procgen_env,
            "normalize_procgen_env_name": normalize_procgen_env_name,
        }
        return _map[name]

    if name in (
        "MINECRAFT_ACTION_SET",
        "MINERL_ENVS",
        "MinecraftDiscreteEnv",
        "list_minecraft_actions",
        "list_minecraft_envs",
        "make_minecraft_env",
    ):
        from .minecraft_env import (
            MINECRAFT_ACTION_SET,
            MINERL_ENVS,
            MinecraftDiscreteEnv,
            list_minecraft_actions,
            list_minecraft_envs,
            make_minecraft_env,
        )

        _map = {
            "MINECRAFT_ACTION_SET": MINECRAFT_ACTION_SET,
            "MINERL_ENVS": MINERL_ENVS,
            "MinecraftDiscreteEnv": MinecraftDiscreteEnv,
            "list_minecraft_actions": list_minecraft_actions,
            "list_minecraft_envs": list_minecraft_envs,
            "make_minecraft_env": make_minecraft_env,
        }
        return _map[name]

    if name in ("BraxImageEnv", "make_brax_env"):
        from .brax_env import BraxImageEnv, make_brax_env

        if name == "BraxImageEnv":
            return BraxImageEnv
        return make_brax_env

    if name in ("DMLabEnv", "make_dmlab_env", "DMLAB_LEVELS"):
        from .dmlab import DMLabEnv, make_dmlab_env, DMLAB_LEVELS

        if name == "DMLabEnv":
            return DMLabEnv
        if name == "make_dmlab_env":
            return make_dmlab_env
        return DMLAB_LEVELS

    if name in ("BSuiteImageEnv", "make_bsuite_env", "list_available_bsuite_ids"):
        from .bsuite_env import (
            BSuiteImageEnv,
            make_bsuite_env,
            list_available_bsuite_ids,
        )

        if name == "BSuiteImageEnv":
            return BSuiteImageEnv
        if name == "make_bsuite_env":
            return make_bsuite_env
        return list_available_bsuite_ids

    if name in ("UnityMLAgentsEnv", "make_unity_mlagents_env"):
        from .unity_env import UnityMLAgentsEnv, make_unity_mlagents_env

        if name == "UnityMLAgentsEnv":
            return UnityMLAgentsEnv
        return make_unity_mlagents_env

    if name in (
        "TimeLimit",
        "ActionRepeat",
        "NormalizeActions",
        "ObsDict",
        "OneHotAction",
        "RewardObs",
        "ResizeImage",
        "RenderImage",
        "SelectAction",
    ):
        from .wrappers import (
            TimeLimit,
            ActionRepeat,
            NormalizeActions,
            ObsDict,
            OneHotAction,
            RewardObs,
            ResizeImage,
            RenderImage,
            SelectAction,
        )

        _map = {
            "TimeLimit": TimeLimit,
            "ActionRepeat": ActionRepeat,
            "NormalizeActions": NormalizeActions,
            "ObsDict": ObsDict,
            "OneHotAction": OneHotAction,
            "RewardObs": RewardObs,
            "ResizeImage": ResizeImage,
            "RenderImage": RenderImage,
            "SelectAction": SelectAction,
        }
        return _map[name]

    if name in ("DeepMindControlEnv", "make_dmc_env"):
        from .dmc import DeepMindControlEnv, make_dmc_env

        if name == "DeepMindControlEnv":
            return DeepMindControlEnv
        return make_dmc_env

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_env(env_id: str, **kwargs: Any) -> Any:
    backend = str(kwargs.pop("backend", "")).lower()
    if backend in {"mujoco", "mjcf", "native_mujoco"}:
        from .mujoco_env import make_mujoco_env

        return make_mujoco_env(env_id, **kwargs)
    if backend in {"bsuite", "behavior_suite", "behaviour_suite"}:
        from .bsuite_env import make_bsuite_env

        return make_bsuite_env(env_id, **kwargs)
    if backend in {"robotics", "gymnasium_robotics"}:
        from .robotics_env import make_robotics_env

        return make_robotics_env(env_id, **kwargs)
    if backend in {"world-model", "world_model", "model", "wm"}:
        from .world_model_env import make_world_model_env

        return make_world_model_env(env_id, **kwargs)
    if backend in {"dmlab", "deepmind_lab", "deepmindlab"}:
        from .dmlab import make_dmlab_env

        return make_dmlab_env(env_id, **kwargs)
    if backend in {"procgen", "coinrun"}:
        from .procgen_env import make_procgen_env

        return make_procgen_env(env_id, **kwargs)
    if backend in {"dmc", "dm_control", "dm-control", "deepmind_control"}:
        from .dmc import make_dmc_env

        return make_dmc_env(env_id, **kwargs)

    try:
        from .gym_env import make_gym_env

        return make_gym_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_gym_env could not create %s", env_id, exc_info=True)

    try:
        from .robotics_env import make_robotics_env

        return make_robotics_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_robotics_env could not create %s", env_id, exc_info=True)

    try:
        from .ale_atari_env import make_atari_env

        return make_atari_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_atari_env could not create %s", env_id, exc_info=True)

    try:
        from .procgen_env import make_procgen_env

        return make_procgen_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_procgen_env could not create %s", env_id, exc_info=True)

    try:
        from .unity_env import make_unity_mlagents_env

        return make_unity_mlagents_env(env_id, **kwargs)
    except Exception:
        logger.debug(
            "make_unity_mlagents_env could not create %s", env_id, exc_info=True
        )

    try:
        from .bsuite_env import make_bsuite_env

        return make_bsuite_env(env_id, **kwargs)
    except Exception:
        logger.debug("make_bsuite_env could not create %s", env_id, exc_info=True)

    from simulusion.utils.gym_compat import gym

    return gym.make(env_id, **kwargs)


__all__ = [
    "DeepMindControlEnv",
    "make_dmc_env",
    "make_atari_env",
    "list_available_atari_envs",
    "make_atari_vector_env",
    "MuJoCoImageEnv",
    "MujocoEnv",
    "make_mujoco_env",
    "make_mujoco_env_from_config",
    "list_gymnasium_robotics_envs",
    "make_robotics_env",
    "register_gymnasium_robotics_envs",
    "GymImageEnv",
    "make_gym_env",
    "DreamEnv",
    "DreamVectorEnv",
    "WorldModelEnv",
    "make_world_model_env",
    "PROCGEN_ENVS",
    "ProcgenImageEnv",
    "list_procgen_envs",
    "make_procgen_env",
    "normalize_procgen_env_name",
    # Minecraft (MineRL / MineDojo)
    "MINECRAFT_ACTION_SET",
    "MINERL_ENVS",
    "MinecraftDiscreteEnv",
    "list_minecraft_actions",
    "list_minecraft_envs",
    "make_minecraft_env",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "DeepMindControlEnv",
    "BraxImageEnv",
    "make_brax_env",
    "DMLabEnv",
    "make_dmlab_env",
    "DMLAB_LEVELS",
    "BSuiteImageEnv",
    "make_bsuite_env",
    "list_available_bsuite_ids",
    "TimeLimit",
    "ActionRepeat",
    "NormalizeActions",
    "ObsDict",
    "OneHotAction",
    "RewardObs",
    "ResizeImage",
    "RenderImage",
    "SelectAction",
    "make_env",
]
