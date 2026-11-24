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

    if name in (
        "MinecraftDiamondEnv",
        "make_minecraft_diamond_env",
        "DIAMOND_ACTIONS",
        "DIAMOND_MILESTONES",
    ):
        from . import minecraft_diamond

        return getattr(minecraft_diamond, name)

    if name in ("BraxImageEnv", "make_brax_env"):
        from .brax_env import BraxImageEnv, make_brax_env

        if name == "BraxImageEnv":
            return BraxImageEnv
        return make_brax_env

    if name in (
        "DMLabEnv",
        "make_dmlab_env",
        "DMLAB_LEVELS",
        "DMLAB30_LEVELS",
        "resolve_dmlab_level",
        "POPART_ACTION_SET",
        "IMPALA_ACTION_SET",
        "embed_instruction",
    ):
        from . import dmlab

        return getattr(dmlab, name)

    if name in (
        "BSuiteImageEnv",
        "make_bsuite_env",
        "list_available_bsuite_ids",
        "list_bsuite_settings",
        "normalize_bsuite_id",
    ):
        from . import bsuite_env

        return getattr(bsuite_env, name)

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

    _BENCHMARK_ENV_MODULES = {
        "CRAFTER_ACHIEVEMENTS": "crafter_env",
        "CrafterEnv": "crafter_env",
        "crafter_score": "crafter_env",
        "crafter_score_from_rates": "crafter_env",
        "crafter_success_rates": "crafter_env",
        "make_crafter_env": "crafter_env",
        "MEMORY_MAZE_TASKS": "memory_maze_env",
        "MemoryMazeEnv": "memory_maze_env",
        "make_memory_maze_env": "memory_maze_env",
        "ATARI_100K_PROTOCOLS": "atari100k_env",
        "ATARI_100K_ROMS": "atari100k_env",
        "Atari100kEnv": "atari100k_env",
        "AtariProtocol": "atari100k_env",
        "list_atari100k_games": "atari100k_env",
        "make_atari100k_env": "atari100k_env",
        "CRAFTAX_ENVS": "craftax_env",
        "CraftaxEnv": "craftax_env",
        "make_craftax_env": "craftax_env",
        "GOAL_ENVS": "goal",
        "GoalEnv": "goal",
        "goal_env_name": "goal",
        "make_goal_env": "goal",
        "TwoRoomEnv": "two_room",
        "make_two_room_env": "two_room",
        "PushTEnv": "pusht",
        "make_pusht_env": "pusht",
        "OGBenchCubeEnv": "ogbench_cube",
        "make_ogbench_cube_env": "ogbench_cube",
        "ReacherGoalEnv": "reacher_goal",
        "make_reacher_goal_env": "reacher_goal",
    }
    if name in _BENCHMARK_ENV_MODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{_BENCHMARK_ENV_MODULES[name]}")
        return getattr(module, name)

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
    if backend in {"minecraft", "minecraft_diamond", "minerl"}:
        from .minecraft_diamond import make_minecraft_diamond_env

        return make_minecraft_diamond_env(env_id, **kwargs)
    if backend in {"procgen", "coinrun"}:
        from .procgen_env import make_procgen_env

        return make_procgen_env(env_id, **kwargs)
    if backend in {"dmc", "dm_control", "dm-control", "deepmind_control"}:
        from .dmc import make_dmc_env

        return make_dmc_env(env_id, **kwargs)
    if backend in {"crafter"}:
        from .crafter_env import make_crafter_env

        return make_crafter_env(env_id, **kwargs)
    if backend in {"memory_maze", "memory-maze", "memorymaze"}:
        from .memory_maze_env import make_memory_maze_env

        return make_memory_maze_env(env_id, **kwargs)
    if backend in {"atari100k", "atari_100k", "atari-100k"}:
        from .atari100k_env import make_atari100k_env

        return make_atari100k_env(env_id, **kwargs)
    if backend in {"craftax"}:
        from .craftax_env import make_craftax_env

        return make_craftax_env(env_id, **kwargs)
    if backend in {"two_room", "two-room", "tworoom"}:
        from .two_room import make_two_room_env

        return make_two_room_env(env_id, **kwargs)
    if backend in {"pusht", "push_t", "push-t"}:
        from .pusht import make_pusht_env

        return make_pusht_env(env_id, **kwargs)
    if backend in {"ogbench_cube", "ogbench-cube", "cube"}:
        from .ogbench_cube import make_ogbench_cube_env

        return make_ogbench_cube_env(env_id, **kwargs)
    if backend in {"reacher_goal", "reacher-goal"}:
        from .reacher_goal import make_reacher_goal_env

        return make_reacher_goal_env(env_id, **kwargs)

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
    # Minecraft Diamond (DreamerV3 benchmark, MineRL 0.4.4)
    "MinecraftDiamondEnv",
    "make_minecraft_diamond_env",
    "DIAMOND_ACTIONS",
    "DIAMOND_MILESTONES",
    "UnityMLAgentsEnv",
    "make_unity_mlagents_env",
    "DeepMindControlEnv",
    "BraxImageEnv",
    "make_brax_env",
    "DMLabEnv",
    "make_dmlab_env",
    "DMLAB_LEVELS",
    "DMLAB30_LEVELS",
    "resolve_dmlab_level",
    "POPART_ACTION_SET",
    "IMPALA_ACTION_SET",
    "embed_instruction",
    "BSuiteImageEnv",
    "make_bsuite_env",
    "list_available_bsuite_ids",
    "list_bsuite_settings",
    "normalize_bsuite_id",
    # Benchmark environments (Crafter, Memory Maze, Atari 100k, Craftax)
    "CRAFTER_ACHIEVEMENTS",
    "CrafterEnv",
    "crafter_score",
    "crafter_score_from_rates",
    "crafter_success_rates",
    "make_crafter_env",
    "MEMORY_MAZE_TASKS",
    "MemoryMazeEnv",
    "make_memory_maze_env",
    "ATARI_100K_PROTOCOLS",
    "ATARI_100K_ROMS",
    "Atari100kEnv",
    "AtariProtocol",
    "list_atari100k_games",
    "make_atari100k_env",
    "CRAFTAX_ENVS",
    "CraftaxEnv",
    "make_craftax_env",
    # Goal-conditioned environments (Two-Room, Push-T, OGBench Cube, Reacher)
    "GOAL_ENVS",
    "GoalEnv",
    "goal_env_name",
    "make_goal_env",
    "TwoRoomEnv",
    "make_two_room_env",
    "PushTEnv",
    "make_pusht_env",
    "OGBenchCubeEnv",
    "make_ogbench_cube_env",
    "ReacherGoalEnv",
    "make_reacher_goal_env",
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
