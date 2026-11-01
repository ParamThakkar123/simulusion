from typing import Any, Optional
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_atari_env(
    env_id: str,
    obs_type: str = "rgb",
    frameskip: int = 4,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    max_episode_steps: Optional[int] = None,
    **kwargs: Any,
) -> gym.Env:
    """
    Create any Atari environment from Arcadic Learning Environment (ALE).

    Args:
        env_id (str): The id of the Atari environment to create.
        obs_type (str): The type of observation to return ("rgb" or "ram").
        frameskip (int): The number of frames to skip between actions.
        repeat_action_probability (float): The probability of repeating the last action.
        full_action_space (bool): Whether to use the full action space.
        max_episode_steps (Optional[int]): Maximum number of steps per episode.
        **kwargs: Additional keyword arguments for environment configuration.

    Returns:
        gym.Env: The created Atari environment.
    """

    return gym.make(
        env_id,
        obs_type=obs_type,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def list_available_atari_envs() -> list[str]:
    """
    Get a list of all available Atari environments in Arcadic Learning Environment (ALE).

    Returns:
        list[str]: List of available Atari environment IDs.
    """

    all_envs = list(gym.envs.registry.keys())
    atari_envs = [env for env in all_envs if env.startswith("ALE/")]
    return sorted(atari_envs)
