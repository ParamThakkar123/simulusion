from typing import Any, Optional, cast
from ale_py.vector_env import AtariVectorEnv


def make_atari_vector_env(
    game: str,
    num_envs: int,
    obs_type: str = "rgb",
    frameskip: int = 4,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> AtariVectorEnv:
    """
    Create vectorized Atari environments using ALE's native AtariVectorEnv.

    Args:
        game (str): The name of the Atari game (e.g., "pong", "breakout").
        num_envs (int): Number of parallel environments.
        obs_type (str): The type of observation to return ("rgb" or "ram").
        frameskip (int): The number of frames to skip between actions.
        repeat_action_probability (float): The probability of repeating the last action.
        full_action_space (bool): Whether to use the full action space.
        max_episode_steps (Optional[int]): Maximum number of steps per episode.
        seed (Optional[int]): Random seed for reproducibility.
        **kwargs: Additional keyword arguments for environment configuration.

    Returns:
        AtariVectorEnv: The vectorized Atari environment.
    """

    # ale_py stubs do not accept some keyword args we pass at runtime; create
    # the env and silence the call-arg check since the runtime accepts these
    # parameters. We keep the return type as AtariVectorEnv for callers.
    env: Any = AtariVectorEnv(  # type: ignore[call-arg]
        game=game,
        num_envs=num_envs,
        obs_type=obs_type,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        max_episode_steps=max_episode_steps,
        seed=seed,
        **kwargs,
    )

    return cast(AtariVectorEnv, env)
