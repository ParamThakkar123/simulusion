from __future__ import annotations

from typing import Any, List

import numpy as np

from simulusion.configs.diamond_config import DiamondConfig
from simulusion.configs.iris_config import IRISConfig
from simulusion.training.train_diamond import DiamondAgent
from simulusion.training.train_iris import IRISTrainer
from simulusion.configs.dreamer_config import DreamerConfig
from simulusion.models.dreamer import DreamerAgent


class BaseAdapter:
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        self.env_spec = env_spec
        self.seed = seed

    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    def evaluate(self, num_episodes: int = 1, render: bool = False) -> dict:
        """Return standardized output. Preferred format: dict with key 'episode_returns' -> List[float]"""
        raise NotImplementedError


class DiamondAdapter(BaseAdapter):
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(env_spec, seed)
        # env_spec can be a dict with keys like 'game', or a simple string game name
        if isinstance(env_spec, dict):
            game = env_spec.get("game", None)
        elif isinstance(env_spec, str):
            game = env_spec
        else:
            game = getattr(env_spec, "game", None)

        preset = kwargs.get("preset", None)
        device = kwargs.get("device", "cuda")

        cfg = DiamondConfig(preset=preset)
        if game:
            cfg.game = game
        cfg.seed = seed
        cfg.device = device

        self.agent = DiamondAgent(cfg)

    def load_checkpoint(self, path: str) -> None:
        try:
            self.agent.load_checkpoint(path)
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False) -> dict:
        # DiamondAgent.evaluate returns a float mean reward per episode by default.
        # To produce per-episode returns we run single-episode evaluations repeatedly.
        episode_returns: List[float] = []
        for _ in range(num_episodes):
            try:
                r = self.agent.evaluate(num_episodes=1)
                # If evaluate returns an average scalar, treat as single-episode reward
                if isinstance(r, (int, float)):
                    episode_returns.append(float(r))
                elif isinstance(r, dict) and "episode_returns" in r:
                    vals = r["episode_returns"]
                    if isinstance(vals, (list, tuple)) and len(vals) > 0:
                        episode_returns.extend([float(v) for v in vals])
                    else:
                        # fallback: use mean
                        episode_returns.append(float(np.mean(vals) if vals else 0.0))
                else:
                    # Unknown format: try to coerce
                    try:
                        episode_returns.append(float(r))
                    except Exception:
                        episode_returns.append(0.0)
            except Exception:
                # If per-episode evaluation fails, append zero as safe fallback
                episode_returns.append(0.0)

        return {"episode_returns": episode_returns}


class IRISAdapter(BaseAdapter):
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(env_spec, seed)
        game = None
        if isinstance(env_spec, dict):
            game = env_spec.get("game")
        elif isinstance(env_spec, str):
            game = env_spec

        device = kwargs.get("device", "cuda")
        config = kwargs.get("config", None)
        if config is None:
            cfg = IRISConfig()
        else:
            cfg = config

        # IRISTrainer accepts (game, device, seed, config)
        self.trainer = IRISTrainer(
            game=game or cfg.env, device=device, seed=seed, config=cfg
        )

    def load_checkpoint(self, path: str) -> None:
        # IRIS agent provides save/load on agent; delegate if trainer has agent
        try:
            # IRISAgent implements load(path)
            if hasattr(self.trainer, "agent") and hasattr(self.trainer.agent, "load"):
                self.trainer.agent.load(path)
            else:
                raise AttributeError("IRIS agent does not expose load(path)")
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False) -> dict:
        res = self.trainer.evaluate(num_episodes=num_episodes, render=render)
        if render:
            # (episode_returns_array, videos_list, latents_array)
            ep_returns, videos, latents = res
            return {
                "episode_returns": list(ep_returns),
                "videos": videos,
                "latents": latents,
            }
        else:
            # trainer.evaluate returns a dict with summary keys
            # We convert to repeated mean for compatibility
            if isinstance(res, dict) and "eval_mean_return" in res:
                return {
                    "episode_returns": [
                        float(res["eval_mean_return"]) for _ in range(num_episodes)
                    ]
                }
            return {"episode_returns": []}


class DreamerAdapter(BaseAdapter):
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(env_spec, seed)
        # env_spec can be dict or string. DreamerConfig expects env_backend and env.
        if isinstance(env_spec, dict):
            game = env_spec.get("game")
            env_backend = env_spec.get("env_backend", None)
        elif isinstance(env_spec, str):
            game = env_spec
            env_backend = None
        else:
            game = getattr(env_spec, "game", None)
            env_backend = getattr(env_spec, "env_backend", None)

        algo = kwargs.get("algo", "dreamerv1")
        config = kwargs.get("config", None)
        if config is None:
            cfg = DreamerConfig()
        else:
            cfg = config

        # If a gym-style game string was provided, set backend accordingly
        if game:
            cfg.env = game
            # If the game looks like an Atari id, prefer gym backend
            if isinstance(game, str) and (
                "ALE/" in game or "-v" in game or "-v5" in game
            ):
                cfg.env_backend = "gym"
        if env_backend:
            cfg.env_backend = env_backend

        cfg.algo = (
            "Dreamerv1"
            if algo.lower().startswith("dreamer") and "v2" not in algo.lower()
            else "Dreamerv2"
        )
        cfg.seed = seed

        # Construct DreamerAgent (it will build envs internally)
        self.agent = DreamerAgent(config=cfg)

    def load_checkpoint(self, path: str) -> None:
        try:
            if hasattr(self.agent, "dreamer") and hasattr(
                self.agent.dreamer, "restore_checkpoint"
            ):
                self.agent.dreamer.restore_checkpoint(path)
            else:
                raise AttributeError("Dreamer agent does not expose restore_checkpoint")
        except Exception:
            raise

    def evaluate(self, num_episodes: int = 1, render: bool = False) -> dict:
        # Configure agent's test episodes
        try:
            self.agent.args.test_episodes = int(num_episodes)
        except Exception:
            pass

        # Dreamer exposes dreamer.evaluate(env, eval_episodes, render=False)
        ep_rews, videos, latents = self.agent.dreamer.evaluate(
            self.agent.test_env, int(num_episodes), render=render
        )

        out = {"episode_returns": list(ep_rews)}
        if render:
            out["videos"] = videos
            out["latents"] = latents
        return out


class DreamerV1Adapter(DreamerAdapter):
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        kwargs = dict(kwargs)
        kwargs.setdefault("algo", "dreamerv1")
        super().__init__(env_spec=env_spec, seed=seed, **kwargs)


class DreamerV2Adapter(DreamerAdapter):
    def __init__(
        self, env_spec: Any | None = None, seed: int = 0, **kwargs: Any
    ) -> None:
        kwargs = dict(kwargs)
        kwargs.setdefault("algo", "dreamerv2")
        super().__init__(env_spec=env_spec, seed=seed, **kwargs)
