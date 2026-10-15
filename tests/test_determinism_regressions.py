from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

if (
    importlib.util.find_spec("gymnasium") is None
    and importlib.util.find_spec("gym") is None
):
    pytest.skip("gym/gymnasium is not installed", allow_module_level=True)

import numpy as np
import torch
from torch.distributions import Categorical

from simulusion.envs.gym_env import GymImageEnv
from simulusion.models.dreamer_rssm import RSSM
from simulusion.training.rl_harness import PPOTrainer
from simulusion.utils.gym_compat import gym

_DATA_DIR = Path(__file__).resolve().parent / "data"
_GOLDEN_PATH = _DATA_DIR / "regression_baselines.json"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _SeededImageEnv:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(8, 8, 3), dtype=np.uint8
        )
        self._rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        reward = float((int(action) + 1) * 0.25 + self._rng.random())
        done = self._step >= 4
        return self._obs(), reward, done, False, {"step": self._step}

    def render(self, *args, **kwargs):
        return self._obs()

    def _obs(self):
        base = self._rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        return (base + self._step).astype(np.uint8)


def _episode(seed: int):
    env = GymImageEnv(_SeededImageEnv(), seed=seed, size=(8, 8))
    obs = env.reset()["image"]
    trajectory = [(obs.copy(), 0.0, False, np.zeros(3, dtype=np.float32))]
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        trajectory.append((obs["image"].copy(), reward, done, info["action"].copy()))
    return trajectory


def test_same_seed_same_wrapped_episode_same_trajectory():
    first = _episode(seed=123)
    second = _episode(seed=123)

    assert len(first) == len(second)
    for (obs_a, reward_a, done_a, action_a), (obs_b, reward_b, done_b, action_b) in zip(
        first, second
    ):
        assert np.array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        assert done_a == done_b
        assert np.array_equal(action_a, action_b)


class _FixedVecEnv:
    total_envs = 2
    action_space = gym.spaces.Discrete(3)
    observation_space: ClassVar[dict[str, Any]] = {
        "image": gym.spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)
    }


def _ppo_regression_metrics():
    _set_all_seeds(7)
    trainer = PPOTrainer(
        _FixedVecEnv(),
        device="cpu",
        lr=1e-4,
        num_epochs=2,
        batch_size=2,
        entropy_coeff=0.0,
    )
    obs = torch.linspace(0.0, 1.0, steps=4 * 2 * 3 * 64 * 64).view(4, 2, 3, 64, 64)
    trajectories = {
        "obs": obs,
        "actions": torch.tensor([[0, 1], [2, 1], [0, 2], [1, 0]], dtype=torch.long),
        "log_probs": torch.full((4, 2), -math.log(3.0)),
        "rewards": torch.tensor(
            [[0.0, 0.25], [0.5, -0.25], [1.0, 0.75], [-0.5, 0.125]],
            dtype=torch.float32,
        ),
        "values": torch.zeros(4, 2),
        "dones": torch.tensor(
            [[False, False], [False, True], [False, False], [True, False]]
        ),
    }

    flat_obs = trajectories["obs"].view(
        -1, *_FixedVecEnv.observation_space["image"].shape
    )
    flat_actions = trajectories["actions"].view(-1)
    flat_rewards = trajectories["rewards"].view(-1)
    flat_values = trajectories["values"].view(-1)
    flat_dones = trajectories["dones"].view(-1)
    advantages = trainer.compute_gae(flat_rewards, flat_values, flat_dones)
    returns = advantages + flat_values
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    logits, values = trainer.policy(flat_obs)
    dist = Categorical(logits=logits)
    policy_loss = -(
        torch.exp(dist.log_prob(flat_actions) - trajectories["log_probs"].view(-1))
        * normalized_advantages
    ).mean()
    value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)
    loss = policy_loss + trainer.value_coeff * value_loss

    trainer.train_step(trajectories)
    post_logits, post_values = trainer.policy(flat_obs[:2])
    return {
        "pre_loss": float(loss.detach()),
        "reward_sum": float(trajectories["rewards"].sum()),
        "post_logits_mean": float(post_logits.mean().detach()),
        "post_values_mean": float(post_values.mean().detach()),
    }


def _dreamer_rssm_regression_metrics():
    _set_all_seeds(11)
    rssm = RSSM(
        action_size=2,
        stoch_size=4,
        deter_size=8,
        hidden_size=8,
        obs_embed_size=6,
        activation="elu",
    )
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    prev_state = rssm.init_state(batch_size=3, device=torch.device("cpu"))
    prev_action = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    obs_embed = torch.linspace(-0.5, 0.5, steps=18).view(3, 6)
    posterior, prior = rssm.observe_step(prev_state, prev_action, obs_embed)
    prior_dist = rssm.get_dist(prior["mean"], prior["std"])
    posterior_dist = rssm.get_dist(posterior["mean"], posterior["std"])
    kl_loss = torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()
    stoch_loss = posterior["stoch"].pow(2).mean()
    loss = kl_loss + 0.1 * stoch_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "posterior_stoch_mean": float(posterior["stoch"].mean().detach()),
    }


_GENERATORS: dict[str, Callable[[], dict[str, float]]] = {
    "ppo": _ppo_regression_metrics,
    "dreamer_rssm": _dreamer_rssm_regression_metrics,
}


# --- Golden baseline loading / update -------------------------------------------


def _load_golden_cases() -> list:
    # A missing baseline file must not take down collection for the whole
    # suite; the parametrized test then reports as skipped instead.
    if not _GOLDEN_PATH.exists():
        return []
    with open(_GOLDEN_PATH) as f:
        raw = json.load(f)
    cases = []
    for name in raw:
        if name not in _GENERATORS:
            continue
        entry = raw[name]
        metrics = {k: v for k, v in entry.items() if k != "_tolerances"}
        tolerances = entry.get("_tolerances", {})
        cases.append(pytest.param(name, metrics, tolerances, id=name))
    return cases


_UPDATE_GOLDEN = os.environ.get("SIMULUSION_UPDATE_GOLDEN")

if _UPDATE_GOLDEN:
    current = {}
    if _GOLDEN_PATH.exists():
        with open(_GOLDEN_PATH) as f:
            current = json.load(f)
    for name, gen_fn in _GENERATORS.items():
        metrics = gen_fn()
        tolerances = current.get(name, {}).get("_tolerances", {})
        tolerances_new = {k: tolerances.get(k, 1e-6) for k in metrics}
        current[name] = {**metrics, "_tolerances": tolerances_new}
    with open(_GOLDEN_PATH, "w") as f:
        json.dump(current, f, indent=2)
        f.write("\n")

_GOLDEN_CASES = _load_golden_cases()


@pytest.mark.parametrize(
    ("name", "baseline", "tolerances"),
    _GOLDEN_CASES,
)
def test_tiny_model_regression_baselines(name, baseline, tolerances):
    gen_fn = _GENERATORS[name]
    metrics = gen_fn()
    assert metrics.keys() == baseline.keys()
    for metric_name, expected in baseline.items():
        tol = tolerances.get(metric_name, 1e-6)
        assert metrics[metric_name] == pytest.approx(expected, abs=tol), (
            f"{name}.{metric_name} drifted: {metrics}"
        )
