from simulusion.benchmarks.adapters import IRISAdapter, DiamondAdapter


def test_iris_adapter_smoke() -> None:
    adapter = IRISAdapter(env_spec={"game": "ALE/Pong-v5"}, seed=0)
    res = adapter.evaluate(num_episodes=1)
    assert "episode_returns" in res


def test_diamond_adapter_smoke() -> None:
    adapter = DiamondAdapter(env_spec={"game": "Breakout-v5"}, seed=0)
    res = adapter.evaluate(num_episodes=1)
    assert "episode_returns" in res
