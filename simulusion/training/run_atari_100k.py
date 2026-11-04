"""
IRIS Atari 100k Benchmark Runner

Runs IRIS on all 26 Atari games from the Atari 100k benchmark.
Computes human-normalized scores and compares to baselines.
"""

from simulusion.training.train_iris import IRISTrainer
from simulusion.configs.iris_config import IRISConfig
from simulusion.benchmarks import (
    ATARI100K_GAMES as _GAMES,
    ATARI_HUMAN_SCORES,
    ATARI_RANDOM_SCORES,
    atari_env_id,
    atari_game,
    human_normalized_score,
)
import numpy as np
import json
import os
from typing import Any, List, Dict, Optional, cast


def game_name(game: str) -> str:
    """``"ALE/MsPacman-v5"`` -> ``"MsPacman"``: the key of the score tables."""
    return game.replace("ALE/", "").replace("-v5", "")


# Atari 100k games and reference scores (Wang et al., 2016), from the shared
# benchmark tables (simulusion.benchmarks), keyed as IRIS names them.
ATARI_100K_GAMES = [atari_env_id(g) for g in _GAMES]
HUMAN_SCORES = {
    game_name(g): ATARI_HUMAN_SCORES[atari_game(g)] for g in ATARI_100K_GAMES
}
RANDOM_SCORES = {
    game_name(g): ATARI_RANDOM_SCORES[atari_game(g)] for g in ATARI_100K_GAMES
}


def compute_human_normalized_score(score: float, game: str) -> float:
    """Compute human-normalized score for a game.

    Formula: (score - random) / (human - random)

    Raises:
        KeyError: ``game`` has no human or random reference score.
    """
    if game not in HUMAN_SCORES or game not in RANDOM_SCORES:
        raise KeyError(
            f"No human/random reference score for '{game}'; known games: "
            f"{', '.join(sorted(HUMAN_SCORES))}."
        )
    return human_normalized_score(score, game)


def interquartile_mean(values: List[float]) -> float:
    """Mean of the middle 50% of ``values`` (25% trimmed from each end).

    The IQM of Agarwal et al. (2021), as ``scipy.stats.trim_mean(values,
    0.25)`` computes it.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(0.25 * len(ordered))
    middle = ordered[cut : len(ordered) - cut]
    return float(np.mean(middle)) if len(middle) else 0.0


def run_single_game(
    game: str,
    config: Optional[IRISConfig] = None,
    device: str = "cuda",
    *,
    save_dir: Optional[str] = None,
    eval_episodes: int = 100,
    env: Optional[Any] = None,
) -> Dict:
    """Run IRIS on a single game.

    Args:
        game: Game name (e.g., "ALE/Pong-v5")
        config: IRIS config
        device: Device to run on
        save_dir: The game's checkpoint folder (``checkpoints/iris/<game>``
            by default), so games never write over each other's runs.
        eval_episodes: Episodes of the final evaluation.
        env: An environment to use instead of making ``game``.

    Returns:
        Dictionary with results, or ``{"game", "error"}`` if the run failed.
    """
    print(f"\n{'=' * 50}")
    print(f"Training on: {game}")
    print(f"{'=' * 50}")

    name = game_name(game)
    try:
        # Ensure we have a config object
        cfg = config if config is not None else IRISConfig()

        trainer = IRISTrainer(game=game, device=device, config=cfg, env=env)

        # Train for full 100k steps
        trainer.train(
            total_epochs=cfg.total_epochs,
            eval_interval=100,
            save_dir=save_dir or os.path.join("checkpoints", "iris", name),
        )

        # Evaluate (explicit non-render mode so the return type is a dict)
        eval_metrics = cast(
            Dict[str, float],
            trainer.evaluate(num_episodes=eval_episodes, render=False),
        )

        mean_return_val = float(eval_metrics["eval_mean_return"])
        hns = compute_human_normalized_score(mean_return_val, name)

        result = {
            "game": game,
            "game_name": name,
            "mean_return": mean_return_val,
            "std_return": float(eval_metrics["eval_std_return"]),
            "human_normalized_score": hns,
        }

        print(f"\nResults for {name}:")
        print(f"  Mean return: {mean_return_val:.2f}")
        print(f"  Human-normalized: {hns:.3f}")

        return result

    except Exception as e:
        print(f"Error training on {game}: {e}")
        return {
            "game": game,
            "error": f"{type(e).__name__}: {e}",
        }


def run_atari_100k(
    games: Optional[List[str]] = None,
    config: Optional[IRISConfig] = None,
    device: str = "cuda",
    output_file: str = "results/iris_atari100k.json",
    *,
    save_dir: str = "checkpoints/iris",
    eval_episodes: int = 100,
) -> tuple[list, dict[str, Any]]:
    """Run IRIS on all Atari 100k games.

    Args:
        games: List of games to run (default: all 26)
        config: IRIS config
        device: Device to run on
        output_file: Output JSON file for results. Games that failed are
            listed under ``"errors"`` and left out of the aggregate.
        save_dir: Each game trains in ``<save_dir>/<game>``.
        eval_episodes: Episodes of each game's final evaluation.
    """
    if games is None:
        games = ATARI_100K_GAMES

    if config is None:
        config = IRISConfig()

    results = []
    errors = []

    for game in games:
        result = run_single_game(
            game,
            config,
            device,
            save_dir=os.path.join(save_dir, game_name(game)),
            eval_episodes=eval_episodes,
        )
        if "error" in result:
            errors.append(result)
        else:
            results.append(result)

    hns_scores = [r["human_normalized_score"] for r in results]
    aggregate = {
        "num_games": len(results),
        "num_failed": len(errors),
        "num_superhuman": sum(score > 1.0 for score in hns_scores),
        "mean_hns": float(np.mean(hns_scores)) if hns_scores else 0.0,
        "median_hns": float(np.median(hns_scores)) if hns_scores else 0.0,
        "iqm_hns": interquartile_mean(hns_scores),
    }

    # Save results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "results": results,
                "errors": errors,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 50}")
    print("ATARI 100K BENCHMARK RESULTS")
    print(f"{'=' * 50}")
    print(f"Games tested: {aggregate['num_games']}")
    if errors:
        print(f"Games failed: {', '.join(e['game'] for e in errors)}")
    print(f"Superhuman games: {aggregate['num_superhuman']}")
    print(f"Mean HNS: {aggregate['mean_hns']:.3f}")
    print(f"Median HNS: {aggregate['median_hns']:.3f}")
    print(f"IQM HNS: {aggregate['iqm_hns']:.3f}")
    print(f"\nResults saved to: {output_file}")

    return results, aggregate


def main(argv: Optional[List[str]] = None) -> None:
    """Run the Atari 100k benchmark, with ``key=value`` overrides.

    ``device=cpu``, ``output=<file>``, ``games=[Pong,Breakout]`` (names as
    in the score tables, or full ``ALE/...-v5`` ids), ``save_dir=<dir>`` and
    ``eval_episodes=<n>``.
    """
    from omegaconf import OmegaConf

    from simulusion.utils.device import default_device_name

    cli_cfg = OmegaConf.from_cli(list(argv) if argv is not None else None)
    device = cli_cfg.get("device", default_device_name())
    output_file = cli_cfg.get("output", "results/iris_atari100k.json")
    games = cli_cfg.get("games")
    if games is not None:
        games = [g if "/" in str(g) else f"ALE/{g}-v5" for g in games]

    # Create config optimized for Atari 100k
    config = IRISConfig()
    config.atari_100k = True
    config.max_env_steps = 100000

    # Run benchmark
    run_atari_100k(
        games=games,
        config=config,
        device=device,
        output_file=output_file,
        save_dir=cli_cfg.get("save_dir", "checkpoints/iris"),
        eval_episodes=int(cli_cfg.get("eval_episodes", 100)),
    )


if __name__ == "__main__":
    main()
