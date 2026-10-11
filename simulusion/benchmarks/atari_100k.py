import numpy as np
from simulusion.utils.device import default_device_name
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from simulusion.configs.diamond_config import (
    DiamondConfig,
    ATARI_100K_GAMES,
)
from simulusion.training.train_diamond import DiamondAgent


def evaluate_atari_100k(
    game: str,
    num_seeds: int = 5,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate DIAMOND on a single Atari game following the Atari 100k protocol.

    Args:
        game: Game name (e.g., "Breakout-v5")
        num_seeds: Number of random seeds to evaluate
        checkpoint_path: Optional path to load checkpoint

    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    hns_scores = []

    for seed in tqdm(range(num_seeds), desc=f"Evaluating {game}"):
        config = DiamondConfig(
            game=game,
            seed=seed,
            device=default_device_name(),
        )

        agent = DiamondAgent(config)

        if checkpoint_path:
            agent.load_checkpoint(checkpoint_path)

        agent.train()

        eval_score = agent.evaluate(num_episodes=5)
        hns = agent._compute_human_normalized_score(eval_score)

        scores.append(eval_score)
        hns_scores.append(hns)

    return {
        "game": game,
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "hns_scores": hns_scores,
        "mean_hns": np.mean(hns_scores),
        "std_hns": np.std(hns_scores),
    }


def run_atari_100k_benchmark(
    games: List[str] = ATARI_100K_GAMES,
    num_seeds: int = 5,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Run the full Atari 100k benchmark evaluation.

    Args:
        games: List of games to evaluate
        num_seeds: Number of seeds per game
        checkpoint_dir: Directory to save/load checkpoints

    Returns:
        Dictionary with results for each game
    """
    results = {}

    for game in tqdm(games, desc="Benchmark"):
        game_results = evaluate_atari_100k(
            game=game,
            num_seeds=num_seeds,
            checkpoint_path=f"{checkpoint_dir}/{game}" if checkpoint_dir else None,
        )
        results[game] = game_results

    return results


def compute_aggregate_metrics(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all games.

    Args:
        results: Results from run_atari_100k_benchmark

    Returns:
        Dictionary with aggregate metrics
    """
    all_hns: List[float] = []
    superhuman_count = 0

    for game, game_results in results.items():
        hns_scores = game_results["hns_scores"]
        all_hns.extend(hns_scores)

        if any(hns >= 1.0 for hns in hns_scores):
            superhuman_count += 1

    all_hns_arr = np.array(all_hns)

    return {
        "mean_hns": float(np.mean(all_hns_arr)),
        "median_hns": float(np.median(all_hns_arr)),
        "iqm_hns": float(np.mean(np.percentile(all_hns_arr, [25, 50, 75]))),
        "superhuman_games": int(superhuman_count),
        "total_games": int(len(results)),
    }


def print_results_table(results: Dict[str, Dict], aggregate: Dict[str, Any]) -> None:
    """Print a formatted table of results."""
    print("\n" + "=" * 100)
    print(f"{'Game':<20} {'Mean Score':>12} {'Std':>8} {'Mean HNS':>10} {'Std HNS':>8}")
    print("=" * 100)

    for game, game_results in results.items():
        print(
            f"{game:<20} "
            f"{game_results['mean_score']:>12.1f} "
            f"{game_results['std_score']:>8.1f} "
            f"{game_results['mean_hns']:>10.3f} "
            f"{game_results['std_hns']:>8.3f}"
        )

    print("=" * 100)
    print("\nAggregate Metrics:")
    print(f"  Mean HNS: {aggregate['mean_hns']:.3f}")
    print(f"  Median HNS: {aggregate['median_hns']:.3f}")
    print(f"  IQM HNS: {aggregate['iqm_hns']:.3f}")
    print(
        f"  Superhuman Games: {aggregate['superhuman_games']}/{aggregate['total_games']}"
    )
    print()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cli_cfg = OmegaConf.from_cli()
    game = cli_cfg.get("game", None)
    run_benchmark = cli_cfg.get("benchmark", False)
    num_seeds = int(cli_cfg.get("num_seeds", 5))

    if game:
        results = evaluate_atari_100k(game, num_seeds=num_seeds)
        print(f"\nResults for {game}:")
        print(f"  Mean Score: {results['mean_score']:.1f} ± {results['std_score']:.1f}")
        print(f"  Mean HNS: {results['mean_hns']:.3f} ± {results['std_hns']:.3f}")

    elif run_benchmark:
        results = run_atari_100k_benchmark(num_seeds=num_seeds)
        aggregate = compute_aggregate_metrics(results)
        print_results_table(results, aggregate)
