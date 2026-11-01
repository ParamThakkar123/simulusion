from __future__ import annotations

from typing import Dict, Any
import csv
import json


def export_csv(results: Dict[str, Any], path: str) -> None:
    seeds = results.get("seeds", {})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean", "std", "episode_returns"])
        for seed, data in seeds.items():
            writer.writerow(
                [
                    seed,
                    data.get("mean", ""),
                    data.get("std", ""),
                    json.dumps(data.get("episode_returns", [])),
                ]
            )


def export_markdown(results: Dict[str, Any], path: str) -> None:
    seeds = results.get("seeds", {})
    lines = []
    lines.append("| seed | mean | std | episode_returns |")
    lines.append("|---:|---:|---:|:---|")
    for seed, data in seeds.items():
        # format episode returns compactly
        er = data.get("episode_returns", [])
        er_str = ", ".join([f"{v:.1f}" for v in er]) if er else "[]"
        mean = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        lines.append(f"| {seed} | {mean:.3f} | {std:.3f} | {er_str} |")

    agg = results.get("aggregate", {})
    lines.append("")
    lines.append("**Aggregate**")
    lines.append(f"Mean: {agg.get('mean', 0.0):.3f}  ")
    lines.append(f"Median: {agg.get('median', 0.0):.3f}  ")
    lines.append(f"IQM: {agg.get('iqm', 0.0):.3f}  ")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def export_latex(
    results: Dict[str, Any], path: str, caption: str = "Benchmark results"
) -> None:
    seeds = results.get("seeds", {})
    agg = results.get("aggregate", {})

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    # header row (LaTeX uses \\\\ to end a row)
    lines.append("Seed & Mean & Std & Episode Returns " + "\\\\")
    lines.append("\\midrule")

    for seed, data in seeds.items():
        er = data.get("episode_returns", [])
        er_str = ", ".join([f"{v:.1f}" for v in er]) if er else "--"
        mean = data.get("mean", 0.0)
        std = data.get("std", 0.0)
        lines.append(f"{seed} & {mean:.1f} & {std:.1f} & {er_str} " + "\\\\")

    lines.append("\\midrule")
    lines.append(f"Aggregate IQM & {agg.get('iqm', 0.0):.3f} & & " + "\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
