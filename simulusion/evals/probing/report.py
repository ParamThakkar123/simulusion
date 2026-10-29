"""One call from a world model and a probe dataset to a full probing report.

:func:`probe_world_model` splits the dataset by episode, reads every latent the
model has, scores each one (probes, DCI, MIG), runs the same scores on the
baselines (pixels, the untrained network, one frame of context), and probes
imagined states. The :class:`ProbeReport` it returns saves as JSON and renders
as Markdown.

.. code-block:: python

    report = probe_world_model(wm, data)
    print(report.to_markdown())
    report.save("probe.json")
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from simulusion.evals.probing.data import ProbeDataset, StateTargets
from simulusion.evals.probing.latents import (
    Latents,
    extract_latents,
    pixel_features,
    reinitialized,
)
from simulusion.evals.probing.metrics import (
    ProbeScores,
    fit_linear_probe,
    nanmean,
    scorable,
    score_latents,
)
from simulusion.evals.probing.rollout import rollout_probe

REPORT_FORMAT = "simulusion.probe_report"
REPORT_VERSION = 1


@dataclass
class ProbeRow:
    """Scores for one feature of one latent source.

    ``source`` names what made the latent: the family (``"dreamer"``), a
    baseline (``"pixels"``, ``"dreamer (untrained)"``) or a variant
    (``"dreamer (1 frame)"``). ``targets`` holds per-target R²s.
    """

    source: str
    feature: str
    dims: int
    linear_r2: Optional[float]
    mlp_r2: Optional[float]
    disentanglement: float
    completeness: float
    mig: Optional[float]
    targets: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    @classmethod
    def from_scores(
        cls, source: str, feature: str, dims: int, scores: ProbeScores
    ) -> "ProbeRow":
        mlp = scores.mlp_r2
        return cls(
            source=source,
            feature=feature,
            dims=dims,
            linear_r2=_number(scores.linear),
            mlp_r2=_number(scores.mlp),
            disentanglement=scores.dci.disentanglement,
            completeness=scores.dci.completeness,
            mig=None if scores.mig is None else scores.mig.mig,
            targets={
                name: {
                    "linear": _number(scores.linear_r2[k]),
                    "mlp": None if mlp is None else _number(mlp[k]),
                }
                for k, name in enumerate(scores.names)
            },
        )


@dataclass
class RolloutRow:
    """Probe R² on imagined states of one feature, per horizon."""

    feature: str
    horizons: List[int]
    r2: List[Optional[float]]
    encoded_r2: List[Optional[float]]
    rollouts: int
    targets: Dict[str, List[Optional[float]]] = field(default_factory=dict)


@dataclass
class ProbeReport:
    """Everything :func:`probe_world_model` measured, ready to save or print."""

    task: str
    family: str
    target_names: List[str]
    episodes: Dict[str, int]
    rows: List[ProbeRow]
    rollouts: List[RolloutRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"format": REPORT_FORMAT, "version": REPORT_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ProbeReport":
        if values.get("format") != REPORT_FORMAT:
            raise ValueError("Not a probe report.")
        return cls(
            task=values["task"],
            family=values["family"],
            target_names=list(values["target_names"]),
            episodes=dict(values["episodes"]),
            rows=[ProbeRow(**row) for row in values["rows"]],
            rollouts=[RolloutRow(**row) for row in values.get("rollouts", [])],
            notes=list(values.get("notes", [])),
        )

    def save(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProbeReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def row(self, source: str, feature: str) -> ProbeRow:
        for row in self.rows:
            if row.source == source and row.feature == feature:
                return row
        raise KeyError(f"No row for {source!r} / {feature!r}.")

    def to_markdown(self) -> str:
        split = ", ".join(f"{k} {v}" for k, v in self.episodes.items())
        lines = [
            f"# Probing {self.family} on {self.task}",
            "",
            f"Episodes: {split}. Scores are on the test episodes.",
            "",
            "| Source | Feature | Dims | Linear R² | MLP R² | Disent. | Compl. | MIG |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.source} | {r.feature} | {r.dims} | {_fmt(r.linear_r2)} | "
                f"{_fmt(r.mlp_r2)} | {_fmt(r.disentanglement)} | "
                f"{_fmt(r.completeness)} | {_fmt(r.mig)} |"
            )
        lines += ["", "## Linear R² per target", ""]
        header = [f"{r.source} / {r.feature}" for r in self.rows]
        lines.append("| Target | " + " | ".join(header) + " |")
        lines.append("|---|" + "---:|" * len(header))
        for name in self.target_names:
            cells = [_fmt(r.targets.get(name, {}).get("linear")) for r in self.rows]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        if self.rollouts:
            lines += [
                "",
                "## Imagined states",
                "",
                "| Feature | Horizon | Imagined R² | Encoded R² |",
                "|---|---:|---:|---:|",
            ]
            for rollout in self.rollouts:
                for h, imagined, encoded in zip(
                    rollout.horizons, rollout.r2, rollout.encoded_r2
                ):
                    lines.append(
                        f"| {rollout.feature} | {h} | {_fmt(imagined)} | "
                        f"{_fmt(encoded)} |"
                    )
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def _number(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN (a target that could not be scored) is None."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _fmt(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.3f}"


def probe_world_model(
    model: Any,
    data: ProbeDataset,
    *,
    keys: Optional[Iterable[str]] = None,
    include: Sequence[str] = ("qpos", "qvel"),
    unobservable: Union[str, Iterable[str], None] = "auto",
    splits: tuple[float, float, float] = (0.6, 0.2, 0.2),
    context: Optional[int] = None,
    action_fn: Optional[Callable[[np.ndarray], Any]] = None,
    baselines: bool = True,
    single_frame: bool = True,
    mlp: bool = True,
    rollout: bool = True,
    rollout_keys: Optional[Iterable[str]] = None,
    horizons: Sequence[int] = (1, 5, 15),
    starts_every: int = 10,
    pixel_size: int = 16,
    seed: int = 0,
    mlp_kwargs: Optional[dict] = None,
) -> ProbeReport:
    """Probe every latent of ``model`` for the true state, with baselines.

    Args:
        model: A :class:`~simulusion.world_model.WorldModel`.
        data: A :class:`ProbeDataset`; split by episode into train, val, test.
        keys: Features to score; all of them by default.
        include, unobservable: Which targets, as for :meth:`ProbeDataset.targets`.
        splits: Train / val / test fractions of the episodes.
        context: Frames of history per latent, as for :func:`extract_latents`.
        action_fn: Map DMC actions to the model's; without it a
            discrete-action model encodes without actions and is not rolled out.
        baselines: Also score pixels and the same architecture untrained.
        single_frame: Also score the model with one frame of context, which
            shows what its memory adds (velocities need more than one frame).
        mlp: Fit MLP probes as well as linear ones.
        rollout: Probe imagined states; ``rollout_keys`` picks the features,
            ``"features"`` (or every feature) by default.
        horizons, starts_every: Passed to :func:`rollout_probe`.
        pixel_size: Side of the downsampled pixel baseline.
        seed: Seeds the split and the probes.
    """
    if len(data) < 3:
        raise ValueError("Probing needs at least three episodes: train, val, test.")
    if not math.isclose(sum(splits), 1.0) or min(splits) <= 0:
        raise ValueError("splits must be three positive fractions summing to 1.")
    parts = _split_three(data, splits, seed)
    target_sets = {
        name: split.targets(include, unobservable=unobservable)
        for name, split in parts.items()
    }
    names = list(target_sets["test"].names)
    flat = ~scorable(
        target_sets["test"].values, target_sets["train"].values.std(axis=0)
    )
    family = model.spec.family
    discrete = model.spec.action.kind == "discrete" and action_fn is None
    notes: List[str] = []
    if discrete:
        notes.append(
            f"{family} takes discrete actions and no action_fn was given: latents "
            "were encoded without actions and imagination was not probed."
        )
    if flat.any():
        notes.append(
            "Too little variation in the test episodes to score "
            f"{', '.join(n for n, f in zip(names, flat) if f)}; their R² is "
            "left out. Record more or longer episodes, or a policy that "
            "reaches those states."
        )
    mlp_kwargs = {"seed": seed, **(mlp_kwargs or {})}

    def extract(m: Any, ctx: Optional[int]) -> Dict[str, Latents]:
        return {
            name: extract_latents(
                m,
                split,
                context=ctx,
                actions=not discrete,
                action_fn=action_fn,
                keys=keys,
            )
            for name, split in parts.items()
        }

    def score(source: str, latents: Dict[str, Latents]) -> List[ProbeRow]:
        rows = []
        matched = {n: latents[n].match(target_sets[n]) for n in parts}
        for feature in sorted(latents["train"].features):
            train, val, test = (
                (latents[n].features[feature], matched[n].values)
                for n in ("train", "val", "test")
            )
            scores = score_latents(
                train, val, test, names, mlp=mlp, mlp_kwargs=mlp_kwargs
            )
            rows.append(
                ProbeRow.from_scores(source, feature, train[0].shape[1], scores)
            )
        return rows

    own = extract(model, context)
    rows = score(family, own)
    if single_frame and context != 1:
        if model.spec.context_frames > 1:
            notes.append(
                f"{family} needs {model.spec.context_frames} context frames, so "
                "the one-frame variant was skipped."
            )
        else:
            rows += score(f"{family} (1 frame)", extract(model, 1))
    if baselines:
        rows += score(
            f"{family} (untrained)", extract(reinitialized(model, seed), context)
        )
        rows += score(
            "pixels", {n: pixel_features(s, pixel_size) for n, s in parts.items()}
        )

    rollouts: List[RolloutRow] = []
    if rollout and not discrete:
        features = own["train"].features
        chosen = (
            list(rollout_keys)
            if rollout_keys is not None
            else (["features"] if "features" in features else sorted(features))
        )
        for feature in chosen:
            if feature == "pixels":
                continue
            try:
                row = _rollout_row(
                    model,
                    parts,
                    own,
                    target_sets,
                    feature,
                    horizons=horizons,
                    starts_every=starts_every,
                    context=context,
                    action_fn=action_fn,
                    seed=seed,
                )
            except ValueError as error:
                notes.append(f"Imagination of {feature!r} was not probed: {error}")
                continue
            rollouts.append(row)
    return ProbeReport(
        task=data.task,
        family=family,
        target_names=names,
        episodes={name: len(split) for name, split in parts.items()},
        rows=rows,
        rollouts=rollouts,
        notes=notes,
    )


def _split_three(
    data: ProbeDataset, splits: tuple[float, float, float], seed: int
) -> Dict[str, ProbeDataset]:
    """Disjoint train / val / test datasets of whole episodes, each non-empty."""
    n = len(data)
    order = np.random.default_rng(seed).permutation(n).tolist()
    n_test = max(1, round(n * splits[2]))
    n_val = max(1, round(n * splits[1]))
    if n - n_test - n_val < 1:
        n_test, n_val = 1, 1
    test, val = order[:n_test], order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    return {
        "train": data._subset(sorted(train)),
        "val": data._subset(sorted(val)),
        "test": data._subset(sorted(test)),
    }


def _rollout_row(
    model: Any,
    parts: Dict[str, ProbeDataset],
    latents: Dict[str, Latents],
    target_sets: Dict[str, StateTargets],
    feature: str,
    **kwargs: Any,
) -> RolloutRow:
    fit = {
        n: (latents[n].features[feature], latents[n].match(target_sets[n]).values)
        for n in ("train", "val")
    }
    probe = fit_linear_probe(*fit["train"], *fit["val"])
    scores = rollout_probe(
        model,
        parts["test"],
        probe,
        key=feature,
        targets=target_sets["test"],
        **kwargs,
    )
    return RolloutRow(
        feature=feature,
        horizons=list(scores.horizons),
        r2=[_number(v) for v in scores.mean()],
        encoded_r2=[_number(v) for v in nanmean(scores.encoded_r2, axis=1)],
        rollouts=scores.rollouts,
        targets={
            name: [_number(v) for v in scores.r2[:, k]]
            for k, name in enumerate(scores.names)
        },
    )
