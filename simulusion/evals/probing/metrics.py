"""Probes and disentanglement scores for latents against the true state.

A probe regresses the true state from a latent. Its held-out R² says how much
of each state variable the latent holds (*informativeness*); a linear probe
asks whether it is there linearly, an MLP whether it is there at all. The gap
between the two is how entangled it is.

Being decodable is not being discovered. :func:`dci` and :func:`mig` ask
whether each state variable lives in a few latent dimensions and each latent
dimension serves one variable, which is what recovering the generative
factors means.

.. code-block:: python

    train, rest = data.split(0.4)
    val, test = rest.split(0.5)
    z = {s: extract_latents(wm, s).features["deter"] for s in (train, val, test)}
    y = {s: ... targets for the same rows ...}
    probe = fit_linear_probe(z[train], y[train], z[val], y[val])
    probe.r2(z[test], y[test])                      # (K,) held-out R²
    dci(probe, names, z[test], y[test]).disentanglement

Everything is in NumPy and PyTorch; no scikit-learn.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np
import torch

DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-3, 5, 17))

#: A target whose spread on the scored frames is below this share of its
#: training spread gets no R²: SST is then so small that any error makes R²
#: hugely negative (a pole that hangs still all test episode, say).
MIN_SPREAD_RATIO = 0.1


def scorable(
    y: np.ndarray, reference_std: np.ndarray, ratio: float = MIN_SPREAD_RATIO
) -> np.ndarray:
    """Which target columns vary enough in ``y`` for R² to mean anything.

    ``reference_std`` is each column's standard deviation where the probe was
    fitted; a column is scorable when its spread in ``y`` is at least
    ``ratio`` of that. A column that does not vary at all never is.
    """
    spread = np.asarray(y, np.float64).std(axis=0)
    floor = np.maximum(ratio * np.asarray(reference_std, np.float64), 1e-8)
    return spread >= floor


def nanmean(values: np.ndarray, axis: Optional[int] = None) -> Any:
    """Mean over the scorable entries; NaN where there are none."""
    values = np.asarray(values, np.float64)
    count = np.sum(~np.isnan(values), axis=axis)
    total = np.nansum(values, axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def r2_score(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """R² per column: ``1 - SSE / SST`` around ``y``'s own mean.

    A column that never changes scores 1 when predicted exactly and 0 otherwise.
    """
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    sse = ((y - pred) ** 2).sum(axis=0)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    out = np.where(sst > 0, 1.0 - sse / np.where(sst > 0, sst, 1.0), 0.0)
    return np.where((sst == 0) & (sse <= 1e-12), 1.0, out)


@dataclass
class _Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "_Scaler":
        x = np.asarray(x, np.float64)
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std > 1e-12, std, 1.0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) / self.std


@dataclass
class LinearProbe:
    """Ridge regression from standardised latents to standardised targets.

    Attributes:
        weight: ``(D, K)`` weights in standardised units, so their magnitudes
            compare across latent dimensions (the DCI importances).
        alpha: ``(K,)`` the ridge penalty chosen for each target.
    """

    weight: np.ndarray
    alpha: np.ndarray
    z_scale: _Scaler
    y_scale: _Scaler

    def predict(self, z: np.ndarray) -> np.ndarray:
        standard = self.z_scale(z) @ self.weight
        return standard * self.y_scale.std + self.y_scale.mean

    def r2(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        return r2_score(y, self.predict(z))


def fit_linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> LinearProbe:
    """Fit ridge probes, choosing each target's penalty on the validation set.

    One thin SVD of the training latents gives the solution for every penalty,
    so the sweep costs little more than a single fit, and it works whether
    there are more latent dimensions or more frames.
    """
    z_scale, y_scale = _Scaler.fit(z_train), _Scaler.fit(y_train)
    zs, ys = z_scale(z_train), y_scale(y_train)
    zv, yv = z_scale(z_val), y_scale(y_val)
    u, s, vt = np.linalg.svd(zs, full_matrices=False)
    uty = u.T @ ys  # (r, K)
    best_err = np.full(ys.shape[1], np.inf)
    weight = np.zeros((zs.shape[1], ys.shape[1]))
    chosen = np.zeros(ys.shape[1])
    for alpha in alphas:
        w = vt.T @ ((s / (s**2 + alpha))[:, None] * uty)
        err = ((zv @ w - yv) ** 2).mean(axis=0)
        better = err < best_err
        best_err[better] = err[better]
        weight[:, better] = w[:, better]
        chosen[better] = alpha
    return LinearProbe(weight, chosen, z_scale, y_scale)


@dataclass
class MLPProbe:
    """A small MLP from standardised latents to standardised targets."""

    net: torch.nn.Module
    z_scale: _Scaler
    y_scale: _Scaler
    device: torch.device

    def predict(self, z: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        x = torch.as_tensor(self.z_scale(z), dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                outs.append(self.net(x[i : i + batch_size].to(self.device)).cpu())
        standard = torch.cat(outs).numpy().astype(np.float64)
        return standard * self.y_scale.std + self.y_scale.mean

    def r2(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        return r2_score(y, self.predict(z))


def fit_mlp_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    patience: int = 10,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> MLPProbe:
    """Fit a two-hidden-layer MLP probe, stopping early on validation loss."""
    torch.manual_seed(seed)
    device = torch.device(device)
    z_scale, y_scale = _Scaler.fit(z_train), _Scaler.fit(y_train)
    x = torch.as_tensor(z_scale(z_train), dtype=torch.float32)
    y = torch.as_tensor(y_scale(y_train), dtype=torch.float32)
    xv = torch.as_tensor(z_scale(z_val), dtype=torch.float32).to(device)
    yv = torch.as_tensor(y_scale(y_val), dtype=torch.float32).to(device)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, y.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale = float("inf"), copy.deepcopy(net.state_dict()), 0
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            index = order[start : start + batch_size]
            xb, yb = x[index].to(device), y[index].to(device)
            loss = torch.nn.functional.mse_loss(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(net(xv), yv))
        if val < best - 1e-6:
            best, best_state, stale = val, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return MLPProbe(net, z_scale, y_scale, device)


_PART = re.compile(r"/(sin|cos|q[wxyz])$")


def factor_groups(names: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Merge target columns that describe one variable.

    ``hinge/sin`` and ``hinge/cos`` are one angle, ``root/qw`` .. ``root/qz``
    one orientation. Returns the factor names and each column's factor index.
    """
    factors: list[str] = []
    index = np.zeros(len(names), dtype=np.int64)
    for column, name in enumerate(names):
        factor = _PART.sub("", name)
        if factor not in factors:
            factors.append(factor)
        index[column] = factors.index(factor)
    return factors, index


@dataclass(frozen=True)
class DCIScores:
    """Disentanglement, completeness and informativeness (Eastwood & Williams).

    Attributes:
        disentanglement: How much each latent dimension serves one factor,
            weighted by how much it matters overall. 1 is one factor each.
        completeness: How much each factor lives in one latent dimension,
            averaged over factors. 1 is one dimension each.
        informativeness: Mean held-out R² over the target columns.
        importance: ``(D, F)`` latent-to-factor importances.
        factors: The ``F`` factor names.
        per_factor_completeness: ``(F,)``.
    """

    disentanglement: float
    completeness: float
    informativeness: float
    importance: np.ndarray
    factors: tuple[str, ...]
    per_factor_completeness: np.ndarray


def _normalised_entropy(p: np.ndarray, axis: int) -> np.ndarray:
    total = p.sum(axis=axis, keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    n = p.shape[axis]
    if n < 2:
        return np.zeros(np.delete(p.shape, axis))
    logs = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logs).sum(axis=axis) / np.log(n)


def dci(
    probe: LinearProbe,
    names: Sequence[str],
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    group: bool = True,
) -> DCIScores:
    """DCI from a linear probe's standardised weights.

    The original uses gradient-boosted trees for the importances; absolute
    linear weights on standardised data are the common linear variant, and
    they need no extra dependency. With ``group`` the columns of one variable
    (sin and cos of an angle, a quaternion's parts) count as one factor.
    """
    columns = np.abs(probe.weight)  # (D, K)
    if group:
        factors, index = factor_groups(names)
        importance = np.zeros((columns.shape[0], len(factors)))
        np.add.at(importance.T, index, columns.T)
    else:
        factors, importance = list(names), columns
    code_entropy = _normalised_entropy(importance, axis=1)  # (D,)
    weight = importance.sum(axis=1)
    weight = weight / weight.sum() if weight.sum() > 0 else weight
    completeness = 1.0 - _normalised_entropy(importance, axis=0)  # (F,)
    return DCIScores(
        disentanglement=float(((1.0 - code_entropy) * weight).sum()),
        completeness=float(completeness.mean()),
        informativeness=float(probe.r2(z_test, y_test).mean()),
        importance=importance,
        factors=tuple(factors),
        per_factor_completeness=completeness,
    )


@dataclass(frozen=True)
class MIGScores:
    """Mutual information gap (Chen et al., 2018).

    Attributes:
        mig: Mean over factors of the gap between the two latent dimensions
            most informative about the factor, over the factor's entropy.
        per_factor: ``(K,)`` the gap for each target column.
        mutual_information: ``(D, K)`` discretised mutual information, nats.
    """

    mig: float
    per_factor: np.ndarray
    mutual_information: np.ndarray


def _quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    """Equal-frequency bin index per column (ties stay in one bin)."""
    out = np.zeros(x.shape, dtype=np.int64)
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    for j in range(x.shape[1]):
        edges = np.unique(np.quantile(x[:, j], qs))
        out[:, j] = np.searchsorted(edges, x[:, j], side="right")
    return out


def _entropy(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    p = counts[counts > 0] / len(labels)
    return float(-(p * np.log(p)).sum())


def mig(z: np.ndarray, y: np.ndarray, *, bins: int = 20) -> MIGScores:
    """Mutual information gap between latent dimensions and target columns."""
    zb, yb = _quantile_bins(np.asarray(z), bins), _quantile_bins(np.asarray(y), bins)
    D, K = zb.shape[1], yb.shape[1]
    hz = np.array([_entropy(zb[:, d]) for d in range(D)])
    hy = np.array([_entropy(yb[:, k]) for k in range(K)])
    mi = np.zeros((D, K))
    for d in range(D):
        for k in range(K):
            mi[d, k] = hz[d] + hy[k] - _entropy(zb[:, d] * (bins + 1) + yb[:, k])
    mi = np.maximum(mi, 0.0)
    top = -np.sort(-mi, axis=0)
    second = top[1] if D > 1 else np.zeros(K)
    gap = np.divide(top[0] - second, hy, out=np.zeros(K), where=hy > 0)
    return MIGScores(float(gap.mean()), gap, mi)


@dataclass(frozen=True)
class ProbeScores:
    """Every score for one set of latents against one set of targets."""

    names: tuple[str, ...]
    linear_r2: np.ndarray
    mlp_r2: Optional[np.ndarray]
    dci: DCIScores
    mig: Optional[MIGScores]

    @property
    def linear(self) -> float:
        """Mean R² over the scorable targets (NaN columns are skipped)."""
        return float(nanmean(self.linear_r2))

    @property
    def mlp(self) -> Optional[float]:
        return None if self.mlp_r2 is None else float(nanmean(self.mlp_r2))


def score_latents(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    names: Sequence[str],
    *,
    mlp: bool = True,
    with_mig: bool = True,
    max_mig_dims: int = 512,
    mlp_kwargs: Optional[dict] = None,
) -> ProbeScores:
    """Linear and MLP probes, DCI and MIG for one latent, on three splits.

    Probes fit on ``train``, choose penalties and stop early on ``val``, and
    report on ``test``. MIG is skipped for latents wider than ``max_mig_dims``,
    where its pairwise histograms get slow and its top-two gap meaningless.
    Targets that barely vary in ``test`` (see :func:`scorable`) get NaN R² and
    are left out of the means.
    """
    keep = scorable(test[1], np.asarray(train[1], np.float64).std(axis=0))
    linear = fit_linear_probe(*train, *val)
    linear_r2 = np.where(keep, linear.r2(*test), np.nan)
    mlp_r2 = None
    if mlp:
        mlp_r2 = fit_mlp_probe(*train, *val, **(mlp_kwargs or {})).r2(*test)
        mlp_r2 = np.where(keep, mlp_r2, np.nan)
    mig_scores = None
    if with_mig and test[0].shape[1] <= max_mig_dims:
        mig_scores = mig(*test)
    scores = dci(linear, names, *test)
    return ProbeScores(
        names=tuple(names),
        linear_r2=linear_r2,
        mlp_r2=mlp_r2,
        dci=replace(scores, informativeness=float(nanmean(linear_r2))),
        mig=mig_scores,
    )
