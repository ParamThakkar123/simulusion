"""Probes and disentanglement scores on synthetic latents with known answers."""

import numpy as np
import pytest

pytest.importorskip("torch")

from simulusion.evals.probing import (  # noqa: E402
    dci,
    factor_groups,
    fit_linear_probe,
    fit_mlp_probe,
    mig,
    r2_score,
    score_latents,
)

K = 4


def _factors(n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, (n, K))


def _splits(make_z, seed=0, n=(2000, 500, 500)):
    """Train / val / test pairs of (latents, factors) from one latent map."""
    out = []
    for i, m in enumerate(n):
        y = _factors(m, seed + 10 * i)
        out.append((make_z(y, np.random.default_rng(seed + i)), y))
    return out


def _permuted(y, rng):
    # Each latent dimension is one factor, rescaled, plus a little noise.
    z = y[:, [2, 0, 3, 1]] * np.array([3.0, -0.5, 10.0, 1.0])
    return z + 0.01 * rng.normal(size=z.shape)


_ROTATION = np.linalg.qr(np.random.default_rng(99).normal(size=(K, K)))[0]


def _rotated(y, rng):
    return y @ _ROTATION + 0.01 * rng.normal(size=y.shape)


def test_r2_score_matches_the_definition_and_handles_constant_columns():
    y = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    pred = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(r2_score(y, pred), [1.0 - 1.0 / 2.0, 1.0])
    assert r2_score(y, pred + 1.0)[1] == 0.0


def test_axis_aligned_latents_are_informative_disentangled_and_complete():
    train, val, test = _splits(_permuted)

    probe = fit_linear_probe(*train, *val)
    names = [f"f{k}" for k in range(K)]
    scores = dci(probe, names, *test, group=False)

    assert probe.r2(*test).min() > 0.99
    assert scores.disentanglement > 0.9 and scores.completeness > 0.9
    assert scores.informativeness > 0.99
    # Factor 0 lives in latent dimension 1 (the permutation above).
    assert int(np.argmax(scores.importance[:, 0])) == 1
    assert mig(*test).mig > 0.5


def test_a_rotation_is_just_as_informative_but_entangled():
    aligned, rotated = _splits(_permuted), _splits(_rotated)

    names = [f"f{k}" for k in range(K)]
    a = dci(fit_linear_probe(*aligned[0], *aligned[1]), names, *aligned[2], group=False)
    r = dci(fit_linear_probe(*rotated[0], *rotated[1]), names, *rotated[2], group=False)

    assert r.informativeness > 0.99
    assert r.disentanglement < a.disentanglement - 0.3
    assert r.completeness < a.completeness - 0.3
    assert mig(*rotated[2]).mig < mig(*aligned[2]).mig


def test_noise_latents_score_nothing_and_get_a_strong_penalty():
    train, val, test = _splits(lambda y, rng: rng.normal(size=(len(y), 8)))

    probe = fit_linear_probe(*train, *val)

    assert np.all(probe.r2(*test) < 0.02)
    assert probe.alpha.min() >= 10.0
    assert mig(*test).mig < 0.05


def test_ridge_works_with_more_dimensions_than_frames():
    train, val, test = _splits(
        lambda y, rng: np.concatenate([y, rng.normal(size=(len(y), 300))], axis=1),
        n=(200, 200, 200),
    )
    probe = fit_linear_probe(*train, *val)
    assert probe.weight.shape == (K + 300, K)
    # 300 noise dimensions on 200 frames cost some R², but the probe still
    # puts each factor's weight on its own signal dimension.
    assert probe.r2(*test).mean() > 0.5
    weights = np.abs(probe.weight)
    np.testing.assert_array_equal(weights.argmax(axis=0), np.arange(K))


def test_mlp_probe_recovers_what_a_linear_probe_cannot():
    # Each factor is an angle stored as (cos, sin): exactly recoverable, but
    # only through atan2, so a linear readout misses much of it.
    train, val, test = _splits(
        lambda y, rng: np.concatenate([np.cos(np.pi * y), np.sin(np.pi * y)], 1)
    )

    linear = fit_linear_probe(*train, *val).r2(*test).mean()
    nonlinear = fit_mlp_probe(*train, *val, epochs=150, patience=20).r2(*test).mean()

    assert nonlinear > 0.85  # the wrap at y = +-1 costs a little
    assert nonlinear > linear + 0.2


def test_factor_groups_merge_angles_and_quaternions():
    names = [
        "qpos/slider",
        "qpos/hinge/sin",
        "qpos/hinge/cos",
        "qpos/root/qw",
        "qpos/root/qx",
        "qpos/root/qy",
        "qpos/root/qz",
        "qvel/hinge",
    ]
    factors, index = factor_groups(names)
    assert factors == ["qpos/slider", "qpos/hinge", "qpos/root", "qvel/hinge"]
    np.testing.assert_array_equal(index, [0, 1, 1, 2, 2, 2, 2, 3])


def test_dci_groups_sin_and_cos_of_one_angle_into_one_factor():
    rng = np.random.default_rng(0)

    def make(n):
        angle = rng.uniform(-np.pi, np.pi, n)
        y = np.stack([np.sin(angle), np.cos(angle)], axis=1)
        return y.copy(), y

    train, val, test = make(1000), make(300), make(300)
    probe = fit_linear_probe(*train, *val)
    grouped = dci(probe, ["a/sin", "a/cos"], *test)
    split = dci(probe, ["a/sin", "a/cos"], *test, group=False)

    assert grouped.factors == ("a",) and grouped.importance.shape == (2, 1)
    assert grouped.completeness < split.completeness  # two dims share one factor
    assert split.disentanglement > 0.9


def test_score_latents_bundles_every_score():
    train, val, test = _splits(_permuted)
    names = ("a", "b", "c", "d")

    scores = score_latents(train, val, test, names, mlp_kwargs={"epochs": 5})

    assert scores.names == names
    assert scores.linear > 0.99 and scores.mlp is not None
    assert scores.mig is not None and scores.dci.factors == names
    wide = score_latents(train, val, test, names, mlp=False, max_mig_dims=2)
    assert wide.mig is None and wide.mlp is None
