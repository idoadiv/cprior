"""
Normal adjusted multivariate testing.
"""

import numpy as np

from pytest import approx

from cprior.models import NormalAdjustedModel
from cprior.models import NormalAdjustedMVTest
from cprior.models import PoissonAdjustedModel
from cprior.models import PoissonMVTest


def _model(data):
    model = NormalAdjustedModel(loc=0, scale=1e6, shape=1e6)
    model.update(data)
    return model


def _mvtest(**arms):
    return NormalAdjustedMVTest({k: _model(v) for k, v in arms.items()},
                                simulations=200000, random_state=42)


def _data():
    rng = np.random.RandomState(0)
    # control mean 2.0, variant mean 3.0 -> relative lift of B over A is +50%
    return rng.normal(2.0, 1.0, 20000), rng.normal(3.0, 1.0, 20000)


def test_normal_adjusted_expected_lift_relative_is_positive_when_variant_wins():
    A, B = _data()
    mvtest = _mvtest(A=A, B=B)

    assert mvtest.expected_lift_relative(control="A", variant="B") == approx(
        0.5, rel=1e-2)


def test_normal_adjusted_expected_lift_relative_mc_matches_exact():
    A, B = _data()
    mvtest = _mvtest(A=A, B=B)

    exact = mvtest.expected_lift_relative(control="A", variant="B")
    mc = mvtest.expected_lift_relative(method="MC", control="A", variant="B")

    assert mc == approx(exact, rel=1e-2)


def test_normal_adjusted_expected_lift_relative_orientation_follows_arguments():
    """Direction must come from control/variant, not from the variant names."""
    A, B = _data()
    mvtest = _mvtest(A=A, B=B)

    # 2.0 relative to 3.0 is -1/3, and is not the negation of the +50% above.
    assert mvtest.expected_lift_relative(control="B", variant="A") == approx(
        -1 / 3, rel=1e-2)


def test_normal_adjusted_expected_lift_relative_matches_poisson():
    """Same orientation as the gamma-Poisson model on equivalent (shifted) data.

    Both models serve continuous KPIs -- the normal one only when the data
    contains negatives -- so they must not disagree about which way is up.
    """
    A, B = _data()
    shift = -A.min() + 1

    normal = _mvtest(A=A, B=B).expected_lift_relative(control="A", variant="B")

    poisson_models = {}
    for name, data in (("A", A), ("B", B)):
        model = PoissonAdjustedModel(shape=1e-6, rate=1e-6)
        model.update(data + shift)
        poisson_models[name] = model
    poisson = PoissonMVTest(poisson_models).expected_lift_relative(
        control="A", variant="B")

    shifted_lift = (B + shift).mean() / (A + shift).mean() - 1

    assert poisson == approx(shifted_lift, rel=1e-2)
    assert np.sign(normal) == np.sign(poisson)


def test_normal_adjusted_expected_lift_relative_vs_all_methods_agree():
    rng = np.random.RandomState(0)
    A = rng.normal(2.0, 1.0, 20000)
    B = rng.normal(3.0, 1.0, 20000)
    C = rng.normal(2.5, 1.0, 20000)
    mvtest = _mvtest(A=A, B=B, C=C)

    quad = mvtest.expected_lift_relative_vs_all(method="quad", variant="B")
    mlhs = mvtest.expected_lift_relative_vs_all(method="MLHS", variant="B")

    # B (3.0) against max(A, C) = 2.5 -> +20%, normalised by the comparison
    # group, not by B.
    assert quad == approx(0.2, rel=5e-2)
    assert mlhs == approx(quad, rel=1e-2)


def test_normal_adjusted_expected_lift_relative_vs_all_is_negative_when_losing():
    rng = np.random.RandomState(0)
    A = rng.normal(2.0, 1.0, 20000)
    B = rng.normal(3.0, 1.0, 20000)
    C = rng.normal(2.5, 1.0, 20000)
    mvtest = _mvtest(A=A, B=B, C=C)

    # A (2.0) against max(B, C) = 3.0 -> -1/3
    assert mvtest.expected_lift_relative_vs_all(
        method="quad", variant="A") == approx(-1 / 3, rel=5e-2)
