"""
Bernoulli distribution testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from pytest import approx, raises

from cprior.models import BernoulliABTest
from cprior.models import BernoulliModel
from cprior.models import BernoulliMVTest
from cprior.models import GeometricModel


def test_bernoulli_model_update():
    model = BernoulliModel(alpha=1, beta=1)
    data = np.array([0, 0, 0, 1, 1])

    model.update(data)

    assert model.n_samples_ == 5
    assert model.n_success_ == 2


def test_bernoulli_model_pppdf_x():
    model = BernoulliModel(alpha=4, beta=6)

    assert model.pppdf([0, 1, 2]) == approx([0.6, 0.4, 0])


def test_bernoulli_model_stats():
    model = BernoulliModel(alpha=4, beta=6)

    assert model.pppdf(0) == approx(0.6)
    assert model.pppdf(1) == approx(0.4)
    assert model.ppmean() == approx(0.4)
    assert model.ppvar() == approx(0.24)


def test_bernoulli_ab_check_models():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = GeometricModel(alpha=1, beta=1)

    with raises(TypeError):
        BernoulliABTest(modelA=modelA, modelB=modelB)


def test_bernoulli_mv_check_model_input():
    modelA = BernoulliModel(alpha=1, beta=1)
    modelB = BernoulliModel(alpha=1, beta=1)

    with raises(TypeError):
        BernoulliMVTest(models=[modelA, modelB])


def test_bernoulli_mv_check_control():
    models = {
        "B": BernoulliModel(name="variant 1", alpha=1, beta=1),
        "C": BernoulliModel(name="variant 2", alpha=1, beta=1)
    }

    with raises(ValueError):
        BernoulliMVTest(models=models)


def test_bernoulli_mv_check_update():
    models = {
        "A": BernoulliModel(name="control", alpha=1, beta=1),
        "B": BernoulliModel(name="variant 2", alpha=1, beta=1)
    }

    mvtest = BernoulliMVTest(models=models)

    with raises(ValueError):
        data = np.array([0, 0, 0, 1, 1])
        mvtest.update(data=data, variant="C")


def _bernoulli_mvtest(**arms):
    models = {}
    for name, (n, successes) in arms.items():
        models[name] = BernoulliModel(name=name)
        models[name].update(np.r_[np.ones(successes), np.zeros(n - successes)])
    return BernoulliMVTest(models)


def test_bernoulli_mv_expected_lift_relative_is_ratio_of_posterior_means():
    # 8/220 vs 10/252: posterior means 4.05% vs 4.33%. The old formula
    # (a0+b0)(a1-1)/(a0(a1+b1-1)) - 1 dropped one success from the variant
    # and returned -2.5%.
    mvtest = _bernoulli_mvtest(A=(220, 8), B=(252, 10))
    lift = mvtest.expected_lift_relative(control="A", variant="B")

    assert lift == approx(
        mvtest.models["B"].mean() / mvtest.models["A"].mean() - 1)
    assert lift == approx(0.0682, abs=1e-4)
    assert mvtest.probability(control="A", variant="B") > 0.5


def test_bernoulli_mv_expected_lift_relative_vs_all_two_arms():
    # With a single other arm E[max(others)] is that arm's mean, so vs. all
    # equals vs. control.
    mvtest = _bernoulli_mvtest(A=(220, 8), B=(252, 10))

    assert mvtest.expected_lift_relative_vs_all(variant="B") == approx(
        mvtest.expected_lift_relative(control="A", variant="B"), abs=1e-6)
    assert mvtest.expected_lift_relative_vs_all(variant="A") == approx(
        mvtest.expected_lift_relative(control="B", variant="A"), abs=1e-6)


def test_bernoulli_mv_expected_lift_relative_vs_all_methods_agree():
    mvtest = _bernoulli_mvtest(A=(220, 8), B=(252, 10), C=(240, 9))
    quad = mvtest.expected_lift_relative_vs_all(method="quad", variant="B")
    mlhs = mvtest.expected_lift_relative_vs_all(method="MLHS", variant="B",
                                                mlhs_samples=10000)

    assert quad == approx(mlhs, abs=1e-3)


def test_bernoulli_mv_expected_lift_relative_vs_all_large_samples():
    # Narrow posteriors: without integration points quad missed the peak of
    # the single other arm, E[max] came out ~0 and the lift ~8.6e8.
    mvtest = _bernoulli_mvtest(A=(50000, 5000), B=(50000, 5200))

    for variant, control in (("A", "B"), ("B", "A")):
        assert mvtest.expected_lift_relative_vs_all(variant=variant) == approx(
            mvtest.expected_lift_relative(control=control, variant=variant),
            abs=1e-6)
