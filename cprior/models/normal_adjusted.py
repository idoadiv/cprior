from cprior.cdist.normal_inverse_gamma import func_mv_elr_mean
from cprior.cdist.utils import check_mv_method
from cprior.models import NormalModel, NormalMVTest

import numpy as np

from multiprocessing import Pool
from scipy import integrate
from scipy import special
from scipy import stats


class NormalAdjustedModel(NormalModel):
    def __init__(self, name="", loc=0.001, variance_scale=0.001, shape=0.001,
                 scale=0.001):
        super().__init__(name=name, loc=loc, variance_scale=variance_scale, shape=shape,
                         scale=scale)

    def mean(self) -> float:
        """
        Mean of the Normal probability (x only).

        Returns
        -------
        x_mean : float
            Mean of the random variate x.
        """
        return super().mean()[0] # option 1

        # x_mean = self.loc option 2
        # return x_mean
    def ppf(self, q: float) -> float:
        """
        Compute the percent-point function (inverse of the CDF) for a given quantile.

        Parameters
        ----------
        q : float
            The quantile for which to compute the percent-point function.

        Returns
        -------
        float
            The value corresponding to the given quantile.
        """
        return super().ppf(q=q)[0]


class NormalAdjustedTest(NormalMVTest):
    def __init__(self, models, simulations=1000000, random_state=None,
                 n_jobs=None):
        super().__init__(models, simulations, random_state, n_jobs)

    def probability(self, method: str = "exact", control: str = "A", variant: str = "B", lift: float = 0) -> float:
        """
        Compute the probability of a variant being better than the control variant
        by a given lift.

        Parameters
        ----------
        method : str
            The method of computation. Options are "exact" and "MC".
        control : str
            The name of the control variant.
        variant : str
            The name of the tested variant.
        lift : float
            The amount of uplift to consider.

        Returns
        -------
        float
            The probability of the variant being better than the control variant
            by the given lift.
        """
        return super().probability(method=method, control=control, variant=variant, lift=lift)[0]

    def expected_lift_relative(self, method="exact", control="A", variant="B"):
        r"""
        Compute expected relative lift for choosing a variant.

        * If ``variant == "A"``, :math:`\mathrm{E}[(B - A) / A]`
        * If ``variant == "B"``, :math:`\mathrm{E}[(A - B) / B]`

        Parameters
        ----------
        method : str (default="exact")
            The method of computation. Options are "exact" and "MC".

        control : str (default="A")
            The control variant.

        variant : str (default="B")
            The tested variant.

        Returns
        -------
        expected_lift_relative : float
        """
        check_mv_method(method=method, method_options=("exact", "MC"),
                        control=control, variant=variant,
                        variants=self.models.keys())

        if method == "exact":
            model_control = self.models[control]
            model_variant = self.models[variant]

            mu_control = model_control.loc_posterior
            mu_variant = model_variant.loc_posterior

            if variant == "A":
                return (mu_variant - mu_control) / mu_control
            else:
                return (mu_control - mu_variant) / mu_variant
        else:  # Monte Carlo method
            data_control = self.models[control].rvs(self.simulations, self.random_state)
            data_variant = self.models[variant].rvs(self.simulations, self.random_state)

            x_control = data_control[:, 0]
            x_variant = data_variant[:, 0]

            if variant == "A":
                return ((x_variant - x_control) / x_control).mean()
            else:
                return ((x_control - x_variant) / x_variant).mean()

    def expected_lift_relative_vs_all(self, method="quad", control="A",
                                      variant="B", mlhs_samples=1000):
        r"""
        Compute the expected relative lift against all variations.
        For example, given variants "A", "B", "C" and "D", and choosing variant="B",
        we compute :math:`\mathrm{E}[(B - \max(A, C, D)) / \max(A, C, D)]`.

        Parameters
        ----------
        method : str (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling) and "quad"
            (numerical integration).

        variant : str (default="B")
            The chosen variant.

        mlhs_samples : int (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        expected_lift_relative_vs_all : float
        """
        check_mv_method(method=method, method_options=("MC", "MLHS", "quad"),
                        control=None, variant=variant,
                        variants=self.models.keys())

        # exclude variant
        variants = list(self.models.keys())
        variants.remove(variant)

        if method == "MC":
            # generate samples from all models in parallel
            xvariant = self.models[variant].rvs(self.simulations,
                                                self.random_state)

            pool = Pool(processes=self.n_jobs)
            processes = [pool.apply_async(self._rvs, args=(v,))
                         for v in variants]
            xall = [p.get() for p in processes]
            maxall = np.maximum.reduce(xall)

            return ((xvariant - maxall) / maxall).mean(axis=0)

        else:
            # prepare parameters
            variant_params = [(self.models[v].loc_posterior,
                               self.models[v].variance_scale_posterior,
                               self.models[v].shape_posterior,
                               self.models[v].scale_posterior)
                              for v in variants]

            mu = self.models[variant].loc_posterior
            la = self.models[variant].variance_scale_posterior
            a = self.models[variant].shape_posterior
            b = self.models[variant].scale_posterior

            if method == "quad":
                t_ppfs = [stats.t(
                    df=2 * self.models[v].shape_posterior,
                    loc=self.models[v].loc_posterior,
                    scale=np.sqrt(
                        self.models[v].scale_posterior /
                        self.models[v].shape_posterior /
                        self.models[v].variance_scale_posterior)).ppf(
                    [0.00000001, 0.99999999]) for v in variants]

                min_t = np.min([q[0] for q in t_ppfs])
                max_t = np.max([q[1] for q in t_ppfs])

                # mean
                e_max = integrate.quad(func=func_mv_elr_mean, a=min_t,
                                       b=max_t, args=(variant_params))[0]

                e_inv_x = (1 + self.models[variant].var()[0] / mu ** 2) / mu

                elr_mean = 1 - (e_max * e_inv_x)

                return elr_mean

            else:  # MLHS
                r = np.arange(1, mlhs_samples + 1)
                np.random.shuffle(r)
                r = (r - 0.5) / mlhs_samples
                r = r[..., np.newaxis]

                n = len(variant_params)
                variant_params.append((mu, la, a, b))
                uu, ll, aa, bb = map(np.array, zip(*variant_params))
                vv = 2 * aa
                ss = np.sqrt(bb / aa / ll)

                # mean
                xx = stats.t(df=vv, loc=uu, scale=ss).ppf(r)
                xr = (1. / xx[:, -1]).mean()

                elr_mean = 1 - (np.sum(
                    xx[:, :-1].T * [np.prod([
                        special.stdtr(vv[j], (xx[:, i] - uu[j]) / ss[j])
                        for j in range(n) if j != i],
                        axis=0) for i in range(n)], axis=0).mean() * xr)

                return elr_mean

    def probability_vs_all(
            self, method: str = "quad", variant: str = "B", lift: float = 0, mlhs_samples: int = 1000
    ) -> float:
        """
        Compute the probability of a variant outperforming all other variants by a given lift.

        Parameters
        ----------
        method : str, optional (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling), and "quad" (numerical integration).
        variant : str, optional (default="B")
            The chosen variant.
        lift : float, optional (default=0.0)
            The amount of uplift.
        mlhs_samples : int, optional (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        float
            The probability of the variant outperforming all other variants by the given lift.
        """
        return super().probability_vs_all(method=method, variant=variant, lift=lift, mlhs_samples=mlhs_samples)[0]

    def expected_loss_vs_all(
            self, method: str = "quad", variant: str = "B", lift: float = 0, mlhs_samples: int = 1000
    ) -> float:
        """
        Compute the expected loss of a variant against all other variants.

        Parameters
        ----------
        method : str, optional (default="quad")
            The method of computation. Options are "MC" (Monte Carlo),
            "MLHS" (Monte Carlo + Median Latin Hypercube Sampling), and "quad" (numerical integration).
        variant : str, optional (default="B")
            The chosen variant.
        lift : float, optional (default=0.0)
            The amount of uplift.
        mlhs_samples : int, optional (default=1000)
            Number of samples for MLHS method.

        Returns
        -------
        float
            The expected loss of the variant against all other variants.
        """
        return super().expected_loss_vs_all(method=method, variant=variant, lift=lift, mlhs_samples=mlhs_samples)[0]

    def expected_loss(
            self, method: str = "exact", control: str = "A", variant: str = "B", lift: float = 0
    ) -> float:
        """
        Compute the expected loss of a variant compared to the control variant.

        Parameters
        ----------
        method : str, optional (default="exact")
            The method of computation. Options are "exact" and "MC".
        control : str, optional (default="A")
            The name of the control variant.
        variant : str, optional (default="B")
            The name of the tested variant.
        lift : float, optional (default=0.0)
            The amount of uplift to consider.

        Returns
        -------
        float
            The expected loss of the variant compared to the control variant.
        """
        return super().expected_loss(method=method, control=control, variant=variant, lift=lift)[0]