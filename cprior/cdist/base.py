"""
Base Bayes model and A/B testing classes.
"""

# guillermo navas-palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

from abc import ABCMeta
from abc import abstractmethod


class BayesModel(object):
    """
    Bayes model class.
    """    
    def update(self):
        """Update posterior parameters."""
        raise NotImplementedError

    def mean(self):
        """Mean of the posterior distribution."""
        self._dist.mean()

    def var(self):
        """Variance of the posterior distribution."""
        self._dist.var()

    def std(self):
        """Standard deviation of the posterior distribution."""
        self._dist.std()

    def pdf(self, x):
        """
        Probability density function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.

        Returns
        -------
        pdf : numpy.ndarray
           Probability density function evaluated at x.
        """
        self._dist.pdf(x)

    def cdf(self, x):
        """
        Cumulative distribution function of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Quantiles.
        
        Returns
        -------
        cdf : numpy.ndarray
            Cumulative distribution function evaluated at x.
        """
        self._dist.cdf(x)

    def ppf(self, q):
        """
        Percent point function (quantile) of the posterior distribution.

        Parameters
        ----------
        x : array-like
            Lower tail probability.

        Returns
        -------
        ppf : numpy.ndarray
            Quantile corresponding to the lower tail probability q.
        """
        self._dist.ppf(x)

    def rvs(self, size=1, random_state=None):
        """
        Random variates of the posterior distribution.

        Parameters
        ----------
        size : int (default=1)
            Number of random variates.

        random_state : int or None (default=None)
            The seed used by the random number generator.

        Returns
        -------
        rvs : numpy.ndarray or scalar
            Random variates of given size.
        """
        self._dist.rvs(size=size, random_state=random_state)

    def credible_interval(self):
        """
        Credible interval of the posterior distribution.

        Parameters
        ----------
        interval_length : float (default=0.9)
            Compute ``interval_length``\% credible interval. This is a value in
            [0, 1].

        Returns
        -------
        interval : tuple
            Lower and upper credible interval limits.
        """
        if interval_length < 0 or interval_length > 1:
            raise ValueError("interval_length must be a value in [0, 1].")

        lower = (1 - interval_length) / 2
        upper = (1 + interval_length) / 2

        return (self.ppf(lower), self.ppf(upper))


class BayesABTest(metaclass=ABCMeta):
    """
    Bayes A/B test abstract class.

    Parameters
    ----------
    modelA : object
        The Bayes model for variant A.

    modelB : object
        The Bayes model for variant B.

    simulations : int or None (default=1000000)
        Number of Monte Carlo simulations.
    random_state : int or None (default=None)
        The seed used by the random number generator.
    """
    def __init__(self, modelA, modelB, simulations=None, random_state=None):
        self.modelA = modelA
        self.modelB = modelB
        self.simulations = simulations
        self.random_state = random_state

    @abstractmethod
    def probability(self):
        pass

    @abstractmethod
    def expected_loss(self):
        pass

    @abstractmethod
    def expected_loss_relative(self):
        pass

    @abstractmethod
    def expected_loss_ci(self):
        pass

    @abstractmethod
    def expected_loss_relative_ci(self):
        pass

    def update_A(self, data):
        self.modelA.update(data)

    def update_B(self, data):
        self.modelB.update(data)
