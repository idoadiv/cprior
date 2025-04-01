"""
Bayesian model with Poisson likelihood.
"""

import numpy as np

from .poisson import PoissonModel


class PoissonAdjustedModel(PoissonModel):
    def update(self, data):
        """
        Update posterior parameters with new data.

        Parameters
        ----------
        data : array-like, shape = (n_samples)
            Data samples from a Poisson distribution.
        """
        x = np.asarray(data)
        n = x.size
        var = np.var(x)
        mean = np.mean(x)
        
        if n == 0:
            return

        # When VAR==0, ie: when all the data is the same, we get div/0, which is. This prevents it
        if var > 0:
            self._shape_posterior +=(mean**2/var) * n
            self._rate_posterior += (mean/var) * n
        self.n_samples_ += n
