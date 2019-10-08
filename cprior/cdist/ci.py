"""
Confidence/credible intervals (CI) methods.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numbers

import numpy as np


def ci_interval(x, interval_length=0.9, method="ETI"):
    r"""
    Compute confidence/credible intervals (CI).

    Parameters
    ----------
    x : array-like, shape = (n_samples)

    interval_length : float (default=0.9)
        Compute ``interval_length``\% credible interval. This is a value in
        [0, 1].

    method : str (default="ETI")
        Method to compute credible intervals. Supported methods are Highest
        Density interval (``method="HDI``) and Equal-tailed interval
        (``method="ETI"``).
    """
    if method not in ("ETI", "HDI"):
        raise ValueError("method {} is not supported. Use 'ETI' or 'HDI"
                         .format(method))

    invalid_length = (interval_length < 0 or interval_length > 1)

    if not isinstance(interval_length, numbers.Number) or invalid_length:
        raise ValueError("Interval length must a value in [0, 1]; got "
                         "interval_length={}.".format(interval_length))

    if method == "ETI":
        lower = 100 * (1 - interval_length) / 2
        upper = 100 * (1 + interval_length) / 2

        return np.percentile(x, [lower, upper])
    else:
        n = len(x)
        xsorted = np.sort(x)
        n_included = int(np.ceil(interval_length * n))
        n_ci = n - n_included
        ci = xsorted[n_included:] - xsorted[:n_ci]
        j = np.argmin(ci)
        hdi_min = xsorted[j]
        hdi_max = xsorted[j + n_included]

        return hdi_min, hdi_max
