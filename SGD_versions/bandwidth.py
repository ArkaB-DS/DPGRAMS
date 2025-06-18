# bandwidth.py
import numpy as np

def silverman_bandwidth(data):
    """
    Silverman's rule of thumb bandwidth estimator.
    Handles both univariate and multivariate data.
    
    Returns:
        float: estimated bandwidth
    """
    data = np.atleast_2d(data)
    if data.shape[1] == 1:
        return _silverman_univariate(data[:, 0])
    else:
        return _silverman_multivariate(data)

def _silverman_univariate(data):
    """
    Silverman's rule for univariate data.
    """
    std = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    n = len(data)
    sigma = min(std, iqr / 1.34)
    h = 2 * sigma * n ** (-1 / 5)
    return h

def _silverman_multivariate(data):
    """
    Silverman's rule for multivariate data.
    """
    n, d = data.shape
    cov = np.cov(data.T)
    tr_cov = np.trace(cov)
    h_squared = (2 / d) * tr_cov * (4 / ((2 * d + 1) * n)) ** (2 / (d + 4))
    return np.sqrt(h_squared)
