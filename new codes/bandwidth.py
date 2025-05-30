import numpy as np

def silverman_bandwidth(data):
    """
    Silverman's rule of thumb bandwidth estimator (robust for univariate).
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
    Robust Silverman's bandwidth for univariate data.
    """
    std = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    n = len(data)
    sigma = min(std, iqr / 1.34)
    h = 2 * sigma * n ** (-1 / 5)
    return h

def _silverman_multivariate(data):
    """
    Multivariate Silverman bandwidth based on trace of covariance.
    """
    n, d = data.shape
    cov = np.cov(data.T)
    tr_cov = np.trace(cov)
    h_squared = (2 / d) * tr_cov * (4 / ((2 * d + 1) * n)) ** (2 / (d + 4))
    return np.sqrt(h_squared)

def robust_fixed_bandwidth(data, c=0.2):
    """
    Robust bandwidth independent of sample size n.
    Uses MAD (Median Absolute Deviation) as scale estimator.

    Works for univariate or multivariate data.

    Args:
        data (np.ndarray): shape (n_samples,) or (n_samples, n_features)
        c (float): Smoothing multiplier

    Returns:
        float: Bandwidth scalar
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    mad = np.median(np.abs(data - np.median(data, axis=0)), axis=0) / 0.6745
    scale = np.mean(mad)
    return c * scale
