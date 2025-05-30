import numpy as np

def silverman_bandwidth(data):
    data = np.atleast_2d(data)
    if data.shape[1] == 1:
        # 1D case: np.cov returns scalar, so we avoid trace
        std = np.std(data, ddof=1)
        n = data.shape[0]
        bandwidth = 1.06 * std * n ** (-1 / 5)  # Classic 1D Silverman
        return bandwidth
    else:
        n, d = data.shape
        cov = np.cov(data.T)
        tr_cov = np.trace(cov)
        bandwidth_squared = (1 / d) * tr_cov * (4 / ((2 * d + 1) * n)) ** (2 / (d + 4))
        return np.sqrt(bandwidth_squared)


def mean_shift_step(data, mode, bandwidth):
    kernel = np.exp(-np.linalg.norm(data - mode, axis=1)**2 / (2 * bandwidth**2))
    return np.sum(kernel[:, np.newaxis] * data, axis=0) / kernel.sum()

def dp_mean_shift_step(data, mode, bandwidth, epsilon, delta, T, C=1.5):
    n, d = data.shape
    K_C = np.exp(-0.5 * C**2)
    S = (4 * np.sqrt(d)) / (n * (bandwidth**d) * (K_C / (2 * np.sqrt(d))**d))
    sigma_squared = 2 * (S**2) * T * np.log(1.25 / delta) / epsilon**2
    noisy_update = np.random.normal(0, np.sqrt(sigma_squared), size=mode.shape)
    return mean_shift_step(data, mode, bandwidth) + noisy_update

def mean_shift(data, epsilon, delta, initial_modes=None, T=100, C=1.1):
    if initial_modes is None:
        initial_modes = data  # Default to the entire dataset
    bandwidth = silverman_bandwidth(data)
    modes = initial_modes
    for _ in range(T):
        new_modes = []
        for mode in modes:
            new_mode = dp_mean_shift_step(data, mode, bandwidth, epsilon, delta, T, C)
            new_modes.append(new_mode)
        modes = np.array(new_modes)
    return modes

def merge_modes(modes, bandwidth, k=2):
    threshold = k * bandwidth
    merged_modes = []
    counts = []

    for mode in modes:
        merged = False
        for i, m in enumerate(merged_modes):
            if np.linalg.norm(mode - m) < threshold:
                new_count = counts[i] + 1
                merged_modes[i] = (counts[i] * m + mode) / new_count
                counts[i] = new_count
                merged = True
                break
        if not merged:
            merged_modes.append(mode)
            counts.append(1)

    return np.array(merged_modes)
