#dpms.py
import numpy as np
from bandwidth import silverman_bandwidth


def mean_shift_step_dp(data, mode, bandwidth, sigma_squared):
    kernel = np.exp(-np.linalg.norm(data - mode, axis=1) ** 2 / (2 * bandwidth ** 2))
    weighted_sum = np.sum(kernel[:, np.newaxis] * data, axis=0)
    kernel_sum = kernel.sum()
    updated_mode = weighted_sum / kernel_sum if kernel_sum != 0 else mode
    noise = np.random.normal(loc=0, scale=np.sqrt(sigma_squared), size=updated_mode.shape)
    return updated_mode + noise


def mean_shift_dp(data, epsilon, delta, initial_modes=None, T=20, bandwidth=None, p=0.01, C=1.1):
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
        print(f"[DEBUG] Bandwidth estimated by Silverman: {bandwidth}")

    if initial_modes is None:
        n_samples = max(1, int(len(data) * p))
        indices = np.random.choice(len(data), size=n_samples, replace=False)
        initial_modes = data[indices]
        print(f"[DEBUG] Initial modes randomly selected: {n_samples} points (p={p})")
    else:
        print(f"[DEBUG] Using provided initial modes: {len(initial_modes)}")

    n, d = data.shape
    R = np.max(np.linalg.norm(data, axis=1))
    K_C = np.exp(-0.5 * C**2)
    S = (4 * R) / (n * (bandwidth ** d) * (K_C / (2 * R) ** d))
    sigma_squared = 2 * (S ** 2) * T * np.log(1.25 / delta) / (epsilon ** 2)
    print(f"[DEBUG] DP noise variance sigma_squared: {sigma_squared}")

    modes = initial_modes
    for t in range(T):
        modes = np.array([
            mean_shift_step_dp(data, mode, bandwidth, sigma_squared) for mode in modes
        ])
        print(f"[DEBUG] Iteration {t + 1}/{T} complete.")

    return modes
