import numpy as np
from bandwidth import silverman_bandwidth
import math


def mean_shift_dp_sgd(
    data,
    epsilon,
    delta,
    initial_modes=None,
    T=20,
    bandwidth=None,
    p=0.01,
    kernel_C=1.1,
    batch_size=10,
    eta=None,
    use_correlated_noise=True
):
    """
    Differentially Private Mean-Shift via Mini-batch SGD using:
    - Global sensitivity S (as derived in the DPMS paper),
    - Correlated Gaussian noise across iterations (optional),
    - Mini-batch SGD updates.
    
    Args:
        data (np.ndarray): shape (n_samples, n_features)
        epsilon (float): total privacy budget ε
        delta (float): total δ
        initial_modes (np.ndarray): shape (k, d), if None select p fraction of data
        T (int): number of DP-SGD iterations
        bandwidth (float): kernel bandwidth; estimated if None
        p (float): fraction of data to initialize modes
        kernel_C (float): C in Gaussian kernel: K_C = exp(-C^2 / 2)
        batch_size (int): mini-batch size
        eta (float or None): learning rate; if None, defaults to (b/n)/sqrt(T)
        use_correlated_noise (bool): whether to use correlated Gaussian noise across T steps

    Returns:
        np.ndarray: final modes, shape (k, d)
    """
    n, d = data.shape

    # Step 1: Estimate bandwidth if needed
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
        print(f"[DEBUG] Bandwidth estimated by Silverman: {bandwidth}")

    # Step 2: Initialize mode seeds
    if initial_modes is None:
        n_modes = max(1, int(n * p))
        indices = np.random.choice(n, size=n_modes, replace=False)
        modes = data[indices].copy()
        print(f"[DEBUG] Initial modes randomly selected: {n_modes} points (p={p})")
    else:
        modes = initial_modes.copy()
        print(f"[DEBUG] Using provided initial modes: {modes.shape[0]}")

    # Step 3: Compute global sensitivity S from your DPMS theory
    R = np.max(np.linalg.norm(data, axis=1))        # max L2 norm of any data point
    K_C = np.exp(-0.5 * kernel_C ** 2)
    S = (4 * R) / (n * (bandwidth ** d) * (K_C / (2 * R) ** d))  # your derived expression
    print(f"[DEBUG] Computed global sensitivity S: {S:.4e}")

    # Step 4: Set default learning rate if not provided
    if eta is None:
        eta = (batch_size / n) / math.sqrt(T)
        print(f"[DEBUG] Eta not provided; using heuristic eta = {eta:.4f}")

    # Step 5: Compute Gaussian noise variance
    # Each update has sensitivity S; use total vector query sensitivity sqrt(T)*S
    # For correlated noise, noise added is Z ~ N(0, Σ) across T steps in R^d
    sigma2_base = 2 * (S ** 2) * np.log(1.25 / delta) / (epsilon ** 2)
    print(f"[DEBUG] DP base noise variance per update direction: {sigma2_base:.4e}")

    # Construct correlated Gaussian noise (T x d) with shared structure
    if use_correlated_noise:
        print(f"[DEBUG] Using correlated noise across {T} steps")
        Sigma_T = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                Sigma_T[i, j] = min(T - i, T - j)  # triangular structure

        Sigma_T *= sigma2_base  # scale the structure
        # Pre-sample noise vectors for each dimension
        noise_tensor = np.zeros((T, d))
        rng = np.random.default_rng()
        for dim in range(d):
            noise_tensor[:, dim] = rng.multivariate_normal(np.zeros(T), Sigma_T)
    else:
        print("[DEBUG] Using i.i.d. Gaussian noise (less optimal)")
        noise_tensor = np.random.normal(loc=0.0, scale=np.sqrt(sigma2_base), size=(T, d))

    # Step 6: DP-SGD Loop for each mode
    for m_idx in range(len(modes)):
        m = modes[m_idx]
        for t in range(T):
            batch_idx = np.random.choice(n, size=batch_size, replace=False)
            batch = data[batch_idx]

            diffs = batch - m  # shape (b, d)
            norms_sq = np.sum(diffs ** 2, axis=1)
            weights = np.exp(-norms_sq / (2 * bandwidth ** 2))  # (b,)
            grads = (diffs / (bandwidth ** 2)) * weights[:, None]  # (b, d)

            grad_avg = np.mean(grads, axis=0)
            z_t = noise_tensor[t]  # (d,)
            m = m + eta * (grad_avg + z_t)

        modes[m_idx] = m

    return modes
