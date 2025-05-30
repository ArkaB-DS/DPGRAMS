import numpy as np
from sklearn.model_selection import KFold
from ms import mean_shift
from merge import merge_modes
from bandwidth import robust_fixed_bandwidth

def calibrate_c_cv(
    data,
    c_values = np.arange(0.2, 2.1, 0.2),
    n_folds=5,
    T=10,
    p=0.1,
    merge_k=1,
    seed=0,
    verbose=False
):
    """
    Cross-validated selection of multiplier `c` for robust_fixed_bandwidth.

    Args:
        data (np.ndarray): Input array of shape (n_samples,) or (n_samples, d).
        c_values (list): List of c candidates for bandwidth = c * MAD / 0.6745.
        n_folds (int): Number of cross-validation folds.
        T (int): Mean shift iterations.
        p (float): Shrinkage parameter.
        merge_k (int): Merge parameter for mode merging.
        seed (int): Random seed for fold splitting.
        verbose (bool): If True, print CV score per c.

    Returns:
        float: Best c minimizing CV error.
    """
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:  # transpose if needed
        data = data.T

    best_c = None
    best_score = np.inf
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for c in c_values:
        bandwidth = robust_fixed_bandwidth(data, c=c)
        fold_scores = []

        for train_idx, test_idx in kf.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]

            # Run Mean Shift
            modes = mean_shift(train_data, initial_modes=None, T=T, bandwidth=bandwidth, p=p)
            merged = merge_modes(modes, bandwidth, k=merge_k)

            # Compute mean squared distance to closest mode
            dists = np.linalg.norm(test_data[:, None, :] - merged[None, :, :], axis=-1)
            fold_scores.append(np.mean(np.min(dists**2, axis=1)))

        avg_score = np.mean(fold_scores)
        if verbose:
            print(f"c = {c:.3f} -> CV MSE = {avg_score:.6f}")

        if avg_score < best_score:
            best_score = avg_score
            best_c = c

    return best_c
