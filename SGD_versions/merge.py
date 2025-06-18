# merge.py
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from ms import mean_shift 


def merge_modes(modes, bandwidth, k=1):
    threshold = k * bandwidth
    merged = []
    used = np.zeros(len(modes), dtype=bool)

    for i, mode in enumerate(modes):
        if used[i]:
            continue
        cluster = [mode]
        used[i] = True
        for j in range(i + 1, len(modes)):
            if not used[j] and np.linalg.norm(modes[j] - mode) <= threshold:
                cluster.append(modes[j])
                used[j] = True
        merged.append(np.mean(cluster, axis=0))

    print(f"[DEBUG] Merged {len(modes)} modes into {len(merged)} modes with threshold {threshold}")
    return np.array(merged)


def merge_modes_dbscan(modes, bandwidth, k=1):
    threshold = k * bandwidth
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(modes)
    labels = clustering.labels_
    merged = [np.mean(modes[labels == label], axis=0) for label in np.unique(labels)]
    print(f"[DEBUG] DBSCAN merged {len(modes)} modes into {len(merged)} clusters with eps={threshold}")
    return np.array(merged)

def refine_dpms_modes_with_ms(dpms_modes, bandwidth, T=10, k=1):
    """
    Refine DPMS modes by applying mean shift to the DPMS modes themselves,
    followed by a standard merge step using threshold k * bandwidth.

    Args:
        dpms_modes (np.ndarray): Modes produced by DPMS.
        bandwidth (float): Bandwidth used in the kernel.
        T (int): Number of mean shift iterations for refinement.
        k (float): Merge threshold multiplier.

    Returns:
        np.ndarray: Refined and merged mode array.
    """
    refined_modes = mean_shift(
        data=dpms_modes,
        initial_modes=dpms_modes,
        T=T,
        bandwidth=bandwidth,
        p=1.0  # Use all points (dpms_modes) as data
    )

    merged_modes = merge_modes(refined_modes, bandwidth, k=k)

    print(f"[DEBUG] Refined DPMS modes using MS + merge: {len(dpms_modes)} → {len(merged_modes)}")
    return merged_modes

def merge_modes_agglomerative(modes, n_clusters):
    """
    Merge modes into exactly n_clusters using agglomerative clustering.
    
    Args:
        modes (np.ndarray): Array of mode points, shape (num_modes, dim).
        n_clusters (int): Desired number of clusters.

    Returns:
        np.ndarray: Array of merged cluster centroids, shape (n_clusters, dim).
    """
    if len(modes) == 0:
        return np.empty((0, modes.shape[1]))

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(modes)

    merged = np.array([modes[labels == i].mean(axis=0) for i in range(n_clusters)])
    print(f"[DEBUG] Agglomerative merged {len(modes)} modes into {n_clusters} clusters.")
    return merged
