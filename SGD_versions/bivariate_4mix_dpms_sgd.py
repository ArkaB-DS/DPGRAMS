# bivariate_4mix_dpms_sgd.py

import os
import numpy as np
import matplotlib.pyplot as plt

from ms import mean_shift
from mean_shift_dp_sgd import mean_shift_dp_sgd
from merge import (
    merge_modes,
    merge_modes_dbscan,
    refine_dpms_modes_with_ms,
    merge_modes_agglomerative
)
from bandwidth import silverman_bandwidth


def generate_4corners(n_samples):
    """4 isotropic Gaussians at the corners of a square."""
    means = np.array([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])
    cov = np.eye(2)
    pts = [np.random.multivariate_normal(m, cov, n_samples // 4) for m in means]
    return np.vstack(pts), means


def compute_mse(true_modes, est_modes):
    """MSE between true modes and estimated modes."""
    return np.mean([
        np.min(np.sum((est_modes - tm)**2, axis=1))
        for tm in true_modes
    ])


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main():
    # ─── Experiment settings ────────────────────────────────────────────────
    n_samples      = 1000
    epsilon, delta = 1.0, 1e-6
    T              = 20
    p              = 0.1
    C              = 1.1
    batch_size     = 100
    n_runs         = 20
    k_merge        = 1

    # ─── Prepare results folder ──────────────────────────────────────────────
    results_dir = "SGD_results/bivariate_4mix_dpms_sgd"
    os.makedirs(results_dir, exist_ok=True)

    # ─── Data ────────────────────────────────────────────────────────────────
    np.random.seed(42)
    data, true_modes = generate_4corners(n_samples)

    # ─── Bandwidth ────────────────────────────────────────────────────────────
    h = silverman_bandwidth(data)

        # ─── Precompute sensitivity & noise var for logging ───────────────────────
    # Global sensitivity S
    R    = np.max(np.linalg.norm(data, axis=1))
    d    = data.shape[1]
    K_C  = np.exp(-0.5 * C**2)
    S    = (4 * R) / (n_samples * (h**d) * (K_C / (2 * R)**d))
    # Independent noise variance (i.i.d.)
    sigma2_iid  = 2 * (S**2) * T * np.log(1.25 / delta) / (epsilon**2)
    # Correlated noise variance for final iterate
    sigma2_corr = 2 * (S**2) * np.log(1.25 / delta) / (epsilon**2)
    print(f"[DEBUG] Using correlated noise -> final variance: {sigma2_corr:.4e}")
    print(f"[DEBUG] Independent noise per-step variance would be: {sigma2_iid:.4e}")

    ms_mses, dp_mses = [], []
    ms_modes_all, dp_modes_all = [], []

    # ─── Runs ────────────────────────────────────────────────────────────────
    for run in range(n_runs):
        print(f"[Run {run+1}/{n_runs}]")

        # Non-private Mean Shift + merge
        ms_raw = mean_shift(data, T=T, bandwidth=h, p=p)
        ms_est = merge_modes(ms_raw, h, k=k_merge)

        # DP‑GEMS: private mean shift via DP-SGD
        dp_raw = mean_shift_dp_sgd(
            data, epsilon, delta,
            initial_modes=None,
            T=T, bandwidth=h, p=p,
            kernel_C=C, batch_size=batch_size,
            eta=None, use_correlated_noise=False
        )
        # Choose one merge method by uncommenting:
        dp_est = merge_modes(dp_raw, h, k=k_merge)
        # dp_est = merge_modes_dbscan(dp_raw, h)
        # dp_est = refine_dpms_modes_with_ms(dp_raw, h, T=10, k=k_merge)
        dp_est = merge_modes_agglomerative(dp_raw, n_clusters=4)

        # Metrics
        ms_mse = compute_mse(true_modes, ms_est)
        dp_mse = compute_mse(true_modes, dp_est)
        ms_mses.append(ms_mse)
        dp_mses.append(dp_mse)
        ms_modes_all.append(ms_est)
        dp_modes_all.append(dp_est)

    # ─── Save stats ───────────────────────────────────────────────────────────
    stats = (
        f"Sensitivity S: {S:.4e}\n"
        f"Data radius R: {R:.4f}\n"
        f"Noise var σ²(iid): {sigma2_iid:.4e}\n"
        f"Noise var σ²(corr): {sigma2_corr:.4e}\n"
        f"Bandwidth h: {h:.4f}\n"
        f"ε, δ = ({epsilon}, {delta})\n"
    )
    save_text(f"{results_dir}/stats.txt", stats)

    # ─── Save MSEs ────────────────────────────────────────────────────────────
    lines = ["Run\tMS_MSE\tDPMS_MSE"] + [
        f"{i+1}\t{ms:.6f}\t{dp:.6f}"
        for i, (ms, dp) in enumerate(zip(ms_mses, dp_mses))
    ]
    save_text(f"{results_dir}/mses.txt", "\n".join(lines))

    # ─── Save Modes ───────────────────────────────────────────────────────────
    mode_lines = []
    for i, (msm, dpm) in enumerate(zip(ms_modes_all, dp_modes_all)):
        mode_lines.append(f"Run {i+1}")
        mode_lines.append("MS Modes:")
        mode_lines.append(np.array2string(msm, precision=4))
        mode_lines.append("DPMS Modes:")
        mode_lines.append(np.array2string(dpm, precision=4))
        mode_lines.append("")
    save_text(f"{results_dir}/modes.txt", "\n".join(mode_lines))

    # ─── Summary to console ───────────────────────────────────────────────────
    print(f"MS  RMSE: {np.sqrt(np.mean(ms_mses)):.4f}")
    print(f"DP  RMSE: {np.sqrt(np.mean(dp_mses)):.4f}")

    # ─── Plot ─────────────────────────────────────────────────────────────────
    x = np.linspace(-10, 10, 200)
    Xg, Yg = np.meshgrid(x, x)
    pos = np.dstack((Xg, Yg))
    cov = np.eye(2)

    def mvn_pdf(pt, mean):
        diff = pt - mean
        inv = np.linalg.inv(cov)
        return (1.0 / (2*np.pi*np.sqrt(np.linalg.det(cov)))) * np.exp(-0.5 * np.sum(diff@inv*diff, axis=-1))

    Z = sum(mvn_pdf(pos, m) for m in true_modes)
    plt.figure(figsize=(8,6))
    plt.contour(Xg, Yg, Z, levels=10, cmap="Blues")
    plt.scatter(*true_modes.T, c="green", s=80, label="True Modes")
    plt.scatter(*ms_est.T,    c="red",   marker="x", s=80, label="MS")
    plt.scatter(*dp_est.T,    c="orange",marker="o", s=80, label="DPMS")
    plt.title("DP-GEMS on 4-Mode Gaussian Mixture")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{results_dir}/plot.png")
    plt.show()


if __name__ == "__main__":
    main()
