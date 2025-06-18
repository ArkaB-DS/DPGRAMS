# univariate_normal_mixture_dpms_sgd.py

import os
import numpy as np
import matplotlib.pyplot as plt

from ms import mean_shift
from mean_shift_dp_sgd import mean_shift_dp_sgd  # DP-GEMS
from merge import (
    merge_modes,
    merge_modes_dbscan,
    refine_dpms_modes_with_ms,
    merge_modes_agglomerative
)
from bandwidth import silverman_bandwidth

def generate_univariate_normal_mixture(n_samples, means):
    samples_per_mode = n_samples // len(means)
    data = np.vstack([
        np.random.normal(loc=m, scale=1.0, size=(samples_per_mode, 1))
        for m in means
    ])
    return data, np.array(means).reshape(-1, 1)

def compute_mse(true_modes, est_modes):
    return np.mean([
        np.min((est_modes.flatten() - tm.item()) ** 2)
        for tm in true_modes
    ])

def save_text(path, text):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)

def main():
    # ─── Experiment settings ───
    n_samples = 1000
    means_list = [-5, 0, 5]
    epsilon, delta = 1.0, 1e-6
    T = 20
    p = 0.1
    C = 1.1
    batch_size = 10
    n_runs = 20
    k_merge = 1

    # ─── Prepare results folder ───
    results_dir = "SGD_results/univariate_normal_mixture"
    os.makedirs(results_dir, exist_ok=True)

    # ─── Data ───
    np.random.seed(42)
    data, true_modes = generate_univariate_normal_mixture(n_samples, means_list)

    # ─── Bandwidth ───
    h = silverman_bandwidth(data)

    # ─── Precompute sensitivity & noise var ───
    R = np.max(np.abs(data))
    d = data.shape[1]
    K_C = np.exp(-0.5 * C**2)
    S = (4 * R) / (n_samples * (h**d) * (K_C / (2 * R)**d))
    sigma2_iid = 2 * (S**2) * T * np.log(1.25 / delta) / (epsilon**2)
    sigma2_corr = 2 * (S**2) * np.log(1.25 / delta) / (epsilon**2)

    print(f"Bandwidth h: {h:.4f}")
    print(f"Sensitivity S: {S:.4e}")
    print(f"\u03c3²_iid (per-step): {sigma2_iid:.4e}")
    print(f"\u03c3²_corr (final): {sigma2_corr:.4e}")

    ms_mses, dp_mses = [], []
    ms_modes_all, dp_modes_all = [], []

    for run in range(n_runs):
        print(f"[Run {run+1}/{n_runs}]")

        # Non-private Mean Shift + merge
        ms_raw = mean_shift(data, T=T, bandwidth=h, p=p)
        ms_est = merge_modes(ms_raw, h, k=k_merge)

        # DP-GEMS
        dp_raw = mean_shift_dp_sgd(
            data, epsilon, delta,
            initial_modes=None,
            T=T, bandwidth=h, p=p,
            kernel_C=C, batch_size=batch_size,
            eta=None, use_correlated_noise=True
        )
        # Choose one merge method by uncommenting:
        # dp_est = merge_modes(dp_raw, h, k=k_merge)
        # dp_est = merge_modes_dbscan(dp_raw, h)
        # dp_est = refine_dpms_modes_with_ms(dp_raw, h, T=10, k=k_merge)
        dp_est = merge_modes_agglomerative(dp_raw, n_clusters=len(means_list))

        # Metrics
        ms_mse = compute_mse(true_modes, ms_est)
        dp_mse = compute_mse(true_modes, dp_est)
        ms_mses.append(ms_mse)
        dp_mses.append(dp_mse)
        ms_modes_all.append(ms_est)
        dp_modes_all.append(dp_est)

    # ─── Save stats ───
    stats_text = (
        f"S: {S:.4e}\n"
        f"σ²_iid: {sigma2_iid:.4e}\n"
        f"σ²_corr: {sigma2_corr:.4e}\n"
    )
    save_text(f"{results_dir}/stats.txt", stats_text)

    # ─── Save MSEs ───
    lines = ["Run\tMS_MSE\tDPMS_MSE"] + [
        f"{i+1}\t{ms:.6f}\t{dp:.6f}"
        for i, (ms, dp) in enumerate(zip(ms_mses, dp_mses))
    ]
    save_text(f"{results_dir}/mses.txt", "\n".join(lines))

    # ─── Save Modes ───
    mode_lines = []
    for i, (msm, dpm) in enumerate(zip(ms_modes_all, dp_modes_all)):
        mode_lines += [
            f"Run {i+1}",
            "MS Modes:", np.array2string(msm, precision=4),
            "DPMS Modes:", np.array2string(dpm, precision=4),
            ""
        ]
    save_text(f"{results_dir}/modes.txt", "\n".join(mode_lines))

    # ─── Summary to console ───
    print(f"MS RMSE: {np.sqrt(np.mean(ms_mses)):.4f}")
    print(f"DP RMSE: {np.sqrt(np.mean(dp_mses)):.4f}")

    # ─── Plot ───
    xs = np.linspace(min(means_list)-5, max(means_list)+5, 1000)
    def gaussian_pdf(x, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    pdf = sum(gaussian_pdf(xs, m, sigma=1.0) for m in means_list)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, pdf, label="True PDF (normal mix)")
    plt.hist(data.flatten(), bins=100, density=True, alpha=0.3, label="Data")
    plt.vlines(true_modes.flatten(), 0, max(pdf), colors="green", linestyle="--", label="True Modes")
    plt.vlines(ms_est.flatten(), 0, max(pdf), colors="red", label="MS Modes")
    plt.vlines(dp_est.flatten(), 0, max(pdf), colors="orange", label="DPMS Modes")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{results_dir}/plot.png")
    plt.show()

if __name__ == "__main__":
    main()
