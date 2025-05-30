# univariate_normal_mixture.py

import numpy as np
import matplotlib.pyplot as plt
import os

from ms import mean_shift
from dpms import mean_shift_dp
from merge import merge_modes, merge_modes_dbscan, refine_dpms_modes_with_ms, merge_modes_agglomerative
from bandwidth import robust_fixed_bandwidth
from bandwidth_cv import calibrate_c_cv


def generate_univariate_mixture(n_samples, means):
    variances = 1.0
    samples_per_mode = n_samples // len(means)
    data = []
    for m in means:
        samples = np.random.normal(loc=m, scale=np.sqrt(variances), size=(samples_per_mode, 1))
        data.append(samples)
    return np.vstack(data), np.array(means).reshape(-1, 1)


def compute_mse(true_modes, predicted_modes):
    mse = 0
    for tm in true_modes:
        dists = np.linalg.norm(predicted_modes - tm, axis=1)
        mse += np.min(dists) ** 2
    return mse / len(true_modes)


def save_stats(filename, S, R, dp_var, bandwidth, epsilon, delta, c_opt=None):
    with open(filename, "w") as f:
        f.write(f"S: {S}\n")
        f.write(f"R: {R}\n")
        f.write(f"DP Variance: {dp_var}\n")
        f.write(f"Bandwidth: {bandwidth}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Delta: {delta}\n")
        if c_opt is not None:
            f.write(f"Optimal c: {c_opt}\n")


def save_mses(filename, ms_mses, dpms_mses):
    with open(filename, "w") as f:
        f.write("Run\tMS_MSE\tDPMS_MSE\n")
        for i, (ms_mse, dpms_mse) in enumerate(zip(ms_mses, dpms_mses)):
            f.write(f"{i+1}\t{ms_mse:.6f}\t{dpms_mse:.6f}\n")


def save_modes(filename, ms_modes_all, dpms_modes_all):
    with open(filename, "w") as f:
        for i, (ms_modes, dpms_modes) in enumerate(zip(ms_modes_all, dpms_modes_all)):
            f.write(f"Run {i+1}\n")
            f.write("MS Modes:\n")
            f.write(np.array2string(ms_modes, precision=4, floatmode='fixed'))
            f.write("\nDPMS Modes:\n")
            f.write(np.array2string(dpms_modes, precision=4, floatmode='fixed'))
            f.write("\n\n")


def main(
    n_samples,
    means_list,
    epsilon,
    delta,
    T,
    p,
    bw_multiplier,
    n_experiments,
    C,
    k_ms,
    k_dp,
    use_cv_bandwidth=True
):
    np.random.seed(42)
    data, true_modes = generate_univariate_mixture(n_samples, means_list)

    if use_cv_bandwidth:
        print("Calibrating c via cross-validation...")
        c_opt = calibrate_c_cv(data, T=T, p=p, merge_k=k_ms, verbose=True)
        print(f"Best c: {c_opt:.4f}")
    else:
        c_opt = bw_multiplier

    bandwidth = robust_fixed_bandwidth(data, c=c_opt)

    R = np.max(np.linalg.norm(data, axis=1))
    d = data.shape[1]
    K_C = np.exp(-0.5 * C ** 2)
    n = data.shape[0]
    S = (4 * R) / (n * (bandwidth ** d) * (K_C / (2 * R) ** d))
    sigma_squared = 2 * (S ** 2) * T * np.log(1.25 / delta) / (epsilon ** 2)

    ms_mses = []
    dpms_mses = []
    ms_modes_all = []
    dpms_modes_all = []

    for run in range(n_experiments):
        print(f"Experiment {run + 1} / {n_experiments}")

        ms_modes_raw = mean_shift(data, initial_modes=None, T=T, bandwidth=bandwidth, p=p)
        ms_modes = merge_modes(ms_modes_raw, bandwidth, k=k_ms)

        dp_modes_raw = mean_shift_dp(data, epsilon, delta, initial_modes=None, T=T, bandwidth=bandwidth, p=p, C=C)
        # dp_modes = merge_modes(dp_modes_raw, bandwidth, k=k_dp)
        # dp_modes = refine_dpms_modes_with_ms(dp_modes_raw, bandwidth, T=10, k=k_dp)
        dp_modes = merge_modes_dbscan(dp_modes_raw, bandwidth)
        # dp_modes = merge_modes_agglomerative(dp_modes_raw, n_clusters=len(means_list))

        ms_mse = compute_mse(true_modes, ms_modes)
        dpms_mse = compute_mse(true_modes, dp_modes)

        ms_mses.append(ms_mse)
        dpms_mses.append(dpms_mse)

        ms_modes_all.append(ms_modes)
        dpms_modes_all.append(dp_modes)

    os.makedirs("results", exist_ok=True)
    save_stats("results/univariate_normal_mixture_stats.txt", S, R, sigma_squared, bandwidth, epsilon, delta, c_opt=c_opt)
    save_mses("results/univariate_normal_mixture_ms_dpms_mses.txt", ms_mses, dpms_mses)
    save_modes("results/univariate_normal_mixture_ms_dpms_modes.txt", ms_modes_all, dpms_modes_all)

    print(f"MS RMSE: {np.sqrt(np.mean(ms_mses)):.4f}")
    print(f"DPMS RMSE: {np.sqrt(np.mean(dpms_mses)):.4f}")

    # Plot
    plt.figure(figsize=(8, 4))
    xs = np.linspace(min(means_list) - 5, max(means_list) + 5, 1000)

    def gaussian_pdf(x, mu, sigma):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    pdf = np.zeros_like(xs)
    for m in means_list:
        pdf += gaussian_pdf(xs, m, sigma=1.0)

    plt.plot(xs, pdf, label="True Distribution", color="blue")
    plt.hist(data, bins=100, density=True, alpha=0.3, color="gray", label="Data")
    plt.vlines(true_modes.flatten(), 0, max(pdf), color="green", linestyle="--", label="True Modes")
    plt.vlines(ms_modes.flatten(), 0, max(pdf), color="red", linestyle="-", label="MS Modes")
    plt.vlines(dp_modes.flatten(), 0, max(pdf), color="orange", linestyle="-", label="DPMS Modes")

    plt.title("Univariate Mean Shift vs DPMS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/univariate_normal_mixture_modes_comparison.png")
    plt.show()


if __name__ == "__main__":
    main(
        n_samples=10000,
        means_list=[-5, 0, 5],
        epsilon=1.0,
        delta=1e-6,
        T=20,
        p=0.1,
        bw_multiplier=0.2,  # fallback if use_cv_bandwidth=False
        n_experiments=20,
        C=1.1,
        k_ms=1,
        k_dp=1,
        use_cv_bandwidth=True
    )
