# bivariate_normal_4mix
import numpy as np
import matplotlib.pyplot as plt
import os

from ms import mean_shift
from dpms import mean_shift_dp
from merge import merge_modes, merge_modes_dbscan, refine_dpms_modes_with_ms, merge_modes_agglomerative
from bandwidth import silverman_bandwidth, robust_fixed_bandwidth
from bandwidth_cv import calibrate_c_cv

def generate_4corner_data(n_samples):
    means = np.array([[5, 5], [5, -5], [-5, 5], [-5, -5]])
    cov = np.eye(2)
    samples_per_corner = n_samples // 4
    data = []
    for mean in means:
        data.append(np.random.multivariate_normal(mean, cov, samples_per_corner))
    return np.vstack(data), means

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
    data, true_modes = generate_4corner_data(n_samples)

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
        # dp_modes = merge_modes_dbscan(dp_modes_raw, bandwidth)
        dp_modes = merge_modes_agglomerative(dp_modes_raw, n_clusters=4)

        ms_mse = compute_mse(true_modes, ms_modes)
        dpms_mse = compute_mse(true_modes, dp_modes)

        ms_mses.append(ms_mse)
        dpms_mses.append(dpms_mse)

        ms_modes_all.append(ms_modes)
        dpms_modes_all.append(dp_modes)

    os.makedirs("results", exist_ok=True)
    save_stats("results/bivariate_normal_4mix_stats.txt", S, R, sigma_squared, bandwidth, epsilon, delta, c_opt=c_opt)
    save_mses("results/bivariate_normal_4mix_ms_dpms_mses.txt", ms_mses, dpms_mses)
    save_modes("results/bivariate_normal_4mix_.txt", ms_modes_all, dpms_modes_all)

    print(f"MS RMSE: {np.sqrt(np.mean(ms_mses)):.4f}")
    print(f"DPMS RMSE: {np.sqrt(np.mean(dpms_mses)):.4f}")

    # Final plotting
    plt.figure(figsize=(8, 6))
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    def mvn_pdf(x, mean, cov):
        size = mean.shape[0]
        det = np.linalg.det(cov)
        norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det))
        diff = x - mean
        inv_cov = np.linalg.inv(cov)
        exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=-1))
        return norm_const * exp_term

    cov = np.eye(2)
    Z = sum(mvn_pdf(pos, m, cov) for m in true_modes)

    plt.contour(X, Y, Z, levels=10, cmap="Blues")
    plt.scatter(true_modes[:, 0], true_modes[:, 1], c="green", s=100, label="True Modes")
    plt.scatter(ms_modes[:, 0], ms_modes[:, 1], c="red", marker="x", s=100, label="MS Modes")
    plt.scatter(dp_modes[:, 0], dp_modes[:, 1], c="orange", marker="o", s=100, label="DPMS Modes")
    plt.title("Mean Shift vs Differentially Private Mean Shift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/bivariate_normal_4mix_modes_comparison.png")
    plt.show()


if __name__ == "__main__":
    main(
        n_samples=10000,
        epsilon=1.0,
        delta=1e-6,
        T=20,
        p=0.1,
        bw_multiplier=0.4,
        n_experiments=20,
        C=1.1,
        k_ms=1,
        k_dp=1,
        use_cv_bandwidth=False
    )
