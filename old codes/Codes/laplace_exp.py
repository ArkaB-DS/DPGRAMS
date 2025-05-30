import numpy as np
import matplotlib.pyplot as plt
from MS import mean_shift as ms_mean_shift, merge_modes as ms_merge
from DPMS import mean_shift as dp_mean_shift, merge_modes as dp_merge, silverman_bandwidth

# Config
true_mode = 5
laplace_scale = 1 / np.sqrt(2)  # since variance = 2 * scale^2 = 1
epsilon = 1.0
delta = 1e-5
C = 1.1
num_experiments = 50
k_merge = 2
k_merge_dp = 2
noise_init_var = 0.25
init_count = 5

configs = [
    {"sample_size": 1000, "T": 20},
    {"sample_size": 1000, "T": 100},
    {"sample_size": 10000, "T": 20},
    {"sample_size": 10000, "T": 100}
]

manual_bandwidths = np.round(np.arange(0.6, 1.01, 0.1), 2)
BANDWIDTH_TYPES = ["silverman"] + [str(bw) for bw in manual_bandwidths]

def run_experiment(sample_size, T):
    results = {}
    for bw_type in BANDWIDTH_TYPES:
        ms_rmses, dp_rmses = [], []
        ms_all_modes, dp_all_modes = [], []
        dp_variances, bandwidths = [], []

        for _ in range(num_experiments):
            data = np.random.laplace(loc=true_mode, scale=laplace_scale, size=(sample_size, 1))
            initial_modes = np.random.normal(loc=true_mode, scale=np.sqrt(noise_init_var), size=(init_count, 1))

            if bw_type == "silverman":
                ms_bw = silverman_bandwidth(data)
                dp_bw = silverman_bandwidth(data)
            else:
                ms_bw = dp_bw = float(bw_type)

            ms_modes = ms_mean_shift(data, initial_modes=initial_modes, T=T)
            ms_merged = ms_merge(ms_modes, ms_bw, k=k_merge)
            ms_rmse = np.mean((ms_merged.flatten() - true_mode)**2)
            ms_rmses.append(ms_rmse)
            ms_all_modes.append(ms_merged.flatten())

            dp_modes = dp_mean_shift(data, epsilon, delta, initial_modes=initial_modes, T=T, C=C)
            dp_merged = dp_merge(dp_modes, dp_bw, k=k_merge_dp)
            dp_rmse = np.mean((dp_merged.flatten() - true_mode)**2)
            dp_rmses.append(dp_rmse)
            dp_all_modes.append(dp_merged.flatten())

            n, d = data.shape
            K_C = np.exp(-0.5 * C**2)
            S = (4 * np.sqrt(d)) / (n * (dp_bw**d) * (K_C / (2 * np.sqrt(d))**d))
            sigma_squared = 2 * (S**2) * T * np.log(1.25 / delta) / epsilon**2
            dp_variances.append(sigma_squared)

            bandwidths.append((ms_bw, dp_bw))

        results[bw_type] = {
            "ms_rmse": np.sqrt(np.mean(ms_rmses)),
            "dp_rmse": np.sqrt(np.mean(dp_rmses)),
            "ms_modes": ms_all_modes,
            "dp_modes": dp_all_modes,
            "dp_variances": dp_variances,
            "bandwidths": bandwidths,
            "ms_rmse_runs": ms_rmses,
            "dp_rmse_runs": dp_rmses
        }
    return results

# Run all configs
all_results = {}
for config in configs:
    key = f"n={config['sample_size']}, T={config['T']}"
    print(f"Running {key}...")
    all_results[key] = run_experiment(**config)

# Save summaries
with open("results_laplace.txt", "w") as f:
    for key, res in all_results.items():
        f.write(f"======== {key} ========\n")
        for bw_type in BANDWIDTH_TYPES:
            r = res[bw_type]
            f.write(f"-- BW={bw_type} --\n")
            f.write(f"MS RMSE     : {r['ms_rmse']:.4f}\n")
            f.write(f"DP MS RMSE  : {r['dp_rmse']:.4f}\n")
            f.write(f"Avg DP Var  : {np.mean(r['dp_variances']):.4f}\n\n")

# Save per-run MSEs and bandwidths
with open("mse_runs_laplace.txt", "w") as f:
    for key, res in all_results.items():
        f.write(f"======== {key} ========\n")
        for bw_type in BANDWIDTH_TYPES:
            f.write(f"-- BW={bw_type} --\n")
            r = res[bw_type]
            for i in range(num_experiments):
                ms_rmse = r['ms_rmse_runs'][i]
                dp_rmse = r['dp_rmse_runs'][i]
                ms_bw, dp_bw = r['bandwidths'][i]
                f.write(f"Run {i+1:02d}: MS RMSE = {ms_rmse:.4f}, DP RMSE = {dp_rmse:.4f}, MS BW = {ms_bw:.4f}, DP BW = {dp_bw:.4f}\n")
            f.write("\n")

# Save predicted modes per run
with open("modes_runs_laplace.txt", "w") as f:
    for key, res in all_results.items():
        f.write(f"======== {key} ========\n")
        for bw_type in BANDWIDTH_TYPES:
            f.write(f"-- BW={bw_type} --\n")
            r = res[bw_type]
            for i in range(num_experiments):
                ms_modes = ", ".join(f"{m:.4f}" for m in r['ms_modes'][i])
                dp_modes = ", ".join(f"{m:.4f}" for m in r['dp_modes'][i])
                f.write(f"Run {i+1:02d}:\n  MS  : [{ms_modes}]\n  DPMS: [{dp_modes}]\n")
            f.write("\n")

# Plot RMSE vs Bandwidth
for config_key, config_res in all_results.items():
    ms_rmse_vals = [config_res[bw]["ms_rmse"] for bw in BANDWIDTH_TYPES]
    dp_rmse_vals = [config_res[bw]["dp_rmse"] for bw in BANDWIDTH_TYPES]

    plt.figure(figsize=(10, 6))
    plt.plot(BANDWIDTH_TYPES, ms_rmse_vals, label='MS RMSE', marker='o')
    plt.plot(BANDWIDTH_TYPES, dp_rmse_vals, label='DPMS RMSE', marker='s')
    plt.xlabel("Bandwidth Types")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Bandwidth for {config_key} (Laplace)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"rmse_vs_bandwidth_laplace_{config_key}.png")
    plt.show()
