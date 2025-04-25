import numpy as np
import matplotlib.pyplot as plt
from MS import mean_shift as ms_mean_shift, merge_modes as ms_merge
from DPMS import mean_shift as dp_mean_shift, merge_modes as dp_merge, silverman_bandwidth

# Config
epsilon = 1.0
delta = 1e-5
C = 1.1
num_experiments = 50
k_merge = 2
k_merge_dp = 2
noise_init_var = 0.025
init_count = 10
k_modal = 3  # You can change this to 2, 5, etc.

configs = [
    {"sample_size": 1000, "T": 20},
    {"sample_size": 1000, "T": 100},
    {"sample_size": 10000, "T": 20},
    {"sample_size": 10000, "T": 100}
]

manual_bandwidths = np.round(np.arange(0.6, 1.01, 0.1), 2)
BANDWIDTH_TYPES = ["silverman"] + [str(bw) for bw in manual_bandwidths]


def compute_mode_rmse(pred_modes, true_modes):
    pred_modes_sorted = np.sort(pred_modes)
    true_modes_sorted = np.sort(true_modes)
    min_len = min(len(pred_modes_sorted), len(true_modes_sorted))
    rmse = np.mean((pred_modes_sorted[:min_len] - true_modes_sorted[:min_len]) ** 2)
    return rmse


def run_experiment(sample_size, T, k=k_modal):
    results = {}
    # True modes (we use 0, 5, and 10 as the true modes)
    true_modes = np.arange(0, 5 * k, 5)  # [0, 5, 10]
    weights = np.ones(k) / k
    variances = np.ones(k)

    for bw_type in BANDWIDTH_TYPES:
        ms_rmses, dp_rmses = [], []
        ms_all_modes, dp_all_modes = [], []
        dp_variances, bandwidths = [], []

        for _ in range(num_experiments):
            data = np.concatenate([
                np.random.normal(loc=mode, scale=np.sqrt(var), size=int(weight * sample_size))
                for mode, var, weight in zip(true_modes, variances, weights)
            ]).reshape(-1, 1)
            np.random.shuffle(data)

            # Generate initial modes with 5 initializations for each true mode
            initial_modes = []
            for mode in true_modes:
                initial_modes.extend(np.random.normal(loc=mode, scale=0.2, size=5))  # 5 noisy initial modes per true mode
            initial_modes = np.array(initial_modes).reshape(-1, 1)

            if bw_type == "silverman":
                ms_bw = silverman_bandwidth(data)
                dp_bw = silverman_bandwidth(data)
            else:
                ms_bw = dp_bw = float(bw_type)

            # MS
            ms_modes = ms_mean_shift(data, initial_modes=initial_modes, T=T)
            ms_merged = ms_merge(ms_modes, ms_bw, k=k_merge)
            ms_rmse = compute_mode_rmse(ms_merged.flatten(), true_modes)
            ms_rmses.append(ms_rmse)
            ms_all_modes.append(ms_merged.flatten())

            # DPMS
            dp_modes = dp_mean_shift(data, epsilon, delta, initial_modes=initial_modes, T=T, C=C)
            dp_merged = dp_merge(dp_modes, dp_bw, k=k_merge_dp)
            dp_rmse = compute_mode_rmse(dp_merged.flatten(), true_modes)
            dp_rmses.append(dp_rmse)
            dp_all_modes.append(dp_merged.flatten())

            # DP variance
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
with open("uni_mix_results.txt", "w") as f:
    for key, res in all_results.items():
        f.write(f"======== {key} ========\n")
        for bw_type in BANDWIDTH_TYPES:
            r = res[bw_type]
            f.write(f"-- BW={bw_type} --\n")
            f.write(f"MS RMSE     : {r['ms_rmse']:.4f}\n")
            f.write(f"DP MS RMSE  : {r['dp_rmse']:.4f}\n")
            f.write(f"Avg DP Var  : {np.mean(r['dp_variances']):.4f}\n\n")

# Save per-run MSEs and bandwidths
with open("uni_mix_mse_runs.txt", "w") as f:
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

# Save all predicted modes per run 
with open("uni_mix_modes.txt", "w") as f:
    for key, res in all_results.items():
        f.write(f"======== {key} ========\n")
        for bw_type in BANDWIDTH_TYPES:
            f.write(f"-- BW={bw_type} --\n")
            r = res[bw_type]
            for i in range(num_experiments):
                ms_modes_arr = r['ms_modes'][i]
                dp_modes_arr = r['dp_modes'][i]
                ms_modes_str = ", ".join(f"{m:.4f}" for m in ms_modes_arr)
                dp_modes_str = ", ".join(f"{m:.4f}" for m in dp_modes_arr)
                f.write(f"Run {i+1:02d}:\n")
                f.write(f"  MS Modes   : [{ms_modes_str}]\n")
                f.write(f"  DPMS Modes : [{dp_modes_str}]\n")
            f.write("\n")

# Plot RMSE vs Bandwidth for MS and DPMS
for config_key, config_res in all_results.items():
    ms_rmse_vals = [config_res[bw]["ms_rmse"] for bw in BANDWIDTH_TYPES]
    dp_rmse_vals = [config_res[bw]["dp_rmse"] for bw in BANDWIDTH_TYPES]

    plt.figure(figsize=(10, 6))
    plt.plot(BANDWIDTH_TYPES, ms_rmse_vals, label='MS RMSE', marker='o')
    plt.plot(BANDWIDTH_TYPES, dp_rmse_vals, label='DPMS RMSE', marker='s')
    plt.xlabel("Bandwidth Types")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Bandwidth for {config_key}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"rmse_vs_bandwidth_{config_key}.png")
    plt.show()
