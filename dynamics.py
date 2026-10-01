"""Dynamics analysis of raw simulation traces: Effective Rank and spike kinetics.

Quantifies the dynamic state of single-neuron reservoir from raw waveforms:
1. Detects bAPs (global soma-originated spikes) and dSpikes (localized dendritic spikes).
2. Decomposes spatio-temporal calcium dynamics into global wave (PC1) and local residual.
3. Computes local Effective Rank (Roy & Vetterli 2007) around spike events.
4. Fits dSpike and bAP slopes (R(PC1 Excluded) vs R(All PCs)) to test APCCAS findings.

Usage:
    # Compute on cluster / local machines:
    uv run python dynamics.py mode=compute 'run_dir=multirun/2026-09-30/local-sweep-caracal/*' data_dir=data/2026-09-30-caracal n_jobs=12

    # Plot across multiple runs:
    uv run python dynamics.py mode=plot \\
        dynamics_path='["data/2026-09-30_sweep200/caracal/dynamics.json", "data/2026-09-30_sweep200/oncilla/dynamics.json"]' \\
        accuracy_path='["data/2026-09-30_sweep200/caracal/results.json", "data/2026-09-30_sweep200/oncilla/results.json"]' \\
        sparsity_path='["data/2026-09-30_sweep200/caracal/sparsity.json", "data/2026-09-30_sweep200/oncilla/sparsity.json"]'
"""

import gc
import glob
import json
import logging
import os
import sys
import zipfile

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# 1. Effective Rank and Spike Detection Utilities
# ==============================================================================

def calculate_effective_rank(sub_data: np.ndarray) -> float:
    """Compute Effective Rank (Roy & Vetterli, 2007) of a matrix.

    sub_data: shape (T_window, N_compartments)
    """
    if sub_data.size == 0 or np.all(sub_data == 0):
        return 1.0

    try:
        _, s, _ = np.linalg.svd(sub_data, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    sigma_sum = np.sum(s)
    if sigma_sum == 0:
        return 1.0

    p = s / sigma_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def get_spike_indices(
    vm: np.ndarray,
    t: np.ndarray,
    v_threshold: float = -30.0,
    exclusion_window_ms: float = 10.0,
):
    """Detect bAP and dSpike timestamps using vectorized NumPy operations.

    vm: shape (T, N_compartments)
    t: shape (T,)
    v_threshold: spike threshold in mV
    exclusion_window_ms: window around bAP to exclude localized spikes
    """
    dt = float(t[1] - t[0])
    mean_vm = np.mean(vm, axis=1)

    # 1. bAP: somatic/global action potential (average Vm crosses threshold upwards)
    bap_cond = (mean_vm[:-1] < v_threshold) & (mean_vm[1:] >= v_threshold)
    bap_indices = np.where(bap_cond)[0]

    # 2. Local spikes: any compartment crossing threshold upwards
    spike_matrix = (vm[:-1, :] < v_threshold) & (vm[1:, :] >= v_threshold)
    all_local_spikes = np.where(np.any(spike_matrix, axis=1))[0]

    # 3. dSpike: local spikes occurring outside +/- exclusion_window of any bAP
    exclusion_samples = int(np.round(exclusion_window_ms / dt))
    if len(bap_indices) == 0:
        dspike_indices = all_local_spikes
    elif len(all_local_spikes) == 0:
        dspike_indices = np.array([], dtype=int)
    else:
        # Vectorized check: distance between every local spike and every bAP
        diffs = np.abs(all_local_spikes[:, None] - bap_indices[None, :])
        min_dist_to_bap = np.min(diffs, axis=1)
        dspike_indices = all_local_spikes[min_dist_to_bap >= exclusion_samples]

    return bap_indices, dspike_indices


# ==============================================================================
# 2. Single Run Processing
# ==============================================================================

def process_single_run(
    job_dir: str,
    window_ms: float = 5.0,
    v_threshold: float = -30.0,
    exclusion_window_ms: float = 10.0,
    test_only: bool = True,
):
    """Extract spikes, calculate Effective Rank, and compute slope metrics for one run."""
    job_dir = os.path.abspath(job_dir)
    cfg_path = os.path.join(job_dir, ".hydra", "config.yaml")
    run_info_path = os.path.join(job_dir, "data", "run_info.npz")

    if not os.path.isdir(job_dir):
        return None

    # Load configuration metadata
    meta = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = OmegaConf.load(f)
            meta["syn_loc_mean"] = cfg.get("syn_loc_mean", None)
            meta["syn_loc_std"] = cfg.get("syn_loc_std", None)
            meta["seed"] = cfg.get("seed", None)
            if "sample" in cfg and isinstance(cfg.sample, dict):
                meta["sample_id"] = cfg.sample.get("sample_id", None)
        except Exception as e:
            logger.warning(f"Failed to read config in {job_dir}: {e}")

    if os.path.exists(run_info_path):
        try:
            with np.load(run_info_path, allow_pickle=True) as info:
                for k in ["sample_id", "seed", "syn_loc_mean", "syn_loc_std"]:
                    if k in info and meta.get(k) is None:
                        val = info[k]
                        meta[k] = val.item() if hasattr(val, "item") else val
        except Exception:
            pass

    meta["job_id"] = os.path.basename(job_dir)
    meta["run_dir"] = job_dir

    # Find buffer files
    buffer_pattern = os.path.join(job_dir, "data", "buffer*.npz")
    all_buffers = sorted(glob.glob(buffer_pattern))

    if not all_buffers:
        logger.warning(f"No buffer files found in {job_dir}")
        return None

    # Filter test-only buffers (typically buffer10 to buffer59 if training has 10 trials)
    if test_only:
        # Buffer files are named buffer00.npz ... buffer59.npz
        # If there are 60 buffers, buffer 0-9 are train, 10-59 are test
        test_buffers = [bf for bf in all_buffers if not os.path.basename(bf).startswith("buffer0")]
        buffers_to_process = test_buffers if test_buffers else all_buffers
    else:
        buffers_to_process = all_buffers

    bap_er_pairs = []
    dspike_er_pairs = []
    total_duration_s = 0.0

    for bf in buffers_to_process:
        try:
            with np.load(bf, allow_pickle=True) as data_ts:
                t = data_ts["t_rec"]
                ca = data_ts["variables"][0]  # Ca (T, N)
                vm = data_ts["variables"][1]  # Vm (T, N)
        except (zipfile.BadZipFile, ValueError, KeyError, IndexError, EOFError, OSError) as e:
            logger.warning(f"Corrupted buffer file {bf}: {e}")
            continue

        # Ensure (T, N) orientation
        if ca.shape[1] > ca.shape[0]:
            ca = ca.T
            vm = vm.T

        dt = float(t[1] - t[0])
        window_size = int(np.round(window_ms / dt))
        half_w = window_size // 2
        T_len = ca.shape[0]

        total_duration_s += float(t[-1] - t[0]) / 1000.0

        # Detect spikes
        bap_idxs, dspike_idxs = get_spike_indices(
            vm, t, v_threshold=v_threshold, exclusion_window_ms=exclusion_window_ms
        )

        # PCA: separate PC1 (global wave) from localized dynamics
        try:
            pca = PCA(n_components=1)
            ca_pca = pca.fit_transform(ca)
            ca_proj = pca.inverse_transform(ca_pca)
            ca_res = ca - ca_proj
        except Exception as e:
            logger.warning(f"PCA failed on {bf}: {e}")
            continue

        # Effective Rank for bAP
        for idx in bap_idxs:
            if idx < half_w or idx > T_len - half_w:
                continue
            w_slice = slice(idx - half_w, idx + half_w)
            er_orig = calculate_effective_rank(ca[w_slice, :])
            er_res = calculate_effective_rank(ca_res[w_slice, :])
            bap_er_pairs.append((er_orig, er_res))

        # Effective Rank for dSpike
        for idx in dspike_idxs:
            if idx < half_w or idx > T_len - half_w:
                continue
            w_slice = slice(idx - half_w, idx + half_w)
            er_orig = calculate_effective_rank(ca[w_slice, :])
            er_res = calculate_effective_rank(ca_res[w_slice, :])
            dspike_er_pairs.append((er_orig, er_res))

        del ca, vm, ca_proj, ca_res
        gc.collect()

    # Aggregate metrics
    meta["total_duration_s"] = total_duration_s
    meta["bap_count"] = len(bap_er_pairs)
    meta["dspike_count"] = len(dspike_er_pairs)
    meta["bap_rate"] = float(len(bap_er_pairs) / total_duration_s) if total_duration_s > 0 else 0.0
    meta["dspike_rate"] = float(len(dspike_er_pairs) / total_duration_s) if total_duration_s > 0 else 0.0

    # Fit dSpike slope
    if len(dspike_er_pairs) >= 2:
        x_d = np.array([p[0] for p in dspike_er_pairs]).reshape(-1, 1)
        y_d = np.array([p[1] for p in dspike_er_pairs])
        lr_d = LinearRegression().fit(x_d, y_d)
        meta["dspike_slope"] = float(lr_d.coef_[0])
        meta["dspike_intercept"] = float(lr_d.intercept_)
        meta["dspike_r2"] = float(lr_d.score(x_d, y_d))
        meta["mean_dspike_er_orig"] = float(np.mean(x_d))
        meta["mean_dspike_er_res"] = float(np.mean(y_d))
    else:
        meta["dspike_slope"] = np.nan
        meta["dspike_intercept"] = np.nan
        meta["dspike_r2"] = np.nan
        meta["mean_dspike_er_orig"] = np.nan
        meta["mean_dspike_er_res"] = np.nan

    # Fit bAP slope
    if len(bap_er_pairs) >= 2:
        x_b = np.array([p[0] for p in bap_er_pairs]).reshape(-1, 1)
        y_b = np.array([p[1] for p in bap_er_pairs])
        lr_b = LinearRegression().fit(x_b, y_b)
        meta["bap_slope"] = float(lr_b.coef_[0])
        meta["bap_intercept"] = float(lr_b.intercept_)
        meta["bap_r2"] = float(lr_b.score(x_b, y_b))
        meta["mean_bap_er_orig"] = float(np.mean(x_b))
        meta["mean_bap_er_res"] = float(np.mean(y_b))
    else:
        meta["bap_slope"] = np.nan
        meta["bap_intercept"] = np.nan
        meta["bap_r2"] = np.nan
        meta["mean_bap_er_orig"] = np.nan
        meta["mean_bap_er_res"] = np.nan

    return meta


# ==============================================================================
# 3. Compute Mode
# ==============================================================================

def run_compute(cfg: DictConfig):
    """Parallel batch computation across all runs."""
    if not cfg.run_dir:
        raise ValueError("run_dir must be specified for mode=compute (e.g. 'multirun/.../*')")

    orig_cwd = hydra.utils.get_original_cwd()
    pattern = cfg.run_dir
    if not os.path.isabs(pattern):
        pattern = os.path.join(orig_cwd, pattern)

    job_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not job_dirs:
        raise FileNotFoundError(f"No run directories matched pattern: {pattern}")

    logger.info(f"Computing dynamics metrics for {len(job_dirs)} runs...")

    data_dir = cfg.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(orig_cwd, data_dir)
    os.makedirs(data_dir, exist_ok=True)

    n_jobs = cfg.get("n_jobs", 12)
    window_ms = cfg.get("window_ms", 5.0)
    v_threshold = cfg.get("v_threshold", -30.0)
    exclusion_window_ms = cfg.get("exclusion_window_ms", 10.0)
    test_only = cfg.get("test_only", True)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_run)(
            jd,
            window_ms=window_ms,
            v_threshold=v_threshold,
            exclusion_window_ms=exclusion_window_ms,
            test_only=test_only,
        )
        for jd in job_dirs
    )

    valid_results = [r for r in results if r is not None]
    out_file = os.path.join(data_dir, "dynamics.json")
    with open(out_file, "w") as f:
        json.dump(valid_results, f, indent=2)

    logger.info(f"Successfully computed dynamics for {len(valid_results)}/{len(job_dirs)} runs.")
    logger.info(f"Results saved to: {out_file}")


# ==============================================================================
# 4. Plot Mode
# ==============================================================================

def _load_json_list(paths, orig_cwd):
    """Load JSON from a single string path or list of paths, resolving relatives."""
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    records = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(orig_cwd, p)
        if not os.path.exists(p):
            logger.warning(f"File not found: {p}")
            continue
        with open(p, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            # If accuracy results dict with 'results' key
            if "results" in data and isinstance(data["results"], list):
                records.extend(data["results"])
            else:
                records.append(data)
    return records


def run_plot(cfg: DictConfig):
    """Merge dynamics, accuracy, and sparsity to evaluate APCCAS relationships."""
    orig_cwd = hydra.utils.get_original_cwd()

    dynamics_records = _load_json_list(cfg.get("dynamics_path"), orig_cwd)
    accuracy_records = _load_json_list(cfg.get("accuracy_path"), orig_cwd)
    sparsity_records = _load_json_list(cfg.get("sparsity_path"), orig_cwd)

    if not dynamics_records:
        raise ValueError("No dynamics data found. Specify dynamics_path.")

    df_dyn = pd.DataFrame(dynamics_records)
    logger.info(f"Loaded dynamics records: {len(df_dyn)}")

    # Merge Accuracy
    if accuracy_records:
        df_acc = pd.DataFrame(accuracy_records)
        # Normalize columns: mean_accuracy or accuracy
        if "mean_accuracy" in df_acc.columns and "accuracy" not in df_acc.columns:
            df_acc["accuracy"] = df_acc["mean_accuracy"]

        # Match key: sample_id or job_id
        merge_key = "sample_id" if ("sample_id" in df_dyn.columns and "sample_id" in df_acc.columns) else "job_id"
        df_merged = pd.merge(df_dyn, df_acc[[merge_key, "accuracy"]].drop_duplicates(subset=[merge_key]), on=merge_key, how="inner")
        logger.info(f"Merged with accuracy on '{merge_key}': {len(df_merged)} rows")
    else:
        logger.warning("No accuracy data provided. Plots against accuracy will be skipped.")
        df_merged = df_dyn

    # Merge Sparsity if present
    if sparsity_records and "sparsity_inter_tree" in pd.DataFrame(sparsity_records).columns:
        df_sp = pd.DataFrame(sparsity_records)
        merge_key_sp = "sample_id" if ("sample_id" in df_merged.columns and "sample_id" in df_sp.columns) else "job_id"
        sp_cols = [c for c in ["sparsity_intra_tree", "sparsity_inter_tree", "mean_cable_distance"] if c in df_sp.columns]
        df_merged = pd.merge(df_merged, df_sp[[merge_key_sp] + sp_cols].drop_duplicates(subset=[merge_key_sp]), on=merge_key_sp, how="left")

    # Filter valid rows
    plot_df = df_merged.dropna(subset=["dspike_slope"]).copy() if "dspike_slope" in df_merged.columns else df_merged.copy()
    logger.info(f"Valid rows for dynamic slope analysis: {len(plot_df)}")

    # Setup 2x3 Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    def _plot_scatter_corr(ax, x_col, y_col, title, x_label, y_label, color="teal"):
        sub = plot_df.dropna(subset=[x_col, y_col])
        if len(sub) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(title)
            return

        x = sub[x_col].values
        y = sub[y_col].values
        r, p = pearsonr(x, y)

        # Linear fit line
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        x_grid = np.linspace(np.min(x), np.max(x), 100)
        y_grid = model.predict(x_grid.reshape(-1, 1))

        ax.scatter(x, y, alpha=0.6, color=color, edgecolors="k", linewidth=0.5, s=35)
        ax.plot(x_grid, y_grid, color="crimson", linestyle="--", linewidth=1.8, label=f"Fit (R² = {r**2:.3f})")

        ax.set_title(f"{title}\n$r = {r:+.3f}$ ($p = {p:.1e}$)", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

    # 1. dSpike slope vs Accuracy (The APCCAS Key Claim)
    if "accuracy" in plot_df.columns:
        _plot_scatter_corr(
            axes[0], "dspike_slope", "accuracy",
            "(a) dSpike Slope vs Accuracy (APCCAS Target)",
            "dSpike Slope (R_PC1_res / R_all)", "Accuracy", color="#1f77b4"
        )
        # 2. bAP slope vs Accuracy
        _plot_scatter_corr(
            axes[1], "bap_slope", "accuracy",
            "(b) bAP Slope vs Accuracy",
            "bAP Slope", "Accuracy", color="#2ca02c"
        )
        # 3. dSpike rate vs Accuracy
        _plot_scatter_corr(
            axes[2], "dspike_rate", "accuracy",
            "(c) dSpike Firing Rate vs Accuracy",
            "dSpike Rate [Hz]", "Accuracy", color="#ff7f0e"
        )
        # 4. bAP rate vs Accuracy
        _plot_scatter_corr(
            axes[3], "bap_rate", "accuracy",
            "(d) bAP Firing Rate vs Accuracy",
            "bAP Rate [Hz]", "Accuracy", color="#d62728"
        )
        # 5. Mean Residual ER vs Accuracy
        _plot_scatter_corr(
            axes[4], "mean_dspike_er_res", "accuracy",
            "(e) Mean Residual ER (dSpike) vs Accuracy",
            "Mean R(PC1 Excluded)", "Accuracy", color="#9467bd"
        )
    else:
        for idx in range(5):
            axes[idx].text(0.5, 0.5, "No Accuracy Data", ha="center", va="center")

    # 6. Inter-branch Sparsity vs dSpike slope (Chain: Sparsity -> Dynamics)
    if "sparsity_inter_tree" in plot_df.columns:
        _plot_scatter_corr(
            axes[5], "sparsity_inter_tree", "dspike_slope",
            "(f) Inter-Branch Sparsity vs dSpike Slope",
            "S_inter (Tree Backtrack Distance) [µm]", "dSpike Slope", color="#8c564b"
        )
    else:
        axes[5].text(0.5, 0.5, "No Sparsity Data", ha="center", va="center")

    plt.tight_layout()

    # Determine output path
    fig_dir = cfg.get("figure_dir", "figure")
    if not os.path.isabs(fig_dir):
        fig_dir = os.path.join(orig_cwd, fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    out_path = cfg.get("output_path", None)
    if not out_path:
        out_path = os.path.join(fig_dir, "dynamics_overview.png")
    elif not os.path.isabs(out_path):
        out_path = os.path.join(orig_cwd, out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Saved dynamics overview plot to: {out_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="conf", config_name="dynamics", version_base=None)
def main(cfg: DictConfig):
    if cfg.mode == "compute":
        run_compute(cfg)
    elif cfg.mode == "plot":
        run_plot(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'compute' or 'plot'.")


if __name__ == "__main__":
    main()
