"""Dynamics analysis of raw simulation traces: Effective Rank and spike kinetics.

Quantifies the dynamic state of single-neuron reservoir from raw waveforms:
1. Detects bAPs (global soma-originated spikes) and dSpikes (localized dendritic spikes).
2. Decomposes spatio-temporal calcium dynamics into global wave (PC1) and local residual.
3. Computes local Effective Rank (Roy & Vetterli 2007) around spike events.
4. Fits dSpike and bAP slopes (R(PC1 Excluded) vs R(All PCs)) to test APCCAS findings.

Usage:
    # Compute on cluster / local machines:
    uv run python dynamics.py mode=compute 'run_dir=multirun/2026-09-30/local-sweep-caracal/*' n_jobs=12

    Every mode writes its outputs (dynamics.json, events.npz, figures) into the
    Hydra run directory created for that invocation, so no run overwrites another.

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
from joblib import Parallel, delayed
from hydra.core.hydra_config import HydraConfig
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
    merge_ms: float = 0.0,
):
    """Detect bAP and dSpike timestamps using vectorized NumPy operations.

    vm: shape (T, N_compartments)
    t: shape (T,)
    v_threshold: spike threshold in mV
    exclusion_window_ms: window around bAP to exclude localized spikes
    merge_ms: a dSpike detection less than this after the previous one is taken
        as the same dSpike, keeping the first time step. One dSpike crosses the
        threshold in neighbouring compartments a few time steps apart, and each
        of those steps would otherwise count as a separate event.

    Returns the bAP indices, the dSpike indices, and the number of dSpike
    detections before merging.
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

    num_unmerged = len(dspike_indices)
    merge_samples = int(np.round(merge_ms / dt))
    if merge_samples > 0 and len(dspike_indices) > 1:
        keep = np.r_[True, np.diff(dspike_indices) >= merge_samples]
        dspike_indices = dspike_indices[keep]

    return bap_indices, dspike_indices, num_unmerged


# ==============================================================================
# 2. Single Run Processing
# ==============================================================================

def process_single_run(
    job_dir: str,
    window_ms: float = 5.0,
    v_threshold: float = -30.0,
    exclusion_window_ms: float = 10.0,
    test_only: bool = True,
    merge_ms: float = 0.0,
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

    # Somatic firing rate as the simulation wrote it: somatic spikes over all
    # training and test trials divided by their total duration.
    # Runs cut short leave the file empty; they get None and drop out of the rates.
    rate_path = os.path.join(job_dir, "data", "firing_rate.txt")
    rate_text = open(rate_path).read().strip() if os.path.exists(rate_path) else ""
    meta["soma_rate"] = float(rate_text) if rate_text else None

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
    # Per-event raw values: (kind, trial, t_ms, er_all, er_pc1_excluded); kind 0 = bAP, 1 = dSpike
    events = []
    total_duration_s = 0.0
    dspike_count_unmerged = 0

    for bf in buffers_to_process:
        trial = int(os.path.basename(bf)[len("buffer"):-len(".npz")])
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
        bap_idxs, dspike_idxs, num_unmerged = get_spike_indices(
            vm, t, v_threshold=v_threshold, exclusion_window_ms=exclusion_window_ms,
            merge_ms=merge_ms,
        )
        dspike_count_unmerged += num_unmerged

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
            events.append((0, trial, float(t[idx]), er_orig, er_res))

        # Effective Rank for dSpike
        for idx in dspike_idxs:
            if idx < half_w or idx > T_len - half_w:
                continue
            w_slice = slice(idx - half_w, idx + half_w)
            er_orig = calculate_effective_rank(ca[w_slice, :])
            er_res = calculate_effective_rank(ca_res[w_slice, :])
            dspike_er_pairs.append((er_orig, er_res))
            events.append((1, trial, float(t[idx]), er_orig, er_res))

        del ca, vm, ca_proj, ca_res
        gc.collect()

    # Aggregate metrics
    meta["total_duration_s"] = total_duration_s
    meta["bap_count"] = len(bap_er_pairs)
    meta["dspike_count"] = len(dspike_er_pairs)
    meta["dspike_count_unmerged"] = dspike_count_unmerged
    meta["bap_rate"] = float(len(bap_er_pairs) / total_duration_s) if total_duration_s > 0 else 0.0
    meta["dspike_rate"] = float(len(dspike_er_pairs) / total_duration_s) if total_duration_s > 0 else 0.0

    # Mean Effective Rank around spike events; left undefined below two events, as in
    # the dynamics.json files the analyses so far were drawn from.
    if len(dspike_er_pairs) >= 2:
        meta["mean_dspike_er_orig"] = float(np.mean([p[0] for p in dspike_er_pairs]))
        meta["mean_dspike_er_res"] = float(np.mean([p[1] for p in dspike_er_pairs]))
    else:
        meta["mean_dspike_er_orig"] = np.nan
        meta["mean_dspike_er_res"] = np.nan

    if len(bap_er_pairs) >= 2:
        meta["mean_bap_er_orig"] = float(np.mean([p[0] for p in bap_er_pairs]))
        meta["mean_bap_er_res"] = float(np.mean([p[1] for p in bap_er_pairs]))
    else:
        meta["mean_bap_er_orig"] = np.nan
        meta["mean_bap_er_res"] = np.nan

    return meta, events


# ==============================================================================
# 3. Compute Mode
# ==============================================================================

def _save_events(valid, out_path):
    """Save every detected spike event with its Effective Rank pair, indexed by run."""
    run_dir, mu, sigma = [], [], []
    columns = {"run": [], "kind": [], "trial": [], "t_ms": [], "er_all": [], "er_pc1_excluded": []}
    for i, (meta, events) in enumerate(valid):
        run_dir.append(meta["run_dir"])
        mu.append(np.nan if meta.get("syn_loc_mean") is None else float(meta["syn_loc_mean"]))
        sigma.append(np.nan if meta.get("syn_loc_std") is None else float(meta["syn_loc_std"]))
        for kind, trial, t_ms, er_all, er_res in events:
            columns["run"].append(i)
            columns["kind"].append(kind)
            columns["trial"].append(trial)
            columns["t_ms"].append(t_ms)
            columns["er_all"].append(er_all)
            columns["er_pc1_excluded"].append(er_res)
    np.savez(
        out_path,
        run_dir=np.array(run_dir),
        syn_loc_mean=np.array(mu),
        syn_loc_std=np.array(sigma),
        run=np.array(columns["run"], dtype=np.int32),
        kind=np.array(columns["kind"], dtype=np.int8),
        trial=np.array(columns["trial"], dtype=np.int32),
        t_ms=np.array(columns["t_ms"]),
        er_all=np.array(columns["er_all"]),
        er_pc1_excluded=np.array(columns["er_pc1_excluded"]),
    )
    logger.info(f"Saved {len(columns['run'])} events from {len(run_dir)} runs to: {out_path}")


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
            merge_ms=cfg.get("merge_ms", 0.0),
        )
        for jd in job_dirs
    )

    valid = [r for r in results if r is not None]
    valid_results = [meta for meta, _ in valid]
    out_dir = HydraConfig.get().runtime.output_dir
    _save_events(valid, os.path.join(out_dir, "events.npz"))

    out_file = os.path.join(out_dir, "dynamics.json")
    with open(out_file, "w") as f:
        json.dump(valid_results, f, indent=2)

    logger.info(f"Successfully computed dynamics for {len(valid_results)}/{len(job_dirs)} runs.")
    logger.info(f"Results saved to: {out_file}")









# ==============================================================================
# 4. Clusters Mode
# ==============================================================================

def run_clusters(cfg: DictConfig):
    """R(All PCs) vs R(PC1 Excluded) of bAP and dSpike events, one panel per run, ordered by mu."""
    orig_cwd = hydra.utils.get_original_cwd()
    paths = cfg.events_path
    if paths is None:
        raise ValueError("events_path must be specified for mode=clusters")
    if isinstance(paths, str):
        paths = [paths]

    runs = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(orig_cwd, p)
        with np.load(p) as ev:
            for i in range(len(ev["run_dir"])):
                sel = ev["run"] == i
                runs.append({
                    "mu": float(ev["syn_loc_mean"][i]),
                    "sigma": float(ev["syn_loc_std"][i]),
                    "kind": ev["kind"][sel],
                    "er_all": ev["er_all"][sel],
                    "er_res": ev["er_pc1_excluded"][sel],
                })

    runs = [r for r in runs if r["sigma"] <= cfg.sigma_max]
    runs.sort(key=lambda r: r["mu"])
    if not runs:
        raise ValueError(f"No runs with sigma <= {cfg.sigma_max}")
    logger.info(f"Plotting {len(runs)} runs with sigma <= {cfg.sigma_max}")

    ncols = cfg.n_cols
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False)
    styles = [(0, "bAP", "tab:blue"), (1, "dSpike", "tab:orange")]

    for ax, r in zip(axes.flat, runs):
        for kind, label, color in styles:
            sel = r["kind"] == kind
            x, y = r["er_all"][sel], r["er_res"][sel]
            ax.scatter(x, y, s=6, alpha=0.4, color=color, edgecolors="none", label=label)
            if len(x) >= 2 and np.ptp(x) > 0:
                lr = LinearRegression().fit(x.reshape(-1, 1), y)
                x_grid = np.linspace(x.min(), x.max(), 2)
                ax.plot(x_grid, lr.predict(x_grid.reshape(-1, 1)), color=color, linewidth=1.5)
        n_bap = int(np.sum(r["kind"] == 0))
        n_ds = int(np.sum(r["kind"] == 1))
        ax.set_title(f"μ = {r['mu']:.0f} µm, σ = {r['sigma']:.0f} µm\nbAP {n_bap}, dSpike {n_ds}", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("R(All PCs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("R(PC1 Excluded)")
    axes[0, 0].legend(loc="upper left", fontsize=8, markerscale=3)

    plt.tight_layout()
    out_path = os.path.join(HydraConfig.get().runtime.output_dir, "clusters.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved cluster plot to: {out_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="conf", config_name="dynamics", version_base=None)
def main(cfg: DictConfig):
    if cfg.mode == "compute":
        run_compute(cfg)
    elif cfg.mode == "clusters":
        run_clusters(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'compute' or 'clusters'.")


if __name__ == "__main__":
    main()
