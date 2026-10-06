"""Separation property of the reservoir states and of the inputs that drive them.

For every run, the test trials are compared pairwise at matching times within the
trial. Pairs of the same class differ only by the input jitter (and by the state
carried over from the preceding trial); pairs of different classes differ in their
template. The squared Euclidean distance between two trials at time t is computed
for

- the calcium state as recorded ("state_raw"),
- the calcium state with each recording site standardised over the run ("state_z"),
- the input spike trains filtered with an exponential kernel ("input").

Per run, D_between(t) / D_within(t) are the mean distances over pairs of different /
same class, and SR = mean_t D_between / mean_t D_within.

Usage:
    # Compute on the machine that holds the runs:
    uv run python separation.py mode=compute 'run_dir=multirun/2026-09-30/local-sweep-caracal/*' n_jobs=12

    # Plot across machines, joined with accuracy and sparsity:
    uv run python separation.py mode=plot \\
        'summary_path=[data/separation/caracal.npz,data/separation/oncilla.npz]' \\
        'accuracy_path=[data/2026-09-30_sweep200/caracal/results.json,data/2026-09-30_sweep200/oncilla/results.json]' \\
        'sparsity_path=[data/2026-09-30_sweep200/caracal/sparsity.json,data/2026-09-30_sweep200/oncilla/sparsity.json]'
"""

import glob
import json
import logging
import os
import zipfile

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

VARIANTS = ("state_raw", "state_z", "input")


# ==============================================================================
# 1. Distances
# ==============================================================================

def filter_spike_trains(spike_times, spike_neurons, grid, num_channels, tau_ms):
    """Exponentially filtered spike trains sampled on `grid` (ms, relative to trial start).

    Each spike adds exp(-(t - s) / tau) for t >= s. The recursion over the grid is
    exact: a spike between two grid points enters at the first grid point after it
    with the decay it has already undergone.
    """
    num_steps = len(grid)
    dt = float(grid[1] - grid[0])
    decay = np.exp(-dt / tau_ms)
    arrival = np.zeros((num_steps, num_channels))
    k = np.searchsorted(grid, spike_times, side="left")
    inside = k < num_steps
    np.add.at(arrival, (k[inside], spike_neurons[inside].astype(int)),
              np.exp(-(grid[k[inside]] - spike_times[inside]) / tau_ms))
    filtered = np.empty_like(arrival)
    filtered[0] = arrival[0]
    for i in range(1, num_steps):
        filtered[i] = filtered[i - 1] * decay + arrival[i]
    return filtered


def standardise_sites(states):
    """Standardise each site over all trials and times of the run; constant sites become 0."""
    flat = states.reshape(-1, states.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    return np.where(std > 0, (states - mean) / std_safe, 0.0)


def pair_distances(trajectories):
    """Squared distance between every pair of trials at every time.

    trajectories: (num_trials, T, N). Returns (T, num_pairs) over pairs i < j.
    """
    squared_norm = np.einsum("itk,itk->ti", trajectories, trajectories)
    gram = np.einsum("itk,jtk->tij", trajectories, trajectories)
    full = squared_norm[:, :, None] + squared_norm[:, None, :] - 2.0 * gram
    i, j = np.triu_indices(trajectories.shape[0], k=1)
    return np.maximum(full[:, i, j], 0.0)


def separation_curves(distances, same_class):
    """D_between(t), D_within(t) and SR from pairwise distances (T, num_pairs)."""
    d_between = distances[:, ~same_class].mean(axis=1)
    d_within = distances[:, same_class].mean(axis=1)
    return d_between, d_within, float(d_between.mean() / d_within.mean())


# ==============================================================================
# 2. Single Run
# ==============================================================================

def load_test_trials(job_dir, step, tau_ms):
    """Calcium states and filtered inputs of the test trials, subsampled every `step` samples."""
    config = OmegaConf.load(os.path.join(job_dir, ".hydra", "config.yaml"))
    num_channels = int(config.task.exc_num_syn)

    states, inputs, labels, data_idx = [], [], [], []
    grid = None
    for path in sorted(glob.glob(os.path.join(job_dir, "data", "buffer*.npz"))):
        with np.load(path, allow_pickle=True) as npz:
            if str(npz["mode"]) != "test":
                continue
            t = npz["t_rec"][::step]
            calcium = npz["variables"][0][::step]
            spikes = npz["input"].item()
            label = int(npz["TrueLabel"])
            index = int(npz["data_idx"])
        t_rel = t - t[0]
        if grid is None:
            grid = t_rel
        states.append(calcium.astype(np.float64))
        inputs.append(filter_spike_trains(
            np.asarray(spikes["spike_times"]) - t[0], np.asarray(spikes["spike_neurons"]),
            grid, num_channels, tau_ms))
        labels.append(label)
        data_idx.append(index)

    meta = {
        "syn_loc_mean": float(config.syn_loc_mean),
        "syn_loc_std": float(config.syn_loc_std),
        "sample_id": int(config.sample_id) if config.get("sample_id") is not None else -1,
    }
    return np.stack(states), np.stack(inputs), np.array(labels), np.array(data_idx), grid, meta


def process_single_run(job_dir, step, tau_ms, pairs_dir):
    """Pairwise distances and separation curves of one run. Returns None if unreadable."""
    job_dir = os.path.abspath(job_dir)
    try:
        states, inputs, labels, data_idx, grid, meta = load_test_trials(job_dir, step, tau_ms)
    except (zipfile.BadZipFile, ValueError, KeyError, OSError, EOFError) as e:
        logger.warning(f"Skipping {job_dir}: {e}")
        return None
    if len(labels) < 2:
        logger.warning(f"Skipping {job_dir}: {len(labels)} test trials")
        return None

    # Pairs follow simulation order, so the preceding trial of each one stays recoverable.
    order = np.argsort(data_idx)
    states, inputs, labels, data_idx = states[order], inputs[order], labels[order], data_idx[order]
    i, j = np.triu_indices(len(labels), k=1)
    same_class = labels[i] == labels[j]

    trajectories = {"state_raw": states, "state_z": standardise_sites(states), "input": inputs}
    result = {"run_dir": job_dir, "labels": labels, "data_idx": data_idx, "same_class": same_class,
              "grid_ms": grid, **meta}
    pairs = {}
    for name in VARIANTS:
        distances = pair_distances(trajectories[name])
        d_between, d_within, sr = separation_curves(distances, same_class)
        result[f"{name}_d_between_t"] = d_between
        result[f"{name}_d_within_t"] = d_within
        result[f"{name}_sr"] = sr
        result[f"{name}_pair_mean"] = distances.mean(axis=0)
        pairs[name] = distances.astype(np.float32)

    np.savez(os.path.join(pairs_dir, f"{os.path.basename(job_dir)}.npz"),
             run_dir=job_dir, pair_i=i, pair_j=j, same_class=same_class, labels=labels,
             data_idx=data_idx, grid_ms=grid, **pairs)
    logger.info(f"{job_dir}: SR raw {result['state_raw_sr']:.3f}, "
                f"z {result['state_z_sr']:.3f}, input {result['input_sr']:.3f}")
    return result


# ==============================================================================
# 3. Compute Mode
# ==============================================================================

def run_compute(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    pattern = cfg.run_dir if os.path.isabs(cfg.run_dir) else os.path.join(orig_cwd, cfg.run_dir)
    job_dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not job_dirs:
        raise FileNotFoundError(f"No run directories matched: {pattern}")

    out_dir = HydraConfig.get().runtime.output_dir
    pairs_dir = os.path.join(out_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)
    logger.info(f"Computing separation for {len(job_dirs)} runs")

    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
        delayed(process_single_run)(jd, cfg.step, cfg.tau_ms, pairs_dir) for jd in job_dirs)
    results = [r for r in results if r is not None]

    # Runs cut short hold fewer test trials; their pairs do not line up with the others.
    counts = [len(r["labels"]) for r in results]
    expected = max(set(counts), key=counts.count)
    for r in results:
        if len(r["labels"]) != expected:
            logger.warning(f"Excluding {r['run_dir']} from the summary: "
                           f"{len(r['labels'])} test trials, expected {expected}")
    results = [r for r in results if len(r["labels"]) == expected]

    # One summary file for all runs: small enough to move between machines, and
    # holding everything the plots need except the per-time pair distances.
    summary = {
        "run_dir": np.array([r["run_dir"] for r in results]),
        "syn_loc_mean": np.array([r["syn_loc_mean"] for r in results]),
        "syn_loc_std": np.array([r["syn_loc_std"] for r in results]),
        "sample_id": np.array([r["sample_id"] for r in results]),
        "same_class": np.stack([r["same_class"] for r in results]),
        "labels": np.stack([r["labels"] for r in results]),
        "grid_ms": results[0]["grid_ms"],
    }
    for name in VARIANTS:
        for key in ("d_between_t", "d_within_t", "sr", "pair_mean"):
            summary[f"{name}_{key}"] = np.array([r[f"{name}_{key}"] for r in results])
    np.savez(os.path.join(out_dir, "summary.npz"), **summary)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump([{"run_dir": r["run_dir"], "syn_loc_mean": r["syn_loc_mean"],
                    "syn_loc_std": r["syn_loc_std"], "sample_id": r["sample_id"],
                    **{f"{n}_sr": r[f"{n}_sr"] for n in VARIANTS}} for r in results], f, indent=2)
    logger.info(f"Computed {len(results)}/{len(job_dirs)} runs; saved to {out_dir}")


# ==============================================================================
# 4. Plot Mode
# ==============================================================================

def _as_list(paths):
    if paths is None:
        return []
    return [paths] if isinstance(paths, str) else list(paths)


def _resolve(path, orig_cwd):
    return path if os.path.isabs(path) else os.path.join(orig_cwd, path)


def load_summaries(paths, orig_cwd):
    """Concatenate summary.npz files from several machines along the run axis."""
    parts = []
    for p in _as_list(paths):
        with np.load(_resolve(p, orig_cwd)) as npz:
            parts.append({k: npz[k] for k in npz.files})
    merged = {k: (parts[0][k] if k == "grid_ms" else np.concatenate([p[k] for p in parts]))
              for k in parts[0]}
    return merged


def load_accuracy(paths, orig_cwd):
    """run_dir -> marginalised test accuracy (first subset size) from readout_subset results."""
    accuracy = {}
    for p in _as_list(paths):
        with open(_resolve(p, orig_cwd)) as f:
            for run in json.load(f)["runs"]:
                accuracy[run["source_run"]] = run["sizes"][0]["test_accuracy_mean"]
    return accuracy


def load_sparsity(paths, orig_cwd):
    """run_dir -> (S_intra, S_inter) from sparsity.py results."""
    sparsity = {}
    for p in _as_list(paths):
        with open(_resolve(p, orig_cwd)) as f:
            for rec in json.load(f):
                sparsity[rec["run_dir"]] = (rec["intra"], rec["inter"])
    return sparsity


def r_squared(columns, y, quadratic=False):
    """Ordinary least squares R² of y on the given columns (with intercept)."""
    x = np.column_stack(columns)
    if quadratic:
        a, b = x[:, 0], x[:, 1]
        x = np.column_stack([a, b, a ** 2, b ** 2, a * b] + [x[:, k] for k in range(2, x.shape[1])])
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(1.0 - residual @ residual / np.sum((y - y.mean()) ** 2))


def local_fluctuation(input_dist, state_dist):
    """Pearson r and log-log slope of state distance on input distance over pairs."""
    r, _ = pearsonr(input_dist, state_dist)
    slope = np.polyfit(np.log(input_dist), np.log(state_dist), 1)[0]
    return float(r), float(slope)


def _scatter(ax, x, y, xlabel, ylabel, color):
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=18, alpha=0.7, color=color, edgecolors="k", linewidth=0.3)
    ax.set_title(f"r = {r:+.3f} (p = {p:.1e})", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)


def run_plot(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    out_dir = HydraConfig.get().runtime.output_dir
    s = load_summaries(cfg.summary_path, orig_cwd)
    accuracy = load_accuracy(cfg.accuracy_path, orig_cwd)
    sparsity = load_sparsity(cfg.sparsity_path, orig_cwd)

    keep = np.array([d in accuracy and d in sparsity for d in s["run_dir"]])
    logger.info(f"Runs: {len(keep)} in summaries, {keep.sum()} joined with accuracy and sparsity")
    s = {k: (v if k == "grid_ms" else v[keep]) for k, v in s.items()}
    acc = np.array([accuracy[d] for d in s["run_dir"]])
    s_intra = np.array([sparsity[d][0] for d in s["run_dir"]])
    s_inter = np.array([sparsity[d][1] for d in s["run_dir"]])
    mu, sigma = s["syn_loc_mean"], s["syn_loc_std"]
    sr_raw, sr_z, sr_in = s["state_raw_sr"], s["state_z_sr"], s["input_sr"]

    # Local fluctuation within each input-distance level, per run.
    fluct = {}
    for name in ("state_raw", "state_z"):
        for level, select in (("within", True), ("between", False)):
            stats = [local_fluctuation(s["input_pair_mean"][k][s["same_class"][k] == select],
                                       s[f"{name}_pair_mean"][k][s["same_class"][k] == select])
                     for k in range(len(acc))]
            fluct[f"{name}_{level}_r"] = np.array([st[0] for st in stats])
            fluct[f"{name}_{level}_slope"] = np.array([st[1] for st in stats])

    # Input distances relative to each run's mean between-class distance.
    rel_input = s["input_pair_mean"] / s["input_pair_mean"][~s["same_class"]].reshape(len(acc), -1).mean(axis=1)[:, None]
    cv = {level: np.array([np.std(s["input_pair_mean"][k][s["same_class"][k] == sel])
                           / np.mean(s["input_pair_mean"][k][s["same_class"][k] == sel])
                           for k in range(len(acc))])
          for level, sel in (("within", True), ("between", False))}

    stats = {
        "num_runs": int(len(acc)),
        "r_squared": {
            "acc_on_sr_raw": r_squared([sr_raw], acc),
            "acc_on_sr_z": r_squared([sr_z], acc),
            "acc_on_sr_raw_over_input": r_squared([sr_raw / sr_in], acc),
            "sr_raw_on_sparsity_linear": r_squared([s_intra, s_inter], sr_raw),
            "sr_raw_on_mu_sigma_linear": r_squared([mu, sigma], sr_raw),
            "sr_raw_on_sparsity_quadratic": r_squared([s_intra, s_inter], sr_raw, quadratic=True),
            "sr_raw_on_mu_sigma_quadratic": r_squared([mu, sigma], sr_raw, quadratic=True),
            "acc_on_sparsity_linear": r_squared([s_intra, s_inter], acc),
            "acc_on_sparsity_linear_plus_sr_raw": r_squared([s_intra, s_inter, sr_raw], acc),
            "acc_on_sparsity_quadratic": r_squared([s_intra, s_inter], acc, quadratic=True),
            "acc_on_sparsity_quadratic_plus_sr_raw": r_squared([s_intra, s_inter, sr_raw], acc, quadratic=True),
        },
        "input_relative_spread": {level: {"median": float(np.median(v)), "max": float(np.max(v))}
                                  for level, v in cv.items()},
        "input_sr": {"min": float(sr_in.min()), "median": float(np.median(sr_in)), "max": float(sr_in.max())},
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(stats, f, indent=2)
    np.savez(os.path.join(out_dir, "curves.npz"), run_dir=s["run_dir"], accuracy=acc,
             s_intra=s_intra, s_inter=s_inter, **fluct)
    logger.info(json.dumps(stats, indent=2))

    # (1) overview: SR against accuracy and sparsity
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _scatter(axes[0, 0], sr_raw, acc, "SR (calcium)", "Accuracy", "#4c72b0")
    _scatter(axes[0, 1], sr_z, acc, "SR (calcium, standardised per site)", "Accuracy", "#55a868")
    _scatter(axes[0, 2], sr_raw / sr_in, acc, "SR (calcium) / SR (input)", "Accuracy", "#8172b2")
    _scatter(axes[1, 0], s_intra, sr_raw, "S_intra [µm]", "SR (calcium)", "#c44e52")
    _scatter(axes[1, 1], s_inter, sr_raw, "S_inter [µm]", "SR (calcium)", "#dd8452")
    axes[1, 2].hist(sr_in, bins=30, color="gray", edgecolor="k")
    axes[1, 2].set_xlabel("SR (input)")
    axes[1, 2].set_ylabel("Runs")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "separation_overview.png"), dpi=200)
    plt.close(fig)

    # (2) D_between and D_within over the trial, runs with small sigma ordered by mu
    grid = s["grid_ms"]
    small = np.where(sigma <= cfg.sigma_max)[0]
    small = small[np.argsort(mu[small])]
    ncols = cfg.n_cols
    nrows = int(np.ceil(len(small) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), sharex=True, squeeze=False)
    for ax, k in zip(axes.flat, small):
        ax.plot(grid, s["state_raw_d_between_t"][k], color="tab:red", lw=1, label="between")
        ax.plot(grid, s["state_raw_d_within_t"][k], color="tab:blue", lw=1, label="within")
        ax.set_title(f"μ = {mu[k]:.0f}, σ = {sigma[k]:.0f} µm\nSR = {sr_raw[k]:.2f}, Acc = {acc[k]:.2f}", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
    for ax in axes.flat[len(small):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time in trial [ms]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Squared distance")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "separation_time.png"), dpi=200)
    plt.close(fig)

    # (3) input distances: are they concentrated on two levels?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, rel_input.max() * 1.02, 120)
    axes[0].hist(rel_input[s["same_class"]], bins=bins, color="tab:blue", alpha=0.7, label="same class")
    axes[0].hist(rel_input[~s["same_class"]], bins=bins, color="tab:red", alpha=0.7, label="different class")
    axes[0].set_xlabel("Input distance / run mean of different-class pairs")
    axes[0].set_ylabel("Pairs (all runs)")
    axes[0].legend()
    axes[1].hist(cv["within"], bins=30, color="tab:blue", alpha=0.7, label="same class")
    axes[1].hist(cv["between"], bins=30, color="tab:red", alpha=0.7, label="different class")
    axes[1].set_xlabel("Relative spread of input distance within a run (SD / mean)")
    axes[1].set_ylabel("Runs")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "separation_input.png"), dpi=200)
    plt.close(fig)

    # (4) does the fluctuation of the input distance show up in the state distance?
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    k = int(np.argsort(acc)[len(acc) // 2])  # median-accuracy run as the example
    same = s["same_class"][k]
    axes[0, 0].scatter(s["input_pair_mean"][k][~same], s["state_raw_pair_mean"][k][~same], s=6, alpha=0.4, color="tab:red", label="different class")
    axes[0, 0].scatter(s["input_pair_mean"][k][same], s["state_raw_pair_mean"][k][same], s=10, alpha=0.8, color="tab:blue", label="same class")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Input distance (pair, trial mean)")
    axes[0, 0].set_ylabel("Calcium distance (pair, trial mean)")
    axes[0, 0].set_title(f"Example: μ = {mu[k]:.0f}, σ = {sigma[k]:.0f} µm, Acc = {acc[k]:.2f}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].hist(fluct["state_raw_within_r"], bins=30, color="tab:blue", alpha=0.7, label="same class")
    axes[0, 1].hist(fluct["state_raw_between_r"], bins=30, color="tab:red", alpha=0.7, label="different class")
    axes[0, 1].set_xlabel("r (input distance, calcium distance) over pairs")
    axes[0, 1].set_ylabel("Runs")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].hist(fluct["state_raw_within_slope"], bins=30, color="tab:blue", alpha=0.7, label="same class")
    axes[0, 2].hist(fluct["state_raw_between_slope"], bins=30, color="tab:red", alpha=0.7, label="different class")
    axes[0, 2].set_xlabel("log-log slope (calcium on input distance)")
    axes[0, 2].set_ylabel("Runs")
    axes[0, 2].legend(fontsize=8)
    _scatter(axes[1, 0], fluct["state_raw_within_slope"], acc, "log-log slope, same class", "Accuracy", "tab:blue")
    _scatter(axes[1, 1], fluct["state_raw_between_slope"], acc, "log-log slope, different class", "Accuracy", "tab:red")
    _scatter(axes[1, 2], s_inter, fluct["state_raw_within_slope"], "S_inter [µm]", "log-log slope, same class", "#dd8452")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "separation_fluctuation.png"), dpi=200)
    plt.close(fig)

    # (5) per-time pair distances for selected runs (fetched pairs/*.npz)
    pair_files = _as_list(cfg.pairs_path)
    if pair_files:
        fig, axes = plt.subplots(1, len(pair_files), figsize=(4.5 * len(pair_files), 4), squeeze=False)
        for ax, p in zip(axes.flat, pair_files):
            with np.load(_resolve(p, orig_cwd)) as npz:
                same = npz["same_class"]
                x_in = npz["input"]
                y_st = npz["state_raw"]
                run = str(npz["run_dir"])
            k = int(np.where(s["run_dir"] == run)[0][0]) if run in s["run_dir"] else None
            for select, color, label in ((False, "tab:red", "different class"), (True, "tab:blue", "same class")):
                xs, ys = x_in[:, same == select].ravel(), y_st[:, same == select].ravel()
                ok = (xs > 0) & (ys > 0)
                ax.hexbin(xs[ok], ys[ok], xscale="log", yscale="log", gridsize=60, bins="log",
                          cmap="Reds" if not select else "Blues", alpha=0.6, mincnt=1)
                ax.plot([], [], color=color, label=label)
            title = os.path.basename(run) if k is None else f"μ = {mu[k]:.0f}, σ = {sigma[k]:.0f} µm, Acc = {acc[k]:.2f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Input distance at time t")
            ax.set_ylabel("Calcium distance at time t")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "separation_time_pairs.png"), dpi=200)
        plt.close(fig)

    logger.info(f"Saved figures to {out_dir}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

@hydra.main(config_path="conf", config_name="separation", version_base=None)
def main(cfg: DictConfig):
    if cfg.mode == "compute":
        run_compute(cfg)
    elif cfg.mode == "plot":
        run_plot(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'compute' or 'plot'.")


if __name__ == "__main__":
    main()
