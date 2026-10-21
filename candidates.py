"""Comparison of dynamics candidates: how much of the sparsity effect on accuracy each captures.

Saved outputs of sparsity.py, readout_subset.py, separation.py and dynamics.py are
joined by run_dir; nothing is recomputed. Candidates:
1. SR
2. SR (standardised per site)
3. Soma firing rate
4. bAP firing rate
5. dSpike firing rate
6. Mean dSpike ER (All PCs)
7. Mean bAP ER (All PCs)
8. bAP rate x mean bAP ER
9. dSpike rate x mean dSpike ER
10. Within-class consistency of bAP-related compartment spikes
11. Within-class consistency of the other compartment spikes

A run without a value for a candidate (e.g. ER with fewer than 2 events) is left out
of that candidate only.

Usage:
    # Run with default inputs configured in conf/candidates.yaml:
    uv run python candidates.py

    # Override paths if needed:
    uv run python candidates.py \\
        'sparsity_path=[data/2026-09-30_sweep200/caracal/sparsity.json,data/2026-09-30_sweep200/oncilla/sparsity.json]'
"""

import json
import logging
import os

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# Set matplotlib fonts and styles
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


# ==============================================================================
# 1. Data Loading and Merging
# ==============================================================================

def _as_list(paths):
    if paths is None:
        return []
    return list(paths) if isinstance(paths, (list, ListConfig)) else [paths]


def _resolve(path, orig_cwd):
    return path if os.path.isabs(path) else os.path.join(orig_cwd, path)


def _value(x):
    """float, or NaN when the saved value is missing (None / NaN)."""
    return float(x) if x is not None and not np.isnan(x) else np.nan


# Read from any of the dynamics.py outputs listed in dynamics_path (rates.json,
# effective_rank.json, consistency.json), joined by run_dir.
DYNAMICS_KEYS = ("soma_rate", "bap_rate", "dspike_rate", "mean_dspike_er_orig", "mean_bap_er_orig",
                 "gamma_within_bap", "gamma_within_dsp")


def load_merged_data(cfg, orig_cwd):
    """Runs with sparsity and accuracy; candidate values missing for a run are NaN."""
    runs = {}

    for p in _as_list(cfg.sparsity_path):
        with open(_resolve(p, orig_cwd)) as f:
            for item in json.load(f):
                runs.setdefault(item["run_dir"], {})["sparsity"] = {
                    "intra": float(item["intra"]),
                    "inter": float(item["inter"]),
                    "syn_loc_mean": float(item["syn_loc_mean"]),
                    "syn_loc_std": float(item["syn_loc_std"]),
                }

    for p in _as_list(cfg.accuracy_path):
        with open(_resolve(p, orig_cwd)) as f:
            for r in json.load(f)["runs"]:
                rd = r["source_run"]
                for s in r["sizes"]:
                    if s["num_readout_sites"] == cfg.subset_size and rd in runs:
                        runs[rd]["accuracy"] = float(s["test_accuracy_mean"])

    for p in _as_list(cfg.separation_path):
        with np.load(_resolve(p, orig_cwd)) as npz:
            for i, rd in enumerate(npz["run_dir"]):
                rd = str(rd)
                if rd in runs:
                    runs[rd]["sr_raw"] = float(npz["state_raw_sr"][i])
                    runs[rd]["sr_z"] = float(npz["state_z_sr"][i])

    for p in _as_list(cfg.dynamics_path):
        with open(_resolve(p, orig_cwd)) as f:
            for item in json.load(f):
                rd = item["run_dir"]
                if rd in runs:
                    for k in DYNAMICS_KEYS:
                        if k in item:
                            runs[rd][k] = _value(item[k])

    valid_runs = [d for d in runs.values() if "accuracy" in d and "sparsity" in d]
    for d in valid_runs:
        for k in ("sr_raw", "sr_z") + DYNAMICS_KEYS:
            d.setdefault(k, np.nan)
    logger.info(f"Runs with sparsity and accuracy (k = {cfg.subset_size}): {len(valid_runs)}")
    return valid_runs


# ==============================================================================
# 2. Regression & Variance Decomposition
# ==============================================================================

def fit_ols_r2(X_cols, y):
    """Compute OLS R² with intercept for design matrix columns X_cols."""
    X = np.column_stack(X_cols)
    design = np.column_stack([np.ones(len(y)), X])
    coef, residuals, rank, s = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    res = y - pred
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(res ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return r2, coef, pred, res


def build_sparsity_design(s_intra, s_inter, quadratic=False):
    if quadratic:
        return [s_intra, s_inter, s_intra ** 2, s_inter ** 2, s_intra * s_inter]
    else:
        return [s_intra, s_inter]


def build_candidate_design(cand, quadratic=False):
    if quadratic:
        return [cand, cand ** 2]
    else:
        return [cand]


def decompose_sparsity_on_candidate(s_intra, s_inter, cand, quadratic=False):
    """Decompose R² of candidate explained by sparsity."""
    if quadratic:
        cols_full = [s_intra, s_inter, s_intra ** 2, s_inter ** 2, s_intra * s_inter]
        cols_intra = [s_intra, s_intra ** 2]
        cols_inter = [s_inter, s_inter ** 2]
    else:
        cols_full = [s_intra, s_inter]
        cols_intra = [s_intra]
        cols_inter = [s_inter]

    r2_full, _, _, _ = fit_ols_r2(cols_full, cand)
    r2_intra_m, _, _, _ = fit_ols_r2(cols_intra, cand)
    r2_inter_m, _, _, _ = fit_ols_r2(cols_inter, cand)

    unique_intra = float(r2_full - r2_inter_m)
    unique_inter = float(r2_full - r2_intra_m)
    common = float(r2_intra_m + r2_inter_m - r2_full)
    unexplained = float(1.0 - r2_full)

    return {
        "r2_full": r2_full,
        "unique_intra": unique_intra,
        "unique_inter": unique_inter,
        "common": common,
        "unexplained": unexplained,
    }


def decompose_accuracy_r2(s_intra, s_inter, cand, accuracy, quadratic=False):
    """Four-way commonality analysis of accuracy variance.
    
    1. unique_sparsity: R²(both) - R²(cand)
    2. unique_cand: R²(both) - R²(sparsity)
    3. common: R²(sparsity) + R²(cand) - R²(both)
    4. unexplained: 1 - R²(both)
    """
    cols_sp = build_sparsity_design(s_intra, s_inter, quadratic=quadratic)
    cols_cand = build_candidate_design(cand, quadratic=quadratic)
    cols_both = cols_sp + cols_cand

    r2_sp, _, pred_sp, res_sp_acc = fit_ols_r2(cols_sp, accuracy)
    r2_cand, _, _, _ = fit_ols_r2(cols_cand, accuracy)
    r2_both, _, _, _ = fit_ols_r2(cols_both, accuracy)

    # For partial regression: candidate residual from sparsity
    _, _, _, res_sp_cand = fit_ols_r2(cols_sp, cand)

    unique_sp = float(r2_both - r2_cand)
    unique_cand = float(r2_both - r2_sp)
    common = float(r2_sp + r2_cand - r2_both)
    unexplained = float(1.0 - r2_both)

    return {
        "r2_sparsity": r2_sp,
        "r2_candidate": r2_cand,
        "r2_both": r2_both,
        "unique_sparsity": unique_sp,
        "unique_candidate": unique_cand,
        "common": common,
        "unexplained": unexplained,
        "res_sp_cand": res_sp_cand,
        "res_sp_acc": res_sp_acc,
    }


def adjusted_r2(r2, n, p):
    """Adjusted R² (Ezekiel): 1 - (1 - R²)(n - 1)/(n - p - 1), p excluding the intercept."""
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def _four_parts(r2_sp, r2_set, r2_both):
    return {
        "r2_sparsity": r2_sp, "r2_candidate": r2_set, "r2_both": r2_both,
        "unique_sparsity": r2_both - r2_set,
        "unique_candidate": r2_both - r2_sp,
        "common": r2_sp + r2_set - r2_both,
        "unexplained": 1.0 - r2_both,
    }


def decompose_accuracy_set(sp_cols, set_cols, accuracy):
    """Four-way partition of accuracy between sparsity and a set of candidates.

    Returned twice: from raw R², and from adjusted R² of the same three models, the
    estimate variation partitioning uses (Peres-Neto et al. 2006).
    """
    n = len(accuracy)
    fits = {}
    for name, cols in (("sp", sp_cols), ("set", set_cols), ("both", sp_cols + set_cols)):
        r2, _, _, _ = fit_ols_r2(cols, accuracy)
        fits[name] = (r2, adjusted_r2(r2, n, len(cols)))
    return {
        "raw": _four_parts(fits["sp"][0], fits["set"][0], fits["both"][0]),
        "adjusted": _four_parts(fits["sp"][1], fits["set"][1], fits["both"][1]),
    }


# ==============================================================================
# 3. Plot Helpers
# ==============================================================================

RAMPS = {"mu": "Blues", "sigma": "Oranges", "accuracy": "Purples"}

# S_intra goes with sigma (orange) and S_inter with mu (blue), as in the scatter
# panels; the four accuracy parts take hues that appear nowhere else.
SPARSITY_PARTS = (
    ("unique_intra", r"$S_{\mathrm{intra}}$ unique", "#dd8452"),
    ("unique_inter", r"$S_{\mathrm{inter}}$ unique", "#4c72b0"),
    ("common", "Common", "#9c755f"),
    ("unexplained", "Unexplained", "#cccccc"),
)
ACCURACY_PARTS = (
    ("unique_sparsity", "(1) Sparsity effect not captured by candidate", "#c49c94"),
    ("common", "(3) Sparsity effect captured by candidate", "#2a9d8f"),
    ("unique_candidate", "(2) Candidate, unrelated to sparsity", "#e9c46a"),
    ("unexplained", "(4) Unexplained", "#cccccc"),
)


def stacked_bars(ax, positions, decomps, parts, width, horizontal):
    """Stack the parts of each decomposition. A negative part (shared variance can
    fall below zero when predictors suppress each other) is drawn hatched on the
    other side of zero so that no segment overlaps another; the positive parts then
    add up to more than 1. Returns the lowest and highest extents."""
    pos_base = np.zeros(len(positions))
    neg_base = np.zeros(len(positions))
    for key, label, colour in parts:
        vals = np.array([d[key] for d in decomps], dtype=float)
        up = np.where(vals > 0, vals, 0.0)
        down = np.where(vals < 0, vals, 0.0)
        style = dict(edgecolor="k", linewidth=0.5, color=colour)
        if horizontal:
            ax.barh(positions, up, height=width, left=pos_base, label=label, **style)
            if np.any(down < 0):
                ax.barh(positions, down, height=width, left=neg_base, hatch="//", **style)
        else:
            ax.bar(positions, up, width, bottom=pos_base, label=label, **style)
            if np.any(down < 0):
                ax.bar(positions, down, width, bottom=neg_base, hatch="//", **style)
        pos_base += up
        neg_base += down
    return float(neg_base.min()), float(pos_base.max())

# ==============================================================================
# 4. Baseline Figure
# ==============================================================================

def plot_baseline_figure(s_intra, s_inter, mu, sigma, accuracy, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # (a) S_intra vs Accuracy
    r_intra, p_intra = pearsonr(s_intra, accuracy)
    axes[0, 0].scatter(s_intra, accuracy, s=24, color="#dd8452", edgecolors="k", linewidth=0.3)
    axes[0, 0].set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0, 0].set_ylabel("Accuracy", fontsize=11)
    axes[0, 0].set_title(f"(a) $S_{{\\mathrm{{intra}}}}$ vs Accuracy\n$r = {r_intra:+.3f}$ ($p = {p_intra:.1e}$)", fontsize=11)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # (b) S_inter vs Accuracy
    r_inter, p_inter = pearsonr(s_inter, accuracy)
    axes[0, 1].scatter(s_inter, accuracy, s=24, color="#4c72b0", edgecolors="k", linewidth=0.3)
    axes[0, 1].set_xlabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0, 1].set_ylabel("Accuracy", fontsize=11)
    axes[0, 1].set_title(f"(b) $S_{{\\mathrm{{inter}}}}$ vs Accuracy\n$r = {r_inter:+.3f}$ ($p = {p_inter:.1e}$)", fontsize=11)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # (c) S_intra x S_inter plane with Accuracy
    sc = axes[1, 0].scatter(s_intra, s_inter, s=28, c=accuracy, cmap=RAMPS["accuracy"], edgecolors="k", linewidth=0.3)
    axes[1, 0].set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[1, 0].set_ylabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[1, 0].set_title("(c) Sparsity Plane coloured by Accuracy", fontsize=11)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    cbar = fig.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy", fontsize=10)

    # (d) R² decomposition of Accuracy on Sparsity (Linear & Quadratic)
    decomp_lin = decompose_sparsity_on_candidate(s_intra, s_inter, accuracy, quadratic=False)
    decomp_quad = decompose_sparsity_on_candidate(s_intra, s_inter, accuracy, quadratic=True)

    categories = ["Linear", "Quadratic"]
    ax = axes[1, 1]
    y_pos = np.arange(len(categories))
    low, high = stacked_bars(ax, y_pos, [decomp_lin, decomp_quad], SPARSITY_PARTS, 0.45, horizontal=True)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel(r"$R^2$ fraction", fontsize=11)
    ax.set_xlim(min(0.0, low) - 0.02, max(1.0, high) + 0.02)
    ax.set_title(f"(d) Accuracy $R^2$ Decomposition\nLinear $R^2={decomp_lin['r2_full']:.3f}$, Quad $R^2={decomp_quad['r2_full']:.3f}$", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.suptitle(f"Baseline: Synapse Sparsity to Accuracy ($N={len(accuracy)}$)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved baseline figure to: {out_path}")
    return {
        "n_runs": int(len(accuracy)),
        "r_intra": float(r_intra), "p_intra": float(p_intra),
        "r_inter": float(r_inter), "p_inter": float(p_inter),
        "linear": decomp_lin, "quadratic": decomp_quad,
    }


# ==============================================================================
# 5. Candidate Figures (7 panels (a)-(g))
# ==============================================================================

def plot_candidate_figure(cand_name, cand_label, cand_values, s_intra, s_inter, mu, sigma, accuracy, out_path):
    valid_mask = ~np.isnan(cand_values) & ~np.isnan(accuracy) & ~np.isnan(s_intra) & ~np.isnan(s_inter)
    cand = cand_values[valid_mask]
    acc = accuracy[valid_mask]
    s_in = s_intra[valid_mask]
    s_out = s_inter[valid_mask]
    m = mu[valid_mask]
    s = sigma[valid_mask]
    n_runs = len(cand)

    # Decompositions
    sp_on_cand_lin = decompose_sparsity_on_candidate(s_in, s_out, cand, quadratic=False)
    sp_on_cand_quad = decompose_sparsity_on_candidate(s_in, s_out, cand, quadratic=True)

    acc_decomp_lin = decompose_accuracy_r2(s_in, s_out, cand, acc, quadratic=False)
    acc_decomp_quad = decompose_accuracy_r2(s_in, s_out, cand, acc, quadratic=True)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1])

    # (a) S_intra vs Candidate
    ax_a = fig.add_subplot(gs[0, 0])
    r_a, p_a = pearsonr(s_in, cand)
    ax_a.scatter(s_in, cand, s=20, color="#dd8452", edgecolors="k", linewidth=0.3)
    ax_a.set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_a.set_ylabel(cand_label, fontsize=10)
    ax_a.set_title(f"(a) $S_{{\\mathrm{{intra}}}}$ vs {cand_name}\n$r = {r_a:+.3f}$ ($p = {p_a:.1e}$)", fontsize=10.5)
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # (b) S_inter vs Candidate
    ax_b = fig.add_subplot(gs[0, 1])
    r_b, p_b = pearsonr(s_out, cand)
    ax_b.scatter(s_out, cand, s=20, color="#4c72b0", edgecolors="k", linewidth=0.3)
    ax_b.set_xlabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_b.set_ylabel(cand_label, fontsize=10)
    ax_b.set_title(f"(b) $S_{{\\mathrm{{inter}}}}$ vs {cand_name}\n$r = {r_b:+.3f}$ ($p = {p_b:.1e}$)", fontsize=10.5)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # (c) Sparsity plane coloured by Candidate
    ax_c = fig.add_subplot(gs[0, 2])
    sc_c = ax_c.scatter(s_in, s_out, s=24, c=cand, cmap="viridis", edgecolors="k", linewidth=0.3)
    ax_c.set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_c.set_ylabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_c.set_title(f"(c) Sparsity Plane coloured by {cand_name}", fontsize=10.5)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    cbar_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label(cand_label, fontsize=9)

    # (d) Candidate R² decomposition by Sparsity
    ax_d = fig.add_subplot(gs[1, 0])
    cats = ["Linear", "Quadratic"]
    y_pos = np.arange(len(cats))
    low, high = stacked_bars(ax_d, y_pos, [sp_on_cand_lin, sp_on_cand_quad], SPARSITY_PARTS, 0.4, horizontal=True)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(cats, fontsize=9.5)
    ax_d.set_xlabel(r"$R^2$ fraction", fontsize=10)
    ax_d.set_xlim(min(0.0, low) - 0.02, max(1.0, high) + 0.02)
    ax_d.set_title(f"(d) {cand_name} $R^2$ from Sparsity\nLin $R^2={sp_on_cand_lin['r2_full']:.3f}$, Quad $R^2={sp_on_cand_quad['r2_full']:.3f}$", fontsize=10.5)
    ax_d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="x")

    # (e) Candidate vs Accuracy (Linear & Quadratic fit)
    ax_e = fig.add_subplot(gs[1, 1])
    r_e, p_e = pearsonr(cand, acc)
    ax_e.scatter(cand, acc, s=20, color="#8172b2", edgecolors="k", linewidth=0.3)
    
    # Fit linear & quadratic
    cand_grid = np.linspace(np.min(cand), np.max(cand), 200)
    _, coef_lin, _, _ = fit_ols_r2([cand], acc)
    pred_grid_lin = coef_lin[0] + coef_lin[1] * cand_grid
    _, coef_quad, _, _ = fit_ols_r2([cand, cand ** 2], acc)
    pred_grid_quad = coef_quad[0] + coef_quad[1] * cand_grid + coef_quad[2] * (cand_grid ** 2)

    ax_e.plot(cand_grid, pred_grid_lin, "b--", lw=1.5, label=f"Lin fit ($R^2={acc_decomp_lin['r2_candidate']:.3f}$)")
    ax_e.plot(cand_grid, pred_grid_quad, "r-", lw=1.5, label=f"Quad fit ($R^2={acc_decomp_quad['r2_candidate']:.3f}$)")
    ax_e.set_xlabel(cand_label, fontsize=10)
    ax_e.set_ylabel("Accuracy", fontsize=10)
    ax_e.set_title(f"(e) {cand_name} vs Accuracy\n$r = {r_e:+.3f}$ ($p = {p_e:.1e}$)", fontsize=10.5)
    ax_e.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax_e.grid(True, linestyle=":", alpha=0.6)

    # (f) Partial Regression Plot (Accuracy residual vs Candidate residual from Sparsity)
    ax_f = fig.add_subplot(gs[1, 2])
    res_x = acc_decomp_quad["res_sp_cand"]
    res_y = acc_decomp_quad["res_sp_acc"]
    r_part, p_part = pearsonr(res_x, res_y)
    ax_f.scatter(res_x, res_y, s=20, color="#55a868", edgecolors="k", linewidth=0.3)
    # fit line on residuals
    _, coef_res, _, _ = fit_ols_r2([res_x], res_y)
    rx_grid = np.linspace(np.min(res_x), np.max(res_x), 100)
    ry_grid = coef_res[0] + coef_res[1] * rx_grid
    ax_f.plot(rx_grid, ry_grid, "k--", lw=1.2, label=f"Partial $r={r_part:+.3f}$")
    ax_f.set_xlabel(f"Residual {cand_name} (Sparsity removed)", fontsize=9.5)
    ax_f.set_ylabel("Residual Accuracy (Sparsity removed)", fontsize=9.5)
    ax_f.set_title(f"(f) Partial Regression (Quad Sparsity removed)\nPartial $r = {r_part:+.3f}$ ($p = {p_part:.1e}$)", fontsize=10.5)
    ax_f.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax_f.grid(True, linestyle=":", alpha=0.6)

    # (g) Accuracy R² 4-way decomposition (Commonality analysis)
    ax_g = fig.add_subplot(gs[2, :])
    cats_g = ["Linear", "Quadratic"]
    y_pos_g = np.arange(len(cats_g))
    low, high = stacked_bars(ax_g, y_pos_g, [acc_decomp_lin, acc_decomp_quad], ACCURACY_PARTS, 0.35, horizontal=True)
    ax_g.set_yticks(y_pos_g)
    ax_g.set_yticklabels(cats_g, fontsize=10)
    ax_g.set_xlabel(r"Accuracy $R^2$ fraction", fontsize=10.5)
    ax_g.set_xlim(min(0.0, low) - 0.02, max(1.0, high) + 0.02)
    ax_g.set_title(f"(g) Accuracy $R^2$ Decomposition: Sparsity vs {cand_name} (Commonality Analysis)\n"
                   f"Linear: Both $R^2={acc_decomp_lin['r2_both']:.3f}$ (Common={acc_decomp_lin['common']:.3f}, Sparsity unique={acc_decomp_lin['unique_sparsity']:.3f}) | "
                   f"Quad: Both $R^2={acc_decomp_quad['r2_both']:.3f}$ (Common={acc_decomp_quad['common']:.3f}, Sparsity unique={acc_decomp_quad['unique_sparsity']:.3f})",
                   fontsize=11)
    ax_g.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=9.5, framealpha=0.9)
    ax_g.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.suptitle(f"Dynamics Candidate: {cand_name} ($N={n_runs}$)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved candidate figure to: {out_path}")

    return {
        "n_runs": n_runs,
        "sparsity_on_candidate_linear": sp_on_cand_lin,
        "sparsity_on_candidate_quadratic": sp_on_cand_quad,
        "accuracy_decomp_linear": {k: float(v) for k, v in acc_decomp_lin.items() if not isinstance(v, np.ndarray)},
        "accuracy_decomp_quadratic": {k: float(v) for k, v in acc_decomp_quad.items() if not isinstance(v, np.ndarray)},
        "r_pearson": float(r_e),
        "p_pearson": float(p_e),
        "r_partial": float(r_part),
        "p_partial": float(p_part),
    }


# ==============================================================================
# 6. Summary Comparison Figure
# ==============================================================================

def plot_comparison_summary(all_results, out_path):
    """Both decompositions for every candidate, linear and quadratic side by side.

    The reference for (b) is each candidate's own sparsity-only R², drawn as a short
    line on its bar: candidates missing a run are fitted on fewer runs, so their
    sparsity-only R² differs from the others'.
    """
    cand_names = list(all_results.keys())
    x_pos = np.arange(len(cand_names))
    width = 0.55
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharex=True)
    for col, model in enumerate(("linear", "quadratic")):
        ax_a, ax_b = axes[0, col], axes[1, col]
        low, high = stacked_bars(ax_a, x_pos, [all_results[c][f"sparsity_on_candidate_{model}"] for c in cand_names],
                                 SPARSITY_PARTS, width, horizontal=False)
        ax_a.set_ylim(min(0.0, low) - 0.02, max(1.0, high) + 0.05)
        ax_a.set_ylabel(r"Candidate $R^2$ fraction", fontsize=11)
        ax_a.set_title(f"(a) Sparsity explaining each candidate ({model})", fontsize=12, fontweight="bold")
        ax_a.grid(True, linestyle=":", alpha=0.6, axis="y")

        acc_decomps = [all_results[c][f"accuracy_decomp_{model}"] for c in cand_names]
        low, high = stacked_bars(ax_b, x_pos, acc_decomps, ACCURACY_PARTS, width, horizontal=False)
        for x, d in zip(x_pos, acc_decomps):
            ax_b.hlines(d["r2_sparsity"], x - width / 2 - 0.08, x + width / 2 + 0.08, colors="k", linestyles="--", lw=1.4)
        ax_b.plot([], [], "k--", lw=1.4, label="Sparsity-only $R^2$ (same runs)")
        ax_b.set_ylim(min(0.0, low) - 0.02, max(1.0, high) + 0.05)
        ax_b.set_ylabel(r"Accuracy $R^2$ fraction", fontsize=11)
        ax_b.set_title(f"(b) Accuracy $R^2$ decomposition ({model})", fontsize=12, fontweight="bold")
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([f"{c}\n(N={all_results[c]['n_runs']})" for c in cand_names], fontsize=9, rotation=35, ha="right")
        ax_b.grid(True, linestyle=":", alpha=0.6, axis="y")
    axes[0, 1].legend(loc="upper right", ncol=4, fontsize=9.5, framealpha=0.9)
    axes[1, 1].legend(loc="upper right", ncol=3, fontsize=9.5, framealpha=0.9)

    fig.suptitle(f"Comparison of {len(cand_names)} dynamics candidates: accuracy $R^2$ shared with sparsity",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved summary comparison figure to: {out_path}")


def plot_combinations(combo_results, out_path):
    """Accuracy partition for each set of candidates: adjusted (top) and raw (bottom),
    linear (left) and quadratic (right). The dashed line on each bar is the
    sparsity-only value on the same runs."""
    names = list(combo_results.keys())
    x_pos = np.arange(len(names))
    width = 0.55
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    for row, kind in enumerate(("adjusted", "raw")):
        for col, model in enumerate(("linear", "quadratic")):
            ax = axes[row, col]
            parts = [combo_results[c][model][kind] for c in names]
            low, high = stacked_bars(ax, x_pos, parts, ACCURACY_PARTS, width, horizontal=False)
            for x, d in zip(x_pos, parts):
                ax.hlines(d["r2_sparsity"], x - width / 2 - 0.08, x + width / 2 + 0.08, colors="k", linestyles="--", lw=1.4)
            ax.plot([], [], "k--", lw=1.4, label="Sparsity-only (same runs)")
            ax.set_ylim(min(0.0, low) - 0.02, max(1.0, high) + 0.05)
            label = "adjusted $R^2$" if kind == "adjusted" else "$R^2$"
            ax.set_ylabel(f"Accuracy {label} fraction", fontsize=11)
            ax.set_title(f"({'ab'[row]}{col + 1}) {model}, {label}", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6, axis="y")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{c}\n(N={combo_results[c]['n_runs']})" for c in names], fontsize=9, rotation=35, ha="right")
    axes[0, 1].legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    fig.suptitle("Accuracy partition between sparsity and sets of dynamics candidates", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved combinations figure to: {out_path}")


def run_combinations(combinations, candidate_values, s_intra, s_inter, accuracy):
    results = {}
    for entry in combinations:
        # An entry is a list of candidate names, or {name, members} to label a long set.
        if isinstance(entry, (dict, DictConfig)):
            members, name = list(entry["members"]), str(entry["name"])
        else:
            members = list(entry)
            name = " + ".join(members)
        values = [candidate_values[m] for m in members]
        mask = ~np.isnan(accuracy)
        for v in values:
            mask &= ~np.isnan(v)
        if mask.sum() < len(accuracy):
            logger.info(f"{name}: {int(len(accuracy) - mask.sum())} run(s) without a value are left out")
        a, b, acc = s_intra[mask], s_inter[mask], accuracy[mask]
        vals = [v[mask] for v in values]
        res = {"members": members, "n_runs": int(mask.sum())}
        for model, quadratic in (("linear", False), ("quadratic", True)):
            set_cols = []
            for v in vals:
                set_cols += build_candidate_design(v, quadratic=quadratic)
            res[model] = decompose_accuracy_set(build_sparsity_design(a, b, quadratic=quadratic), set_cols, acc)
            res[model]["n_terms_candidates"] = len(set_cols)
        results[name] = res
    return results


# ==============================================================================
# 7. Main Execution
# ==============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="candidates")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    out_dir = HydraConfig.get().runtime.output_dir
    runs = load_merged_data(cfg, orig_cwd)

    s_intra = np.array([r["sparsity"]["intra"] for r in runs])
    s_inter = np.array([r["sparsity"]["inter"] for r in runs])
    mu = np.array([r["sparsity"]["syn_loc_mean"] for r in runs])
    sigma = np.array([r["sparsity"]["syn_loc_std"] for r in runs])
    accuracy = np.array([r["accuracy"] for r in runs])

    # Candidate definitions
    sr_raw = np.array([r["sr_raw"] for r in runs])
    sr_z = np.array([r["sr_z"] for r in runs])
    soma_rate = np.array([r["soma_rate"] for r in runs])
    bap_rate = np.array([r["bap_rate"] for r in runs])
    dspike_rate = np.array([r["dspike_rate"] for r in runs])
    dspike_er = np.array([r["mean_dspike_er_orig"] for r in runs])
    bap_er = np.array([r["mean_bap_er_orig"] for r in runs])
    
    bap_rate_x_er = bap_rate * bap_er
    dspike_rate_x_er = dspike_rate * dspike_er
    gamma_bap = np.array([r["gamma_within_bap"] for r in runs])
    gamma_dsp = np.array([r["gamma_within_dsp"] for r in runs])

    candidates = [
        ("SR", "SR (calcium)", sr_raw),
        ("SR (Standardised)", "SR (calcium, per-site standardised)", sr_z),
        ("Soma Rate", "Soma Rate [Hz]", soma_rate),
        ("bAP Rate", "bAP Rate [Hz]", bap_rate),
        ("dSpike Rate", "dSpike Rate [Hz]", dspike_rate),
        ("dSpike ER", "Mean dSpike ER (All PCs)", dspike_er),
        ("bAP ER", "Mean bAP ER (All PCs)", bap_er),
        ("bAP Rate x ER", "bAP Rate x Mean bAP ER [Hz]", bap_rate_x_er),
        ("dSpike Rate x ER", "dSpike Rate x Mean dSpike ER [Hz]", dspike_rate_x_er),
        ("bAP Consistency", "Within-class coincidence, bAP-related compartment spikes", gamma_bap),
        ("dSpike Consistency", "Within-class coincidence, other compartment spikes", gamma_dsp),
    ]

    # 1. Baseline Figure
    baseline_path = os.path.join(out_dir, "candidates_baseline.png")
    baseline_stats = plot_baseline_figure(s_intra, s_inter, mu, sigma, accuracy, baseline_path)

    # 2. Candidate Figures
    results = {}
    slugs = {
        "SR": "sr_raw",
        "SR (Standardised)": "sr_z",
        "Soma Rate": "soma_rate",
        "bAP Rate": "bap_rate",
        "dSpike Rate": "dspike_rate",
        "dSpike ER": "dspike_er",
        "bAP ER": "bap_er",
        "bAP Rate x ER": "bap_rate_x_er",
        "dSpike Rate x ER": "dspike_rate_x_er",
        "bAP Consistency": "bap_consistency",
        "dSpike Consistency": "dspike_consistency",
    }

    for name, label, values in candidates:
        fig_path = os.path.join(out_dir, f"candidate_{slugs[name]}.png")
        if np.all(np.isnan(values)):
            logger.info(f"{name}: no run has a value in the given inputs; skipped")
            continue
        n_missing = int(np.isnan(values).sum())
        if n_missing:
            logger.info(f"{name}: {n_missing} run(s) without a value are left out")
        results[name] = plot_candidate_figure(name, label, values, s_intra, s_inter, mu, sigma, accuracy, fig_path)

    # 3. Summary Comparison Figure
    plot_comparison_summary(results, os.path.join(out_dir, "candidates_comparison.png"))

    # 4. Sets of candidates
    combinations = cfg.get("combinations") or []
    values_by_name = {name: values for name, _, values in candidates}
    combo_results = run_combinations(combinations, values_by_name, s_intra, s_inter, accuracy)
    if combo_results:
        plot_combinations(combo_results, os.path.join(out_dir, "candidates_combinations.png"))

    # 4. Save JSON results
    out_json_file = os.path.join(out_dir, "results.json")
    with open(out_json_file, "w") as f:
        json.dump({"baseline": baseline_stats, "candidates": results, "combinations": combo_results}, f, indent=2)
    logger.info(f"Saved results to: {out_json_file}")


if __name__ == "__main__":
    main()
