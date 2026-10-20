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

A run without a value for a candidate (e.g. ER with fewer than 2 events) is left out
of that candidate only.

Usage:
    uv run python candidates.py \\
        'sparsity_path=[data/2026-09-30_sweep200/caracal/sparsity.json,data/2026-09-30_sweep200/oncilla/sparsity.json]' \\
        'accuracy_path=[data/2026-09-30_sweep200/caracal/results.json,data/2026-09-30_sweep200/oncilla/results.json]' \\
        'separation_path=[data/2026-10-06-separation/caracal.npz,data/2026-10-06-separation/oncilla.npz]' \\
        'dynamics_path=[data/2026-10-11-rates/caracal.json,data/2026-10-11-rates/oncilla.json]'
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


DYNAMICS_KEYS = ("soma_rate", "bap_rate", "dspike_rate", "mean_dspike_er_orig", "mean_bap_er_orig")


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
                        runs[rd][k] = _value(item.get(k))

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
    3. common (mediated): R²(sparsity) + R²(cand) - R²(both)
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


# ==============================================================================
# 3. Plot Helpers
# ==============================================================================

RAMPS = {"mu": "Blues", "sigma": "Oranges", "accuracy": "Purples"}

def shade(values, ramp):
    cmap = plt.get_cmap(ramp)
    span = max(1e-9, values.max() - values.min())
    return cmap(0.25 + 0.75 * (values - values.min()) / span)


# ==============================================================================
# 4. Baseline Figure
# ==============================================================================

def plot_baseline_figure(s_intra, s_inter, mu, sigma, accuracy, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # (a) S_intra vs Accuracy
    r_intra, p_intra = pearsonr(s_intra, accuracy)
    axes[0, 0].scatter(s_intra, accuracy, s=24, alpha=0.8, c=shade(sigma, RAMPS["sigma"]), edgecolors="k", linewidth=0.3)
    axes[0, 0].set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0, 0].set_ylabel("Accuracy", fontsize=11)
    axes[0, 0].set_title(f"(a) $S_{{\\mathrm{{intra}}}}$ vs Accuracy\n$r = {r_intra:+.3f}$ ($p = {p_intra:.1e}$)", fontsize=11)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # (b) S_inter vs Accuracy
    r_inter, p_inter = pearsonr(s_inter, accuracy)
    axes[0, 1].scatter(s_inter, accuracy, s=24, alpha=0.8, c=shade(mu, RAMPS["mu"]), edgecolors="k", linewidth=0.3)
    axes[0, 1].set_xlabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0, 1].set_ylabel("Accuracy", fontsize=11)
    axes[0, 1].set_title(f"(b) $S_{{\\mathrm{{inter}}}}$ vs Accuracy\n$r = {r_inter:+.3f}$ ($p = {p_inter:.1e}$)", fontsize=11)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # (c) S_intra x S_inter plane with Accuracy
    sc = axes[1, 0].scatter(s_intra, s_inter, s=28, alpha=0.85, c=accuracy, cmap=RAMPS["accuracy"], edgecolors="k", linewidth=0.3)
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
    u_intra = [decomp_lin["unique_intra"], decomp_quad["unique_intra"]]
    u_inter = [decomp_lin["unique_inter"], decomp_quad["unique_inter"]]
    com = [decomp_lin["common"], decomp_quad["common"]]
    unexp = [decomp_lin["unexplained"], decomp_quad["unexplained"]]

    ax = axes[1, 1]
    y_pos = np.arange(len(categories))
    h = 0.45

    b1 = np.zeros(2)
    ax.barh(y_pos, u_intra, height=h, left=b1, label=r"$S_{\mathrm{intra}}$ unique", color="#4c72b0", edgecolor="k", linewidth=0.5)
    b1 += np.array(u_intra)
    ax.barh(y_pos, u_inter, height=h, left=b1, label=r"$S_{\mathrm{inter}}$ unique", color="#dd8452", edgecolor="k", linewidth=0.5)
    b1 += np.array(u_inter)
    ax.barh(y_pos, com, height=h, left=b1, label="Common", color="#8172b2", edgecolor="k", linewidth=0.5)
    b1 += np.array(com)
    ax.barh(y_pos, unexp, height=h, left=b1, label="Unexplained", color="#cccccc", edgecolor="k", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel(r"$R^2$ fraction", fontsize=11)
    ax.set_xlim(0, 1.0)
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
    ax_a.scatter(s_in, cand, s=20, alpha=0.8, c=shade(s, RAMPS["sigma"]), edgecolors="k", linewidth=0.3)
    ax_a.set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_a.set_ylabel(cand_label, fontsize=10)
    ax_a.set_title(f"(a) $S_{{\\mathrm{{intra}}}}$ vs {cand_name}\n$r = {r_a:+.3f}$ ($p = {p_a:.1e}$)", fontsize=10.5)
    ax_a.grid(True, linestyle=":", alpha=0.6)

    # (b) S_inter vs Candidate
    ax_b = fig.add_subplot(gs[0, 1])
    r_b, p_b = pearsonr(s_out, cand)
    ax_b.scatter(s_out, cand, s=20, alpha=0.8, c=shade(m, RAMPS["mu"]), edgecolors="k", linewidth=0.3)
    ax_b.set_xlabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_b.set_ylabel(cand_label, fontsize=10)
    ax_b.set_title(f"(b) $S_{{\\mathrm{{inter}}}}$ vs {cand_name}\n$r = {r_b:+.3f}$ ($p = {p_b:.1e}$)", fontsize=10.5)
    ax_b.grid(True, linestyle=":", alpha=0.6)

    # (c) Sparsity plane coloured by Candidate
    ax_c = fig.add_subplot(gs[0, 2])
    sc_c = ax_c.scatter(s_in, s_out, s=24, alpha=0.85, c=cand, cmap="viridis", edgecolors="k", linewidth=0.3)
    ax_c.set_xlabel(r"$S_{\mathrm{intra}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_c.set_ylabel(r"$S_{\mathrm{inter}}$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_c.set_title(f"(c) Sparsity Plane coloured by {cand_name}", fontsize=10.5)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    cbar_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label(cand_label, fontsize=9)

    # (d) Candidate R² decomposition by Sparsity
    ax_d = fig.add_subplot(gs[1, 0])
    cats = ["Linear", "Quadratic"]
    u_in = [sp_on_cand_lin["unique_intra"], sp_on_cand_quad["unique_intra"]]
    u_out = [sp_on_cand_lin["unique_inter"], sp_on_cand_quad["unique_inter"]]
    c_sp = [sp_on_cand_lin["common"], sp_on_cand_quad["common"]]
    un_sp = [sp_on_cand_lin["unexplained"], sp_on_cand_quad["unexplained"]]

    y_pos = np.arange(len(cats))
    h = 0.4
    b = np.zeros(2)
    ax_d.barh(y_pos, u_in, height=h, left=b, label=r"$S_{\mathrm{intra}}$ unique", color="#4c72b0", edgecolor="k", linewidth=0.5)
    b += np.array(u_in)
    ax_d.barh(y_pos, u_out, height=h, left=b, label=r"$S_{\mathrm{inter}}$ unique", color="#dd8452", edgecolor="k", linewidth=0.5)
    b += np.array(u_out)
    ax_d.barh(y_pos, c_sp, height=h, left=b, label="Common", color="#8172b2", edgecolor="k", linewidth=0.5)
    b += np.array(c_sp)
    ax_d.barh(y_pos, un_sp, height=h, left=b, label="Unexplained", color="#cccccc", edgecolor="k", linewidth=0.5)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(cats, fontsize=9.5)
    ax_d.set_xlabel(r"$R^2$ fraction", fontsize=10)
    ax_d.set_xlim(0, 1.0)
    ax_d.set_title(f"(d) {cand_name} $R^2$ from Sparsity\nLin $R^2={sp_on_cand_lin['r2_full']:.3f}$, Quad $R^2={sp_on_cand_quad['r2_full']:.3f}$", fontsize=10.5)
    ax_d.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_d.grid(True, linestyle=":", alpha=0.6, axis="x")

    # (e) Candidate vs Accuracy (Linear & Quadratic fit)
    ax_e = fig.add_subplot(gs[1, 1])
    r_e, p_e = pearsonr(cand, acc)
    ax_e.scatter(cand, acc, s=20, alpha=0.8, color="#8172b2", edgecolors="k", linewidth=0.3)
    
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
    ax_f.scatter(res_x, res_y, s=20, alpha=0.8, color="#55a868", edgecolors="k", linewidth=0.3)
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
    u_sp_g = [acc_decomp_lin["unique_sparsity"], acc_decomp_quad["unique_sparsity"]]
    u_cand_g = [acc_decomp_lin["unique_candidate"], acc_decomp_quad["unique_candidate"]]
    med_g = [acc_decomp_lin["common"], acc_decomp_quad["common"]]
    un_g = [acc_decomp_lin["unexplained"], acc_decomp_quad["unexplained"]]

    y_pos_g = np.arange(len(cats_g))
    h_g = 0.35
    bg = np.zeros(2)

    ax_g.barh(y_pos_g, u_sp_g, height=h_g, left=bg, label="(1) Sparsity unique (unmediated)", color="#dd8452", edgecolor="k", linewidth=0.5)
    bg += np.array(u_sp_g)
    ax_g.barh(y_pos_g, med_g, height=h_g, left=bg, label="(3) Common (mediated by candidate)", color="#4c72b0", edgecolor="k", linewidth=0.5)
    bg += np.array(med_g)
    ax_g.barh(y_pos_g, u_cand_g, height=h_g, left=bg, label="(2) Candidate unique (independent)", color="#55a868", edgecolor="k", linewidth=0.5)
    bg += np.array(u_cand_g)
    ax_g.barh(y_pos_g, un_g, height=h_g, left=bg, label="(4) Unexplained residual", color="#cccccc", edgecolor="k", linewidth=0.5)

    ax_g.set_yticks(y_pos_g)
    ax_g.set_yticklabels(cats_g, fontsize=10)
    ax_g.set_xlabel(r"Accuracy $R^2$ fraction", fontsize=10.5)
    ax_g.set_xlim(0, 1.0)
    ax_g.set_title(f"(g) Accuracy $R^2$ Decomposition: Sparsity vs {cand_name} (Commonality Analysis)\n"
                   f"Linear: Both $R^2={acc_decomp_lin['r2_both']:.3f}$ (Med={acc_decomp_lin['common']:.3f}, Unmed={acc_decomp_lin['unique_sparsity']:.3f}) | "
                   f"Quad: Both $R^2={acc_decomp_quad['r2_both']:.3f}$ (Med={acc_decomp_quad['common']:.3f}, Unmed={acc_decomp_quad['unique_sparsity']:.3f})",
                   fontsize=11)
    ax_g.legend(loc="upper right", bbox_to_anchor=(1.0, 1.35), ncol=2, fontsize=9.5, framealpha=0.9)
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
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    cand_names = list(all_results.keys())
    x_pos = np.arange(len(cand_names))
    width = 0.55

    # (a) Sparsity R² on Candidate (Quadratic)
    ax_a = axes[0]
    u_in = [all_results[c]["sparsity_on_candidate_quadratic"]["unique_intra"] for c in cand_names]
    u_out = [all_results[c]["sparsity_on_candidate_quadratic"]["unique_inter"] for c in cand_names]
    com_sp = [all_results[c]["sparsity_on_candidate_quadratic"]["common"] for c in cand_names]
    un_sp = [all_results[c]["sparsity_on_candidate_quadratic"]["unexplained"] for c in cand_names]

    b = np.zeros(len(cand_names))
    ax_a.bar(x_pos, u_in, width, bottom=b, label=r"$S_{\mathrm{intra}}$ unique", color="#4c72b0", edgecolor="k", linewidth=0.5)
    b += np.array(u_in)
    ax_a.bar(x_pos, u_out, width, bottom=b, label=r"$S_{\mathrm{inter}}$ unique", color="#dd8452", edgecolor="k", linewidth=0.5)
    b += np.array(u_out)
    ax_a.bar(x_pos, com_sp, width, bottom=b, label="Common", color="#8172b2", edgecolor="k", linewidth=0.5)
    b += np.array(com_sp)
    ax_a.bar(x_pos, un_sp, width, bottom=b, label="Unexplained", color="#cccccc", edgecolor="k", linewidth=0.5)

    ax_a.set_ylabel(r"Candidate $R^2$ fraction", fontsize=11)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("(a) Sparsity Explaining Each Dynamics Candidate ($R^2$ Decomposition, Quadratic)", fontsize=12, fontweight="bold")
    ax_a.legend(loc="upper right", ncol=4, fontsize=9.5, framealpha=0.9)
    ax_a.grid(True, linestyle=":", alpha=0.6, axis="y")

    # (b) Accuracy R² 4-way decomposition (Quadratic)
    ax_b = axes[1]
    u_sp = [all_results[c]["accuracy_decomp_quadratic"]["unique_sparsity"] for c in cand_names]
    med = [all_results[c]["accuracy_decomp_quadratic"]["common"] for c in cand_names]
    u_cand = [all_results[c]["accuracy_decomp_quadratic"]["unique_candidate"] for c in cand_names]
    un_acc = [all_results[c]["accuracy_decomp_quadratic"]["unexplained"] for c in cand_names]

    bg = np.zeros(len(cand_names))
    ax_b.bar(x_pos, u_sp, width, bottom=bg, label="(1) Sparsity unique (unmediated)", color="#dd8452", edgecolor="k", linewidth=0.5)
    bg += np.array(u_sp)
    ax_b.bar(x_pos, med, width, bottom=bg, label="(3) Common (mediated by candidate)", color="#4c72b0", edgecolor="k", linewidth=0.5)
    bg += np.array(med)
    ax_b.bar(x_pos, u_cand, width, bottom=bg, label="(2) Candidate unique (independent)", color="#55a868", edgecolor="k", linewidth=0.5)
    bg += np.array(u_cand)
    ax_b.bar(x_pos, un_acc, width, bottom=bg, label="(4) Unexplained residual", color="#cccccc", edgecolor="k", linewidth=0.5)

    ax_b.set_ylabel(r"Accuracy $R^2$ fraction", fontsize=11)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_title("(b) Accuracy $R^2$ 4-Way Decomposition across Candidates (Commonality Analysis, Quadratic)", fontsize=12, fontweight="bold")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f"{c}\n(N={all_results[c]['n_runs']})" for c in cand_names], fontsize=9.5, rotation=15, ha="right")
    ax_b.legend(loc="upper right", ncol=2, fontsize=9.5, framealpha=0.9)
    ax_b.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Add reference line for baseline sparsity R²
    baseline_quad_r2 = all_results[cand_names[0]]["accuracy_decomp_quadratic"]["r2_sparsity"]
    ax_b.axhline(baseline_quad_r2, color="k", linestyle="--", lw=1.2, alpha=0.7, label=f"Baseline Sparsity Quad $R^2={baseline_quad_r2:.3f}$")

    fig.suptitle("Comparison of All 9 Dynamics Candidates Mediating Sparsity to Accuracy", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved summary comparison figure to: {out_path}")


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
    }

    for name, label, values in candidates:
        fig_path = os.path.join(out_dir, f"candidate_{slugs[name]}.png")
        n_missing = int(np.isnan(values).sum())
        if n_missing:
            logger.info(f"{name}: {n_missing} run(s) without a value are left out")
        results[name] = plot_candidate_figure(name, label, values, s_intra, s_inter, mu, sigma, accuracy, fig_path)

    # 3. Summary Comparison Figure
    plot_comparison_summary(results, os.path.join(out_dir, "candidates_comparison.png"))

    # 4. Save JSON results
    out_json_file = os.path.join(out_dir, "results.json")
    with open(out_json_file, "w") as f:
        json.dump({"baseline": baseline_stats, "candidates": results}, f, indent=2)
    logger.info(f"Saved results to: {out_json_file}")


if __name__ == "__main__":
    main()
