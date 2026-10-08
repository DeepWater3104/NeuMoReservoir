"""Synapse sparsity from placement alone: how far apart the synapses sit on the tree.

The quantity of interest is the cable distance between two synapses — the route
along the dendrite from one to the other — because that is what decides whether
they interact. With d_i the path distance of synapse i from the soma and c_ij the
depth of the point where their two routes to the soma part company,

    cable(i,j) = d_i + d_j - 2 c_ij

and that splits identically into two terms:

    cable(i,j) = |d_i - d_j|              how differently deep they sit
               + 2 (min(d_i,d_j) - c_ij)  the backtrack forced by a branching

The second term is zero when one synapse lies on the route to the other, and
grows the closer to the soma their routes separate. Averaging each term over all
pairs gives the two measures, in micrometres, and their sum is the mean cable
distance. Neither refers to the recording sites, and neither needs a notion of
"branch" to be fixed in advance, so the same formulas apply to any morphology.

    uv run sparsity.py 'run_dir=multirun/2026-09-28/11-06-34/*'
"""

import glob
import json
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RUN_INFO_FILENAME = os.path.join("data", "run_info.npz")
HYDRA_CONFIG_FILENAME = os.path.join(".hydra", "config.yaml")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_cell_tree(cell_name):
    """Return, for one cell, each section's parent and the soma distance of its ends.

    This is a property of the morphology rather than of any run, so it is read
    once from the model and applies to every run of that cell.
    """
    import neuron
    from neuron_simulation import (run_nrnivmodl, get_hoc_morph_for_emodel_folder,
                                   extract_template_name, check_line_in_file)

    # The worker compiles the mechanisms before importing NEURON, which then picks
    # them up from the current directory. Here NEURON is already imported by the
    # time we get a cell, so they have to be loaded explicitly — and without them
    # the template's hoc fails to load at all, since it names them. Compiling into
    # the repository root also keeps the build out of every output directory.
    cell_dir = os.path.join(REPO_ROOT, "cells", cell_name)
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        run_nrnivmodl(cell_dir)
    finally:
        os.chdir(previous)
    if not neuron.load_mechanisms(REPO_ROOT, warn_if_already_loaded=False):
        raise RuntimeError(f"No compiled mechanisms found under {REPO_ROOT}")

    from neuron import h as nrn

    # Absolute, so that the files the template pulls in by bare name resolve.
    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(cell_dir)
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())
    template = extract_template_name(hoc_path.as_posix())
    cell = (getattr(nrn, template)(0, cell_dir + "morphology", morph_path.name)
            if check_line_in_file(hoc_path.as_posix(), "gid = $1")
            else getattr(nrn, template)(cell_dir + "morphology", morph_path.name))

    nrn.distance(0, 0.5, sec=cell.soma[0])

    tree = {}
    for sec in cell.all:
        parent_seg = sec.parentseg()
        tree[sec.name()] = {
            "parent": None if parent_seg is None else parent_seg.sec.name(),
            # Where this section attaches to its parent, and how long it is, are
            # what turn a synapse's position within a section into a distance.
            "origin": nrn.distance(sec(0.0)),
            "length": sec.L,
        }
    return tree, cell


def route_to_soma(tree, section):
    """The chain of sections from this one back to the root, root last."""
    route = []
    while section is not None:
        route.append(section)
        section = tree[section]["parent"]
    return route


def branch_point_depths(tree, syn_names, syn_distance):
    """Soma distance of the point where each pair of synapses' routes separate.

    Two synapses share the stretch of dendrite their routes have in common. That
    stretch ends either at a branch point, or — when one lies on the route to the
    other — at the nearer synapse itself, in which case the depth is simply the
    smaller of their two distances.
    """
    routes = {name: set(route_to_soma(tree, name)) for name in set(syn_names)}
    chains = {name: route_to_soma(tree, name) for name in set(syn_names)}

    n = len(syn_names)
    depths = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            name_i, name_j = syn_names[i], syn_names[j]

            if name_i == name_j or name_j in routes[name_i] or name_i in routes[name_j]:
                # One lies on the other's route: they never part before the nearer.
                depth = min(syn_distance[i], syn_distance[j])
            else:
                # The deepest section both routes pass through; they separate where
                # it ends, so the depth is that section's far end.
                common = routes[name_i] & routes[name_j]
                deepest = max(common, key=lambda s: tree[s]["origin"] + tree[s]["length"])
                depth = tree[deepest]["origin"] + tree[deepest]["length"]
                depth = min(depth, syn_distance[i], syn_distance[j])

            depths[i, j] = depths[j, i] = depth
    return depths


def sparsity_for_run(tree, syn_names, syn_distance):
    """The two measures, and the mean cable distance they add up to."""
    n = len(syn_names)
    depths = branch_point_depths(tree, list(syn_names), syn_distance)

    upper = np.triu_indices(n, k=1)
    d_i = syn_distance[upper[0]]
    d_j = syn_distance[upper[1]]
    c = depths[upper]

    depth_difference = np.abs(d_i - d_j)
    backtrack = 2.0 * (np.minimum(d_i, d_j) - c)

    return {
        "intra": float(depth_difference.mean()),
        "inter": float(backtrack.mean()),
        "cable": float((depth_difference + backtrack).mean()),
        "fraction_branching": float((backtrack > 0).mean()),
    }


def run_plot(cfg, original_cwd):
    """Five views of the two measures, on one page.

    They answer three questions in order: whether the measures are separable at
    all, whether they carry anything beyond the parameters that generated them,
    and whether they reach the outcome. Colour is the placement mean throughout,
    on one sequential ramp, so a point keeps its identity across panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cfg.sparsity_path is None:
        raise ValueError("sparsity_path is required in plot mode")

    from omegaconf import ListConfig
    sparsity_inputs = cfg.sparsity_path
    if isinstance(sparsity_inputs, (list, ListConfig)):
        sparsity_paths = list(sparsity_inputs)
    else:
        sparsity_paths = [sparsity_inputs]

    rows = []
    for p in sparsity_paths:
        if not os.path.isabs(p):
            p = os.path.join(original_cwd, p)
        with open(p) as f:
            batch_rows = json.load(f)
            rows.extend(batch_rows)
    logger.info(f"Loaded {len(rows)} total sparsity records from {len(sparsity_paths)} path(s)")

    # The accuracy carried in sparsity.json is the one the run itself reported,
    # fitted on every recorded site. That is not the quantity the design calls
    # for: accuracy is defined as the mean over random readouts of a fixed size,
    # so that it does not depend on which sites were used and stays comparable
    # across cells with different compartment counts. Reading every site instead
    # raises it — measured at 0.72 against 0.53 for a hundred sites on this cell
    # — and pushes the runs toward the ceiling, where differences compress.
    accuracy_label = "Test accuracy (all sites)"
    if cfg.accuracy_path is not None:
        acc_inputs = cfg.accuracy_path
        if isinstance(acc_inputs, (list, ListConfig)):
            acc_paths = list(acc_inputs)
        else:
            acc_paths = [acc_inputs]

        marginalised = {}
        for ap in acc_paths:
            if not os.path.isabs(ap):
                ap = os.path.join(original_cwd, ap)
            with open(ap) as f:
                summary = json.load(f)
            for run in summary["runs"]:
                sizes = run["sizes"]
                if len(sizes) != 1:
                    raise ValueError(
                        f"Expected one subset size per run, found {[s['num_readout_sites'] for s in sizes]}")
                marginalised[run["source_run"]] = (sizes[0]["test_accuracy_mean"],
                                                   sizes[0]["num_readout_sites"],
                                                   sizes[0]["num_draws"])

        common_rows = [r for r in rows if r["run_dir"] in marginalised]
        if not common_rows:
            raise ValueError("No common runs found between sparsity.json and accuracy results.json")

        # Deduplicate rows by run_dir if any overlap
        seen_runs = set()
        dedup_rows = []
        for r in common_rows:
            if r["run_dir"] not in seen_runs:
                seen_runs.add(r["run_dir"])
                dedup_rows.append(r)
        rows = dedup_rows

        logger.info(f"Matched {len(rows)} unique common runs across both pipelines "
                    f"(sparsity: {len(rows)}, accuracy: {len(marginalised)})")
        k = {marginalised[r["run_dir"]][1] for r in rows}
        draws = {marginalised[r["run_dir"]][2] for r in rows}
        for r in rows:
            r["test_accuracy"] = marginalised[r["run_dir"]][0]
        accuracy_label = f"Test accuracy (mean over {max(draws)} readouts of {max(k)} sites)"
        logger.info(f"Using marginalised accuracy: k={max(k)}, {max(draws)} draws per run")

    intra = np.array([r["intra"] for r in rows])
    inter = np.array([r["inter"] for r in rows])
    mu = np.array([r["syn_loc_mean"] for r in rows])
    sigma = np.array([r["syn_loc_std"] for r in rows])
    accuracy = np.array([r["test_accuracy"] for r in rows])

    # Three quantities are used as colour, so each gets its own single-hue ramp,
    # light to dark, with the palest quarter cut so no point vanishes against the
    # page. Keeping one hue per quantity means a colour can be read back to what
    # it stands for without consulting which panel it came from.
    RAMPS = {"mu": "Blues", "sigma": "Oranges", "accuracy": "Purples"}

    def shade(values, ramp):
        cmap = plt.get_cmap(ramp)
        span = max(1e-9, values.max() - values.min())
        return cmap(0.25 + 0.75 * (values - values.min()) / span)

    # Each panel of (2) is drawn against the parameter that explains its measure —
    # sigma sets how far apart in depth the synapses fall, mu sets which part of
    # the tree they land on and so how finely it has branched there — and coloured
    # by the other, so what the pair leaves unexplained shows as scatter the colour
    # does not organise. Panels (3) keep that pairing.
    by_mu = shade(mu, RAMPS["mu"])
    by_sigma = shade(sigma, RAMPS["sigma"])
    by_accuracy = shade(accuracy, RAMPS["accuracy"])

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5))
    panels = [
        # Row 0: Sparsity plane & parameters to sparsity
        (axes[0][0], intra, inter, by_accuracy,
         "S_intra  (depth difference) [um]", "S_inter  (backtrack) [um]",
         "(1) Separable, and where accuracy sits"),
        (axes[0][1], sigma, intra, by_mu,
         "sigma_syn [um]", "S_intra [um]",
         "(2a) S_intra against sigma, coloured by mu"),
        (axes[0][2], mu, inter, by_sigma,
         "mu_syn [um]", "S_inter [um]",
         "(2b) S_inter against mu, coloured by sigma"),

        # Row 1: Parameters directly to accuracy
        (axes[1][0], sigma, accuracy, by_mu,
         "sigma_syn [um]", accuracy_label,
         "(3a) Accuracy against sigma, coloured by mu"),
        (axes[1][1], mu, accuracy, by_sigma,
         "mu_syn [um]", accuracy_label,
         "(3b) Accuracy against mu, coloured by sigma"),

        # Row 2: Sparsity to accuracy (slid from old Row 1)
        (axes[2][0], intra, accuracy, by_mu,
         "S_intra [um]", accuracy_label,
         "(4a) Does S_intra reach the outcome?"),
        (axes[2][1], inter, accuracy, by_sigma,
         "S_inter [um]", accuracy_label,
         "(4b) Does S_inter reach the outcome?"),
    ]
    for ax, x, y, colours, xlabel, ylabel, title in panels:
        ax.scatter(x, y, c=colours, s=42, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.97, 0.04, f"r = {np.corrcoef(x, y)[0, 1]:+.2f}",
                transform=ax.transAxes, ha="right", fontsize=9, color="#52514e")

    axes[1][2].axis("off")
    axes[2][2].axis("off")

    fig.suptitle(f"Synapse placement sparsity ({len(rows)} runs, cell1)", fontsize=13)
    fig.tight_layout()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cb_items = [
        (mu, "mu_syn [um]", RAMPS["mu"], 0.72),
        (sigma, "sigma_syn [um]", RAMPS["sigma"], 0.42),
        (accuracy, accuracy_label, RAMPS["accuracy"], 0.12),
    ]

    for values, title_text, ramp, y_rel in cb_items:
        cax = inset_axes(
            axes[2][2],
            width="82%",
            height="9%",
            loc="lower left",
            bbox_to_anchor=(0.08, y_rel, 1.0, 1.0),
            bbox_transform=axes[2][2].transAxes,
            borderpad=0,
        )
        norm = plt.Normalize(values.min(), values.max())
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(ramp)),
            cax=cax,
            orientation="horizontal",
        )
        cax.set_title(title_text, fontsize=8.5, pad=5, loc="left")
        cbar.ax.tick_params(labelsize=8)
    os.makedirs(cfg.figure_dir, exist_ok=True)
    figure_path = os.path.join(cfg.figure_dir, "sparsity_overview.png")
    fig.savefig(figure_path, dpi=200)
    logger.info(f"Saved {figure_path}")
    plt.close(fig)


def _r_squared(columns, y, quadratic=False):
    """Least-squares R² of y on two columns, linear or full quadratic, with intercept."""
    a, b = columns
    x = np.column_stack([a, b, a ** 2, b ** 2, a * b] if quadratic else [a, b])
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return float(1.0 - residual @ residual / np.sum((y - y.mean()) ** 2))


def run_by_size(cfg, original_cwd):
    """How the sparsity–accuracy relation changes with the number of readout sites k.

    accuracy_path holds readout_subset results with several subset sizes; each
    size gives one marginalised accuracy per run, joined to sparsity by run_dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import ListConfig
    from scipy.stats import pearsonr

    def as_list(paths):
        return list(paths) if isinstance(paths, (list, ListConfig)) else [paths]

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(original_cwd, path)

    sparsity = {}
    for p in as_list(cfg.sparsity_path):
        with open(resolve(p)) as f:
            for rec in json.load(f):
                sparsity[rec["run_dir"]] = rec
    accuracy = {}  # run_dir -> {k: accuracy}
    for p in as_list(cfg.accuracy_path):
        with open(resolve(p)) as f:
            for run in json.load(f)["runs"]:
                accuracy[run["source_run"]] = {s["num_readout_sites"]: s["test_accuracy_mean"]
                                               for s in run["sizes"]}

    runs = sorted(d for d in accuracy if d in sparsity)
    sizes = sorted({k for d in runs for k in accuracy[d]})
    runs = [d for d in runs if all(k in accuracy[d] for k in sizes)]
    logger.info(f"{len(runs)} runs with sparsity and accuracy at k = {sizes}")

    intra = np.array([sparsity[d]["intra"] for d in runs])
    inter = np.array([sparsity[d]["inter"] for d in runs])
    acc = {k: np.array([accuracy[d][k] for d in runs]) for k in sizes}

    stats = {"num_runs": len(runs), "by_size": {}}
    for k in sizes:
        a = acc[k]
        stats["by_size"][str(k)] = {
            "accuracy_mean": float(a.mean()), "accuracy_sd": float(a.std()),
            "accuracy_min": float(a.min()), "accuracy_max": float(a.max()),
            "fraction_at_or_above_0.9": float(np.mean(a >= 0.9)),
            "r_intra": float(pearsonr(intra, a)[0]), "r_inter": float(pearsonr(inter, a)[0]),
            "r2_linear": _r_squared((intra, inter), a),
            "r2_quadratic": _r_squared((intra, inter), a, quadratic=True),
        }
    out_dir = HydraConfig.get().runtime.output_dir
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(json.dumps(stats, indent=2))

    # Top: S_intra against accuracy at each k. Bottom: the summaries across k.
    fig = plt.figure(figsize=(3.2 * len(sizes), 7.2))
    grid = fig.add_gridspec(2, len(sizes), height_ratios=[1, 1])
    for i, k in enumerate(sizes):
        ax = fig.add_subplot(grid[0, i])
        ax.scatter(intra, acc[k], s=12, alpha=0.7, c=inter, cmap="Oranges", edgecolors="k", linewidth=0.2)
        s = stats["by_size"][str(k)]
        ax.set_title(f"k = {k}\nr(S_intra) = {s['r_intra']:+.2f}", fontsize=9)
        ax.set_xlabel("S_intra [µm]")
        if i == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.6)
    third = max(1, len(sizes) // 3)
    ax = fig.add_subplot(grid[1, :third])
    ax.boxplot([acc[k] for k in sizes], labels=[str(k) for k in sizes])
    ax.set_xlabel("k (readout sites)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across runs", fontsize=9)
    ax = fig.add_subplot(grid[1, third:2 * third])
    ax.plot(sizes, [stats["by_size"][str(k)]["r_intra"] for k in sizes], "o-", label="r(S_intra, Acc)")
    ax.plot(sizes, [stats["by_size"][str(k)]["r_inter"] for k in sizes], "s-", label="r(S_inter, Acc)")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(k) for k in sizes])
    ax.set_xlabel("k (readout sites)")
    ax.set_title("Correlation with accuracy", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax = fig.add_subplot(grid[1, 2 * third:])
    ax.plot(sizes, [stats["by_size"][str(k)]["r2_linear"] for k in sizes], "o-", label="linear")
    ax.plot(sizes, [stats["by_size"][str(k)]["r2_quadratic"] for k in sizes], "s-", label="quadratic")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(k) for k in sizes])
    ax.set_xlabel("k (readout sites)")
    ax.set_title("R² of accuracy on (S_intra, S_inter)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    path = os.path.join(out_dir, "sparsity_by_readout_size.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved {path}")


@hydra.main(version_base=None, config_path="conf", config_name="sparsity")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    original_cwd = get_original_cwd()

    if cfg.mode == "plot":
        run_plot(cfg, original_cwd)
        return
    if cfg.mode == "by_size":
        run_by_size(cfg, original_cwd)
        return

    patterns = [cfg.run_dir] if isinstance(cfg.run_dir, str) else list(cfg.run_dir)
    run_dirs = []
    for pattern in patterns:
        if not os.path.isabs(pattern):
            pattern = os.path.join(original_cwd, pattern)
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        run_dirs += [m for m in matches if os.path.exists(os.path.join(m, RUN_INFO_FILENAME))]
    if not run_dirs:
        raise FileNotFoundError(f"No run directory holding {RUN_INFO_FILENAME} matched {patterns}")
    logger.info(f"Found {len(run_dirs)} run(s)")

    trees = {}
    rows = []
    for run_dir in run_dirs:
        try:
            config = OmegaConf.load(os.path.join(run_dir, HYDRA_CONFIG_FILENAME))
            cell_name = str(config.cell_name)
            if cell_name not in trees:
                trees[cell_name], _ = load_cell_tree(cell_name)
                logger.info(f"Loaded morphology for {cell_name}: {len(trees[cell_name])} sections")

            with np.load(os.path.join(run_dir, RUN_INFO_FILENAME), allow_pickle=False) as npz:
                syn_names = npz["syn_names"]
                syn_distance = npz["syn_distance"]
                accuracy = float((npz["test_predicted_label"] == npz["test_label"]).mean())

            row = sparsity_for_run(trees[cell_name], syn_names, syn_distance)
            row.update({
                "run_dir": run_dir,
                "cell_name": cell_name,
                "sample_id": int(config.sample_id) if config.sample_id is not None else -1,
                "syn_loc_mean": float(config.syn_loc_mean),
                "syn_loc_std": float(config.syn_loc_std),
                "seed": int(config.seed),
                "test_accuracy": accuracy,
            })
            rows.append(row)
        except Exception as e:
            logger.warning(f"Skipping run {run_dir} in sparsity: {e}")
            continue

    if not rows:
        raise RuntimeError("No run was successfully processed in sparsity compute.")

    os.makedirs(cfg.data_dir, exist_ok=True)
    results_path = os.path.join(cfg.data_dir, "sparsity.json")
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved {results_path}")

    intra = np.array([r["intra"] for r in rows])
    inter = np.array([r["inter"] for r in rows])
    logger.info(f"intra: {intra.min():.1f} - {intra.max():.1f} um  (mean {intra.mean():.1f})")
    logger.info(f"inter: {inter.min():.1f} - {inter.max():.1f} um  (mean {inter.mean():.1f})")
    if len(rows) > 2:
        logger.info(f"corr(intra, inter) = {np.corrcoef(intra, inter)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
