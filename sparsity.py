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

    path = cfg.sparsity_path
    if path is None:
        raise ValueError("sparsity_path is required in plot mode")
    if not os.path.isabs(path):
        path = os.path.join(original_cwd, path)
    with open(path) as f:
        rows = json.load(f)

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

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    panels = [
        (axes[0][0], intra, inter, by_accuracy,
         "S_intra  (depth difference) [um]", "S_inter  (backtrack) [um]",
         "(1) Separable, and where accuracy sits"),
        (axes[0][1], sigma, intra, by_mu,
         "sigma_syn [um]", "S_intra [um]",
         "(2a) S_intra against sigma, coloured by mu"),
        (axes[0][2], mu, inter, by_sigma,
         "mu_syn [um]", "S_inter [um]",
         "(2b) S_inter against mu, coloured by sigma"),
        (axes[1][0], intra, accuracy, by_mu,
         "S_intra [um]", "Test accuracy",
         "(3a) Does it reach the outcome?"),
        (axes[1][1], inter, accuracy, by_sigma,
         "S_inter [um]", "Test accuracy",
         "(3b) Does it reach the outcome?"),
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
    for values, label, ramp in ((mu, "mu_syn [um]", RAMPS["mu"]),
                                (sigma, "sigma_syn [um]", RAMPS["sigma"]),
                                (accuracy, "Test accuracy", RAMPS["accuracy"])):
        bar = fig.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(values.min(), values.max()), cmap=plt.get_cmap(ramp)),
            ax=axes[1][2], fraction=0.26, aspect=10, location="left")
        bar.set_label(label)

    fig.suptitle(f"Synapse placement sparsity ({len(rows)} runs, cell1)", fontsize=13)
    fig.tight_layout()
    os.makedirs(cfg.figure_dir, exist_ok=True)
    figure_path = os.path.join(cfg.figure_dir, "sparsity_overview.png")
    fig.savefig(figure_path, dpi=200)
    logger.info(f"Saved {figure_path}")
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="sparsity")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    original_cwd = get_original_cwd()

    if cfg.mode == "plot":
        run_plot(cfg, original_cwd)
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
