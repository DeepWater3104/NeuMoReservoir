"""Count dendritic branches as a function of path distance from the soma.

This is the script version of the Sholl-like analysis prototyped in
``analysis/CellVisualization.ipynb``. The horizontal axis is deliberately the
same quantity that ``syn_loc_mean`` sweeps over in ``main.py`` (path distance
from the centre of ``soma[0]``, in um), so the resulting curve can be overlaid
directly on an accuracy-vs-mu_syn plot.

All parameters live in ``conf/branch_analysis.yaml`` and are overridable on the
command line in the usual Hydra style:

    uv run branch_analysis.py
    uv run branch_analysis.py domain=basal dist_step=25
    uv run branch_analysis.py -m domain=apical,basal
"""

import contextlib
import logging
import os

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_cell(cell_name, compile_mechanisms=True):
    """Instantiate the NEURON cell template, mirroring main.py's loading logic.

    Because hydra.job.chdir is True, the process runs inside the dated output
    directory. nrnivmodl always writes its architecture folder (x86_64, arm64,
    ...) into the *current* directory, so the compilation is pinned to the repo
    root and the resulting library is then loaded explicitly. That keeps the
    output directories free of build artefacts and lets every run reuse the same
    compiled mechanisms, whatever the architecture of the machine.
    """
    import neuron
    from neuron_simulation import (
        run_nrnivmodl,
        get_hoc_morph_for_emodel_folder,
        extract_template_name,
        check_line_in_file,
    )

    cell_dir = os.path.join(REPO_ROOT, "cells", cell_name)
    if compile_mechanisms:
        with working_directory(REPO_ROOT):
            run_nrnivmodl(cell_dir)

    # load_mechanisms resolves the architecture subdirectory by itself.
    if not neuron.load_mechanisms(REPO_ROOT, warn_if_already_loaded=False):
        raise RuntimeError(
            f"No compiled mechanisms found under {REPO_ROOT}. "
            "Run with compile_mechanisms=True."
        )

    from neuron import h as nrn

    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(cell_dir)
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())

    template_name = extract_template_name(hoc_path.as_posix())
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(nrn, template_name)(0, cell_dir + "morphology", morph_path.name)
    else:
        cell = getattr(nrn, template_name)(cell_dir + "morphology", morph_path.name)

    return cell


def get_sections(cell, domain):
    """Return the section list matching the synapse placement domain.

    The domain names follow the ``syn_loc_condition`` values used in conf/config.yaml:
    ``gaussian-apical`` -> apical, ``gaussian-basal`` -> basal, ``gaussian`` -> all.
    """
    if domain == "apical":
        return list(cell.apical)
    if domain == "basal":
        return list(cell.basal)
    if domain == "all":
        return list(cell.apical) + list(cell.basal)
    raise ValueError(f"Unknown domain: {domain}")


def get_section_distance_ranges(cell, sections):
    """Path-distance interval [start, end] of every section, measured from the soma."""
    from neuron import h as nrn

    nrn.finitialize(-65)
    # Reference point for h.distance: the centre of the soma is 0 um,
    # exactly as in NeuronalReservoir.create_synapses.
    nrn.distance(0, 0.5, sec=cell.soma[0])

    ranges = []
    for sec in sections:
        d_start = nrn.distance(sec(0))
        d_end = nrn.distance(sec(1))
        ranges.append((min(d_start, d_end), max(d_start, d_end)))
    return ranges


def count_branches(ranges, distance_grid):
    """Number of sections whose distance interval contains each grid point."""
    return np.array([
        sum(1 for start, end in ranges if start <= d <= end)
        for d in distance_grid
    ], dtype=int)


def plot_branch_counts(distance_grid, branch_counts, domain, filename, show=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distance_grid, branch_counts, color='forestgreen', linewidth=2.5,
            marker='o', label='Number of Branches')
    ax.fill_between(distance_grid, branch_counts, color='forestgreen', alpha=0.1)

    ax.set_xlabel(r"Path Distance from Soma ($\mu$m) [0 = Soma]", fontsize=12)
    ax.set_ylabel("Branch Count (Number of Sections)", fontsize=12)
    ax.set_title(f"{domain.capitalize()} Branch Complexity (Sholl-like Path Analysis)",
                 fontsize=14, pad=20)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(distance_grid[0], distance_grid[-1])
    ax.set_ylim(0, branch_counts.max() + 1 if branch_counts.size else 5)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    logger.info(f"Figure saved to {filename}")
    if show:
        plt.show()
    plt.close(fig)


def save_csv(distance_grid, branch_counts, filename):
    np.savetxt(
        filename,
        np.column_stack([distance_grid, branch_counts]),
        delimiter=",",
        header="distance_um,branch_count",
        comments="",
        fmt=["%.4f", "%d"],
    )
    logger.info(f"CSV saved to {filename}")


@hydra.main(version_base=None, config_path="conf", config_name="branch_analysis")
def main(cfg: DictConfig):
    params = OmegaConf.to_container(cfg, resolve=True)

    logger.info("--- Branch Analysis Configuration ---")
    logger.info(OmegaConf.to_yaml(cfg))

    if not params['show']:
        matplotlib.use("Agg")

    cell = load_cell(params['cell_name'], compile_mechanisms=params['compile_mechanisms'])

    domain = params['domain']
    sections = get_sections(cell, domain)
    if not sections:
        raise RuntimeError(f"No sections found for domain '{domain}' in cell {params['cell_name']}")
    logger.info(f"Domain '{domain}': {len(sections)} sections")

    ranges = get_section_distance_ranges(cell, sections)
    max_extent = max(end for _, end in ranges)
    logger.info(f"Most distal point of the domain: {max_extent:.1f} um")

    step = float(params['dist_step'])
    stop = max_extent if params['full_extent'] else float(params['dist_stop'])
    # +step/2 keeps `stop` itself on the axis despite floating-point accumulation.
    distance_grid = np.arange(float(params['dist_start']), stop + step / 2, step)

    branch_counts = count_branches(ranges, distance_grid)
    for d, n in zip(distance_grid, branch_counts):
        logger.info(f"  {d:7.1f} um : {n} branches")

    # hydra.job.chdir=True なので、ここは outputs/YYYY-MM-DD/HH-MM-SS 以下になる
    figure_dir, data_dir = params['figure_dir'], params['data_dir']
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Output directory: {os.getcwd()}")

    prefix = params['prefix'] or f"branch_count_{domain}"
    save_csv(distance_grid, branch_counts, os.path.join(data_dir, f"{prefix}.csv"))
    plot_branch_counts(distance_grid, branch_counts, domain,
                       os.path.join(figure_dir, f"{prefix}.png"), show=params['show'])


if __name__ == "__main__":
    main()
