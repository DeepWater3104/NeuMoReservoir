"""Accuracy as a function of how many recording sites feed the readout.

A single ``main.py`` run records ``num_states`` sites and fits one readout on all
of them. Whether the resulting accuracy reflects the dendritic dynamics or the
particular sites that happened to be drawn is not answerable from that one
number. Refitting the readout is pure linear algebra, though, so the question can
be settled from the saved state matrices alone: draw a subset of the recorded
sites, refit, reclassify, and repeat.

The horizontal axis is therefore the number of sites feeding the readout and the
vertical axis is accuracy, with a spread over subsets at each size.

Subsetting one run holds the dynamics fixed, so it measures only how much the
choice of readout sites matters. It says nothing about how much the accuracy
would move had the synapses landed elsewhere, which is a separate and larger
source of variation. Passing several runs — a seed sweep, named individually or
by a glob over a multirun — separates the two: each run is reduced to its mean
at every size, and the spread of those means is reported alongside the spread
within a run.

Following SPEC_NumericalCode the computation never draws: ``mode=compute`` writes
``curves.npz`` and ``results.json``, and ``mode=plot`` reads them back, so the
figure can be reworked without recomputing.

    uv run readout_subset.py run_dir=outputs/2026-09-24/15-16-45
    uv run readout_subset.py 'run_dir=multirun/2026-09-25/11-29-26/*'
    uv run readout_subset.py mode=plot curves_path=outputs/.../data/curves.npz
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

STATES_FILENAME = os.path.join("data", "reservoir_states.npz")
RESULTS_FILENAME = os.path.join("data", "classification_results.npz")

# Categorical slots 1 and 2 of the reference palette, light mode.
COLOR_TEST = "#2a78d6"
COLOR_TRAIN = "#eb6834"


def load_states(run_dir):
    """Read the state matrices a main.py run saved, plus its reference accuracy.

    The reference accuracy is recovered from the run's confusion matrix when it is
    present. It is what the simulation itself reported using every recording site,
    so reproducing it from the state matrices confirms that the refit here agrees
    with the pipeline it is standing in for.
    """
    states_path = os.path.join(run_dir, STATES_FILENAME)
    if not os.path.exists(states_path):
        raise FileNotFoundError(
            f"{states_path} not found. Only runs made after reservoir_states.npz "
            f"was introduced carry the state matrices needed here."
        )

    with np.load(states_path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    logger.info(f"Loaded {states_path}")

    # The readout was fitted on whichever quantity record_target names. Older files
    # carried that copy separately; it is now read from the per-quantity matrices.
    if "train_state_vars" not in data:
        quantity = {"potential": "potential",
                    "calcium_acum": "calcium"}[str(data["record_target"])]
        data["train_state_vars"] = data[f"train_states_{quantity}"]
        data["test_state_vars"] = data[f"test_states_{quantity}"]

    reference_accuracy = None
    results_path = os.path.join(run_dir, RESULTS_FILENAME)
    if os.path.exists(results_path):
        with np.load(results_path, allow_pickle=False) as npz:
            confusion_matrix = npz["confusion_matrix"]
        total = confusion_matrix.sum()
        if total > 0:
            reference_accuracy = float(np.trace(confusion_matrix) / total)
            logger.info(f"Run reported test accuracy {reference_accuracy:.4f} (all sites)")

    return data, reference_accuracy


def trial_slices(bin_counts):
    """Convert per-trial bin counts into (start, stop) row ranges."""
    slices = []
    start = 0
    for count in bin_counts:
        stop = start + int(count)
        slices.append((start, stop))
        start = stop
    return slices


def fit_readout(state_vars, target, reg):
    """Ridge-fit the readout weights, mirroring neuronalreservoir.optimize().

    The identity is sized to the number of columns actually supplied rather than
    to num_states, which is what makes the fit valid for a subset of the sites.
    """
    num_features = state_vars.shape[1]
    gram = state_vars.T @ state_vars + reg * np.eye(num_features)
    return np.linalg.inv(gram) @ state_vars.T @ target


def classify(state_vars, weights, slices):
    """Label each trial by the output neuron that wins the most of its bins.

    This reproduces neuronalreservoir_classification.classify(): argmax per bin,
    then the most frequent winner over the trial, ties going to the lowest label
    because np.unique returns its values sorted.
    """
    predicted = np.empty(len(slices))
    for trial_idx, (start, stop) in enumerate(slices):
        output = state_vars[start:stop, :] @ weights
        winners = np.argmax(output, axis=1)
        labels, counts = np.unique(winners, return_counts=True)
        predicted[trial_idx] = labels[np.argmax(counts)]
    return predicted


def accuracy_for_columns(data, columns, reg, train_slices, test_slices):
    """Refit on the given recording sites and score training and test trials."""
    weights = fit_readout(data["train_state_vars"][:, columns],
                          data["trainingdata_target"], reg)

    train_predicted = classify(data["train_state_vars"][:, columns], weights, train_slices)
    test_predicted = classify(data["test_state_vars"][:, columns], weights, test_slices)

    train_accuracy = float(np.mean(train_predicted == data["train_label"]))
    test_accuracy = float(np.mean(test_predicted == data["test_label"]))
    return train_accuracy, test_accuracy


def sweep_subset_sizes(data, subset_sizes, num_draws, seed, reg):
    """Score random subsets of recording sites across the requested sizes.

    Results come back in long form — one row per draw — because the number of
    draws differs between sizes: using every site admits a single subset, so that
    size is evaluated once instead of num_draws times.
    """
    num_sites = data["train_state_vars"].shape[1]
    train_size = int(data["train_dataset_size"])
    test_size = int(data["test_dataset_size"])
    bin_counts = data["len_data"]

    train_slices = trial_slices(bin_counts[:train_size])
    test_slices = trial_slices(bin_counts[train_size:train_size + test_size])

    # The trial boundaries come from len_data while the rows come from the
    # concatenated per-trial states. If the two disagree the accuracies would still
    # be produced, just silently against misaligned trials, so check rather than
    # trust.
    for name, slices, matrix, labels in (
            ("training", train_slices, data["train_state_vars"], data["train_label"]),
            ("test", test_slices, data["test_state_vars"], data["test_label"])):
        if slices[-1][1] != matrix.shape[0]:
            raise ValueError(
                f"{name} trial boundaries cover {slices[-1][1]} rows but the state "
                f"matrix has {matrix.shape[0]}")
        if len(slices) != len(labels):
            raise ValueError(
                f"{name} set has {len(slices)} trials but {len(labels)} labels")

    usable_sizes = sorted({int(size) for size in subset_sizes if 0 < int(size) <= num_sites})
    dropped = sorted({int(size) for size in subset_sizes} - set(usable_sizes))
    if dropped:
        logger.warning(f"Ignoring subset sizes outside 1..{num_sites}: {dropped}")

    rng = np.random.default_rng(seed)
    rows = {"size": [], "draw": [], "train_accuracy": [], "test_accuracy": []}

    for size in usable_sizes:
        draws = 1 if size == num_sites else num_draws
        for draw in range(draws):
            columns = (np.arange(num_sites) if size == num_sites
                       else rng.choice(num_sites, size=size, replace=False))
            train_accuracy, test_accuracy = accuracy_for_columns(
                data, columns, reg, train_slices, test_slices)
            rows["size"].append(size)
            rows["draw"].append(draw)
            rows["train_accuracy"].append(train_accuracy)
            rows["test_accuracy"].append(test_accuracy)
        scores = np.array(rows["test_accuracy"][-draws:])
        logger.info(f"sites={size:4d}  draws={draws:4d}  "
                    f"test accuracy mean={scores.mean():.4f} sd={scores.std():.4f} "
                    f"min={scores.min():.4f} max={scores.max():.4f}")

    curves = {key: np.array(value) for key, value in rows.items()}
    curves["num_sites"] = np.array(num_sites)
    return curves


def summarise(curves, reference_accuracy):
    """Reduce the per-draw scores to the numbers worth reading off directly."""
    summary = {"num_sites": int(curves["num_sites"]), "sizes": []}

    for size in sorted(set(curves["size"].tolist())):
        mask = curves["size"] == size
        test_scores = curves["test_accuracy"][mask]
        train_scores = curves["train_accuracy"][mask]
        # Both series are summarised the same way; reporting a spread for only one
        # of them is what made the earlier figure misleading.
        summary["sizes"].append({
            "num_readout_sites": int(size),
            "num_draws": int(mask.sum()),
            "test_accuracy_mean": float(test_scores.mean()),
            "test_accuracy_sd": float(test_scores.std()),
            "test_accuracy_min": float(test_scores.min()),
            "test_accuracy_max": float(test_scores.max()),
            "train_accuracy_mean": float(train_scores.mean()),
            "train_accuracy_sd": float(train_scores.std()),
            "train_accuracy_min": float(train_scores.min()),
            "train_accuracy_max": float(train_scores.max()),
        })

    full = [entry for entry in summary["sizes"]
            if entry["num_readout_sites"] == summary["num_sites"]]
    if full:
        summary["all_sites_test_accuracy"] = full[0]["test_accuracy_mean"]
    if reference_accuracy is not None:
        summary["run_reported_test_accuracy"] = reference_accuracy
        if full:
            summary["refit_minus_reported"] = (
                full[0]["test_accuracy_mean"] - reference_accuracy)

    return summary


def resolve_run_dirs(run_dir, original_cwd):
    """Expand the run_dir setting into a list of directories.

    A single path, a list of paths, or a glob such as
    multirun/2026-09-25/11-29-26/* are all accepted, so a sweep's jobs can be
    analysed together without naming each one.
    """
    if run_dir is None:
        raise ValueError("run_dir is required in compute mode "
                         "(e.g. run_dir=outputs/2026-09-24/15-16-45, or a glob over "
                         "a multirun's job directories)")

    patterns = [run_dir] if isinstance(run_dir, str) else list(run_dir)
    resolved = []
    for pattern in patterns:
        if not os.path.isabs(pattern):
            pattern = os.path.join(original_cwd, pattern)
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        resolved.extend(match for match in matches
                        if os.path.exists(os.path.join(match, STATES_FILENAME)))

    if not resolved:
        raise FileNotFoundError(f"No run directory holding {STATES_FILENAME} matched {patterns}")
    return resolved


def run_compute(cfg, original_cwd):
    run_dirs = resolve_run_dirs(cfg.run_dir, original_cwd)
    logger.info(f"Analysing {len(run_dirs)} run(s)")

    per_run, summaries = [], []
    for run_index, run_dir in enumerate(run_dirs):
        data, reference_accuracy = load_states(run_dir)
        reg = float(data["reg"]) if cfg.reg is None else float(cfg.reg)

        curves = sweep_subset_sizes(data, cfg.subset_sizes, int(cfg.num_draws),
                                    int(cfg.seed) + run_index, reg)
        curves["run"] = np.full(curves["size"].shape, run_index)
        per_run.append(curves)

        summary = summarise(curves, reference_accuracy)
        summary["source_run"] = run_dir
        summary["reg"] = reg
        summaries.append(summary)
        if "refit_minus_reported" in summary:
            logger.info(f"  [{run_index}] refit on all sites differs from the run's own "
                        f"accuracy by {summary['refit_minus_reported']:+.4f}")

    num_sites = {int(c["num_sites"]) for c in per_run}
    if len(num_sites) > 1:
        raise ValueError(f"Runs record different numbers of sites ({sorted(num_sites)}); "
                         f"their curves do not share an axis")

    combined = {key: np.concatenate([c[key] for c in per_run])
                for key in ("size", "draw", "run", "train_accuracy", "test_accuracy")}
    combined["num_sites"] = per_run[0]["num_sites"]

    os.makedirs(cfg.data_dir, exist_ok=True)
    curves_path = os.path.join(cfg.data_dir, "curves.npz")
    np.savez_compressed(curves_path, **combined,
                        source_runs=np.array(run_dirs),
                        reg=np.array(summaries[0]["reg"]))
    logger.info(f"Saved {curves_path}")

    results = {"num_runs": len(run_dirs), "runs": summaries,
               "across_runs": summarise_across_runs(combined)}
    results_path = os.path.join(cfg.data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {results_path}")


def summarise_across_runs(curves):
    """Separate the two things that make accuracy vary.

    Within a run the dynamics are fixed and only the choice of readout sites
    changes; across runs the synapse placement and the input differ too. Each run
    is first reduced to its mean at every size, and the spread of those means is
    what the across-run figures show.
    """
    rows = []
    for size in sorted(set(curves["size"].tolist())):
        at_size = curves["size"] == size
        run_means = np.array([curves["test_accuracy"][at_size & (curves["run"] == r)].mean()
                              for r in sorted(set(curves["run"][at_size].tolist()))])
        train_means = np.array([curves["train_accuracy"][at_size & (curves["run"] == r)].mean()
                                for r in sorted(set(curves["run"][at_size].tolist()))])
        within = np.mean([curves["test_accuracy"][at_size & (curves["run"] == r)].std()
                          for r in sorted(set(curves["run"][at_size].tolist()))])
        rows.append({
            "num_readout_sites": int(size),
            "num_runs": int(len(run_means)),
            "test_accuracy_mean": float(run_means.mean()),
            "test_accuracy_sd_across_runs": float(run_means.std()),
            "test_accuracy_sd_within_run": float(within),
            "test_accuracy_min": float(run_means.min()),
            "test_accuracy_max": float(run_means.max()),
            "train_accuracy_mean": float(train_means.mean()),
            "train_accuracy_sd_across_runs": float(train_means.std()),
        })
    return rows


def run_plot(cfg, original_cwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = cfg.curves_path
    if paths is None:
        raise ValueError("curves_path is required in plot mode")
    if isinstance(paths, str):
        paths = [paths]

    os.makedirs(cfg.figure_dir, exist_ok=True)

    # With several runs the two sources of variation can be told apart, and which of
    # them dominates turns out to depend on the number of sites, so they get a panel
    # of their own rather than a sentence in the caption.
    with np.load(paths[0] if os.path.isabs(paths[0]) else os.path.join(original_cwd, paths[0]),
                 allow_pickle=False) as probe:
        multi_run = "run" in probe.files and len(set(probe["run"].tolist())) > 1

    if multi_run:
        fig, (ax, ax_sd) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                                        gridspec_kw={"height_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax_sd = None

    for path in paths:
        resolved = path if os.path.isabs(path) else os.path.join(original_cwd, path)
        with np.load(resolved, allow_pickle=False) as npz:
            curves = {key: npz[key] for key in npz.files}

        sizes = np.array(sorted(set(curves["size"].tolist())))

        # Accuracy varies for two reasons and they are not the same size. Within one
        # simulation only the choice of readout sites changes; across simulations the
        # synapse placement and the input change too. When several runs are present
        # each is reduced to its mean first, and the band is the spread of those
        # means — the uncertainty of the condition, not of one draw within it.
        runs = curves["run"] if "run" in curves else np.zeros(curves["size"].shape, dtype=int)
        run_ids = sorted(set(runs.tolist()))
        across_runs = len(run_ids) > 1
        spread_label = ("± 1 sd across simulations" if across_runs
                        else "± 1 sd across random subsets")

        # Both series get the same treatment; showing a spread for only one of them
        # invites reading the other as having none.
        series = (("test_accuracy", "Test", COLOR_TEST, "-"),
                  ("train_accuracy", "Training", COLOR_TRAIN, "--"))

        for key, label, color, linestyle in series:
            mean, low, high = [], [], []
            for size in sizes:
                at_size = curves["size"] == size
                if across_runs:
                    scores = np.array([curves[key][at_size & (runs == r)].mean()
                                       for r in run_ids if (at_size & (runs == r)).any()])
                else:
                    scores = curves[key][at_size]
                centre, spread = scores.mean(), scores.std()
                mean.append(centre)
                low.append(centre - spread)
                high.append(centre + spread)

            ax.fill_between(sizes, low, high, color=color, alpha=0.18,
                            linewidth=0, zorder=2)
            ax.plot(sizes, mean, color=color, linewidth=2, linestyle=linestyle,
                    marker="o", markersize=5, zorder=3, label=label)

        if ax_sd is not None:
            across, within = [], []
            for size in sizes:
                at_size = curves["size"] == size
                per_run = [curves["test_accuracy"][at_size & (runs == r)]
                           for r in run_ids if (at_size & (runs == r)).any()]
                across.append(np.array([scores.mean() for scores in per_run]).std())
                within.append(np.mean([scores.std() for scores in per_run]))

            ax_sd.plot(sizes, across, color=COLOR_TEST, linewidth=2, marker="o",
                       markersize=5, label="Across simulations")
            ax_sd.plot(sizes, within, color=COLOR_TEST, linewidth=2, marker="o",
                       markersize=5, linestyle=":", alpha=0.65,
                       label="Across subsets, within one simulation")

    ax.set_ylabel("Accuracy")
    ax.set_title("Readout accuracy vs. number of recording sites", pad=20)
    ax.text(0.5, 1.03, f"Line: mean   ·   Band: {spread_label}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#52514e")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    if ax_sd is not None:
        ax_sd.set_ylabel("sd of test accuracy")
        ax_sd.set_ylim(bottom=0)
        ax_sd.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax_sd.spines[["top", "right"]].set_visible(False)
        ax_sd.legend(frameon=False, loc="upper left", fontsize=9)

    bottom_axis = ax_sd if ax_sd is not None else ax
    bottom_axis.set_xlabel("Number of recording sites feeding the readout")

    fig.tight_layout()
    figure_path = os.path.join(cfg.figure_dir, "readout_subset_accuracy.png")
    fig.savefig(figure_path, dpi=300)
    logger.info(f"Saved {figure_path}")
    if cfg.show:
        plt.show()
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="readout_subset")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    logger.info("--- readout_subset configuration ---")
    logger.info(OmegaConf.to_yaml(cfg))

    original_cwd = get_original_cwd()
    if cfg.mode == "compute":
        run_compute(cfg, original_cwd)
    elif cfg.mode == "plot":
        run_plot(cfg, original_cwd)
    else:
        raise ValueError(f"unknown mode: {cfg.mode} (expected compute or plot)")


if __name__ == "__main__":
    main()
