"""How long the class of a trial stays readable once the input has moved on.

Accuracy is scored by fitting a readout on the states of a trial and letting it
vote, over the trial, for the class it sees. Sliding that window later by a lag L
asks a different question: with the window now partly or wholly inside the
following trials, can the states still tell which class trial n was? The answer as
a function of L is a decay curve, and the lag at which it has fallen halfway from
its value at L = 0 to chance is the half-life of the class memory.

The readout is refitted at every lag. Reusing the L = 0 readout would only show it
switching its vote to the next trial's class as the window enters it, which is set
by how much of the window still lies in trial n rather than by what the cell keeps.

Only the test trials are used, split into folds that each hold one trial of every
class. The training trials run through the classes in a fixed order, so a readout
fitted on shifted windows there could learn "the previous class is the current one
minus one" — a rule that fails on the shuffled test order and would make the decay
look faster than the memory is. The test order is random, so no such rule exists
to be learned.

Fitting with readout_subset.fit_readout at every lag, draw and fold would take
hours per run. The fit only needs X^T X and X^T Y, and both are sums over trials,
so they are accumulated once per lag and fold over every recording site and the
subset is indexed out of them. That is the same estimate; at L = 0 the first
check_draws draws are refitted with fit_readout itself and compared.

Following SPEC_NumericalCode the computation never draws: ``mode=compute`` writes
``memory_decay_sNNN.npz`` and ``results.json``, and ``mode=plot`` reads one back.

    uv run python memory_decay.py run_dir=multirun/2026-09-30/local-sweep-caracal/0
    uv run python memory_decay.py mode=plot decay_path=outputs/.../data/memory_decay_s000.npz
"""

import json
import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from readout_subset import (COLOR_TEST, HYDRA_CONFIG_FILENAME, classify, fit_readout,
                            load_run, resolve_run_dirs)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

COLOR_REFERENCE = "#52514e"


def continuous_test_series(data):
    """Join the test trials into one time series without repeating the boundary.

    Each trial file holds both of its end points, so the last sample of a trial is
    the first sample of the next one. Dropping the repeat makes the series uniform
    in time, and a window of one trial's length starting at a trial's onset is then
    exactly that trial's rows.
    """
    train_size = int(data["train_dataset_size"])
    test_size = int(data["test_dataset_size"])
    rows = data["trial_rows"][train_size:train_size + test_size]
    if len(set(rows.tolist())) != 1:
        raise ValueError(f"test trials differ in length: {sorted(set(rows.tolist()))}")
    trial_rows = int(rows[0])

    states = data["test_state_vars"]
    series = np.empty(((trial_rows - 1) * test_size + 1, states.shape[1]), dtype=states.dtype)
    for n in range(test_size):
        first = n * trial_rows
        if n + 1 < test_size and not np.array_equal(states[first + trial_rows - 1],
                                                    states[first + trial_rows]):
            raise ValueError(f"test trials {n} and {n + 1} do not share their boundary sample")
        series[n * (trial_rows - 1):(n + 1) * (trial_rows - 1)] = states[first:first + trial_rows - 1]
    series[-1] = states[-1]
    onsets = np.arange(test_size) * (trial_rows - 1)
    return series, onsets, trial_rows


def assign_folds(labels, num_folds):
    """Put the f-th trial of every class, in simulation order, into fold f."""
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        trials = np.flatnonzero(labels == label)
        if len(trials) != num_folds:
            raise ValueError(f"class {label} has {len(trials)} test trials; "
                             f"num_folds={num_folds} needs exactly that many")
        folds[trials] = np.arange(num_folds)
    return folds


def accumulate_lag(series, onsets, trial_rows, shift, usable, labels, folds,
                   num_folds, num_classes):
    """X^T X and X^T Y of every fold at one lag, over all recording sites."""
    num_sites = series.shape[1]
    gram = np.zeros((num_folds, num_sites, num_sites))
    cross = np.zeros((num_folds, num_sites, num_classes))
    for n in np.flatnonzero(usable):
        start = onsets[n] + shift
        window = np.asarray(series[start:start + trial_rows], dtype=np.float64)
        gram[folds[n]] += window.T @ window
        # The target is one-hot over every row of the trial, so its product with
        # the states is the column sum of the window placed in the trial's class.
        cross[folds[n], :, labels[n]] += window.sum(axis=0)
    return gram, cross


def score_draw(series, onsets, trial_rows, shifts, usable, labels, folds,
               fold_gram, fold_cross, columns, reg):
    """Predicted label of every test trial at every lag for one subset of sites."""
    num_folds = fold_gram.shape[1]
    predicted = np.full((len(shifts), len(labels)), -1, dtype=np.int16)
    for lag_index, shift in enumerate(shifts):
        gram = fold_gram[lag_index][np.ix_(np.arange(num_folds), columns, columns)]
        cross = fold_cross[lag_index][:, columns, :]
        gram_total, cross_total = gram.sum(axis=0), cross.sum(axis=0)
        for fold in range(num_folds):
            weights = np.linalg.inv(gram_total - gram[fold] + reg * np.eye(len(columns))) \
                @ (cross_total - cross[fold])
            for n in np.flatnonzero(usable[lag_index] & (folds == fold)):
                start = onsets[n] + shift
                window = series[start:start + trial_rows][:, columns]
                predicted[lag_index, n] = classify(window, weights, [(0, trial_rows)])[0]
    return predicted


def check_against_fit_readout(series, onsets, trial_rows, labels, folds, num_folds,
                              num_classes, fold_gram, fold_cross, columns, reg):
    """Refit at L = 0 with fit_readout and compare weights and predicted labels."""
    gram = fold_gram[0][np.ix_(np.arange(num_folds), columns, columns)]
    cross = fold_cross[0][:, columns, :]
    worst_weight_gap, mismatched = 0.0, 0
    for fold in range(num_folds):
        train = np.flatnonzero(folds != fold)
        states = np.concatenate([series[onsets[n]:onsets[n] + trial_rows][:, columns]
                                 for n in train])
        target = np.zeros((len(train) * trial_rows, num_classes))
        for i, n in enumerate(train):
            target[i * trial_rows:(i + 1) * trial_rows, labels[n]] = 1.0
        reference = fit_readout(states, target, reg)
        accumulated = np.linalg.inv(gram.sum(axis=0) - gram[fold] + reg * np.eye(len(columns))) \
            @ (cross.sum(axis=0) - cross[fold])
        worst_weight_gap = max(worst_weight_gap, float(
            np.abs(reference - accumulated).max() / np.abs(reference).max()))
        for n in np.flatnonzero(folds == fold):
            window = series[onsets[n]:onsets[n] + trial_rows][:, columns]
            mismatched += int(classify(window, reference, [(0, trial_rows)])[0]
                              != classify(window, accumulated, [(0, trial_rows)])[0])
    return worst_weight_gap, mismatched


def half_life(lags_ms, accuracy, chance, min_initial_accuracy):
    """First lag at which accuracy falls below halfway between L = 0 and chance.

    Returns (half_life_ms, threshold, status). The crossing is placed by a straight
    line between the two lags around it.
    """
    initial = float(accuracy[0])
    threshold = (initial + chance) / 2
    if initial <= min_initial_accuracy:
        return None, threshold, "initial_accuracy_too_low"
    below = np.flatnonzero(accuracy < threshold)
    if len(below) == 0:
        return None, threshold, "not_reached"
    i = int(below[0])
    fraction = (accuracy[i - 1] - threshold) / (accuracy[i - 1] - accuracy[i])
    return float(lags_ms[i - 1] + fraction * (lags_ms[i] - lags_ms[i - 1])), threshold, "reached"


def _process_one_run(run_index, run_dir, cfg):
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    data, _ = load_run(run_dir)
    reg = float(data["reg"]) if cfg.reg is None else float(cfg.reg)
    config = OmegaConf.load(os.path.join(run_dir, HYDRA_CONFIG_FILENAME))

    series, onsets, trial_rows = continuous_test_series(data)
    del data["test_state_vars"], data["train_state_vars"]
    labels = np.asarray(data["test_label"], dtype=int)
    num_classes = int(np.max(data["train_label"])) + 1
    num_sites = series.shape[1]
    folds = assign_folds(labels, int(cfg.num_folds))

    step_ms = float(config.task.pattern_duration_ms) / (trial_rows - 1)
    lags_ms = np.arange(0, float(cfg.lag_max_ms) + float(cfg.lag_step_ms) / 2,
                        float(cfg.lag_step_ms))
    shifts = np.rint(lags_ms / step_ms).astype(int)
    if not np.allclose(shifts * step_ms, lags_ms):
        raise ValueError(f"lag step {cfg.lag_step_ms} ms is not a multiple of the "
                         f"sampling step {step_ms} ms")
    # A trial whose window would run past the end of the recording is left out of
    # both fitting and scoring at that lag only.
    usable = onsets[None, :] + shifts[:, None] + trial_rows <= series.shape[0]
    logger.info(f"{run_dir}: {len(labels)} test trials, {num_sites} sites, "
                f"{len(lags_ms)} lags of {cfg.lag_step_ms} ms, step {step_ms} ms")

    # Drawn exactly as readout_subset.py draws one subset size, so the same seed and
    # run order give the same sites for the first draws.
    rng = np.random.default_rng(int(cfg.seed) + run_index)
    columns = np.array([rng.choice(num_sites, size=int(cfg.subset_size), replace=False)
                        for _ in range(int(cfg.num_draws))])

    n_jobs = int(cfg.n_jobs)
    # Threads share the series instead of copying it into each worker; BLAS is held
    # to one thread so that the workers do not oversubscribe the cores.
    with threadpool_limits(limits=1):
        accumulated = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(accumulate_lag)(series, onsets, trial_rows, shift, usable[i], labels,
                                    folds, int(cfg.num_folds), num_classes)
            for i, shift in enumerate(shifts))
        fold_gram = np.stack([g for g, _ in accumulated])
        fold_cross = np.stack([c for _, c in accumulated])
        del accumulated
        logger.info("Accumulated X^T X and X^T Y for every lag and fold")

        checks = [check_against_fit_readout(series, onsets, trial_rows, labels, folds,
                                            int(cfg.num_folds), num_classes, fold_gram,
                                            fold_cross, columns[d], reg)
                  for d in range(min(int(cfg.check_draws), len(columns)))]
        for d, (gap, mismatched) in enumerate(checks):
            logger.info(f"check draw {d}: largest weight gap relative to fit_readout "
                        f"{gap:.2e}, predicted labels that differ {mismatched}")
        if any(mismatched for _, mismatched in checks):
            logger.warning("accumulated fit and fit_readout disagree on some labels at L = 0")

        predicted = np.stack(Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(
            delayed(score_draw)(series, onsets, trial_rows, shifts, usable, labels, folds,
                                fold_gram, fold_cross, columns[d], reg)
            for d in range(len(columns))), axis=1)

    correct = (predicted == labels[None, None, :]) & usable[:, None, :]
    accuracy_per_draw = correct.sum(axis=2) / usable.sum(axis=1)[:, None]
    accuracy = accuracy_per_draw.mean(axis=1)
    chance = 1.0 / num_classes
    half_life_ms, threshold, status = half_life(lags_ms, accuracy, chance,
                                                float(cfg.min_initial_accuracy))
    logger.info(f"Accuracy(0) = {accuracy[0]:.4f}, threshold {threshold:.4f}, "
                f"half-life {half_life_ms} ms ({status})")

    sample_id = int(config.sample_id) if config.get("sample_id") is not None else run_index
    os.makedirs(cfg.data_dir, exist_ok=True)
    decay_path = os.path.join(cfg.data_dir, f"memory_decay_s{sample_id:03d}.npz")
    np.savez_compressed(decay_path, lags_ms=lags_ms, predicted=predicted, usable=usable,
                        test_label=labels.astype(np.int16), folds=folds.astype(np.int16),
                        columns=columns.astype(np.int16),
                        accuracy_per_draw=accuracy_per_draw, accuracy=accuracy,
                        threshold=threshold, chance=chance,
                        half_life_ms=np.nan if half_life_ms is None else half_life_ms,
                        status=status, source_run=run_dir, reg=reg, sample_id=sample_id)
    logger.info(f"Saved {decay_path}")

    return {"source_run": run_dir, "sample_id": sample_id, "run_index": run_index,
            "reg": reg, "accuracy_at_zero": float(accuracy[0]),
            "accuracy_at_zero_sd": float(accuracy_per_draw[0].std()),
            "threshold": threshold, "half_life_ms": half_life_ms, "status": status,
            "check_against_fit_readout": [{"draw": d, "weight_gap": gap, "labels_differing": m}
                                          for d, (gap, m) in enumerate(checks)],
            "decay_path": os.path.abspath(decay_path)}


def run_compute(cfg, original_cwd):
    run_dirs = resolve_run_dirs(cfg.run_dir, original_cwd)
    logger.info(f"Analysing {len(run_dirs)} run(s)")
    # One run at a time: each already uses every core through its threads, and
    # holding several runs' states at once would multiply the memory.
    summaries = [_process_one_run(i, d, cfg) for i, d in enumerate(run_dirs)]
    results_path = os.path.join(cfg.data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"num_runs": len(run_dirs), "runs": summaries}, f, indent=2)
    logger.info(f"Saved {results_path}")


def run_plot(cfg, original_cwd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cfg.decay_path is None:
        raise ValueError("decay_path is required in plot mode")
    path = cfg.decay_path if os.path.isabs(cfg.decay_path) else os.path.join(original_cwd, cfg.decay_path)
    with np.load(path, allow_pickle=False) as npz:
        decay = {key: npz[key] for key in npz.files}

    lags_s = decay["lags_ms"] / 1000
    mean = decay["accuracy"]
    spread = decay["accuracy_per_draw"].std(axis=1)
    half_life_ms = float(decay["half_life_ms"])

    os.makedirs(cfg.figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(lags_s, mean - spread, mean + spread, color=COLOR_TEST, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.plot(lags_s, mean, color=COLOR_TEST, linewidth=2, zorder=3, label="Test (cross-validated)")
    ax.axhline(float(decay["chance"]), color=COLOR_REFERENCE, linewidth=1, linestyle="--",
               label="Chance")
    ax.axhline(float(decay["threshold"]), color=COLOR_REFERENCE, linewidth=1, linestyle=":",
               label="Halfway to chance")
    if np.isfinite(half_life_ms):
        ax.axvline(half_life_ms / 1000, color=COLOR_REFERENCE, linewidth=1, linestyle=":")
        ax.annotate(f"half-life {half_life_ms:.0f} ms", xy=(half_life_ms / 1000, 1.0),
                    xytext=(4, -2), textcoords="offset points", fontsize=9,
                    color=COLOR_REFERENCE, va="top")

    ax.set_xlabel("Shift of the scoring window after the trial onset (s)")
    ax.set_ylabel("Accuracy for the shifted trial's class")
    ax.set_title(f"Class memory of sample {int(decay['sample_id']):03d}", pad=20)
    ax.text(0.5, 1.03, "Line: mean   ·   Band: ± 1 sd across random subsets",
            transform=ax.transAxes, ha="center", fontsize=9, color=COLOR_REFERENCE)
    ax.set_xlim(lags_s[0], lags_s[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    figure_path = os.path.join(cfg.figure_dir, f"memory_decay_s{int(decay['sample_id']):03d}.png")
    fig.savefig(figure_path, dpi=300)
    logger.info(f"Saved {figure_path}")
    if cfg.show:
        plt.show()
    plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="memory_decay")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd

    logger.info("--- memory_decay configuration ---")
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
