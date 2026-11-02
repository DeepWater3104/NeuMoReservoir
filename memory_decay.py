"""Whether the class of an earlier trial can be read from the state at one instant.

For a delay d, every time t whose earlier time t - d lies in a different trial is
labelled with the class of the trial running at t - d, and a readout refitted for
that delay is asked to name it from the state at t alone — one row, one answer.

Nothing is aggregated over time. A vote over a window lets a small share of rows
that still carry the trial's own input decide the answer, so the decay it shows is
set by how much of the window overlaps the trial rather than by what the cell
keeps. Restricting to times outside the trial whose class is asked for removes the
trial's own input altogether; whatever is read there is held in the state.

The readout is refitted at every delay. The readout fitted on the current class
would only show the previous class as a mistake about the present one; refitting
asks whether the earlier class is held in the state in a form a readout can use.

Only the test trials are used, split into folds that each hold one trial of every
class, and each time belongs to the fold of the trial whose class it is asked for.
The training trials run through the classes in a fixed order, so a readout of the
previous class fitted there could learn the order instead of anything held in the
state. Chance is measured rather than assumed: the class sequence is shuffled and
the same computation repeated, which also carries any bias of the folding itself.

For a source trial m the times used form one contiguous stretch, starting
max(d, T) after its onset and lasting min(d, T), T being the trial length. Fitting
needs only X^T X and X^T Y over those stretches, so they are accumulated once per
delay and fold over every recording site and the subset is indexed out of them; at
check_delay_ms the first check_draws draws are refitted with
readout_subset.fit_readout itself and compared.

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

from readout_subset import (COLOR_TEST, HYDRA_CONFIG_FILENAME, fit_readout, load_run,
                            resolve_run_dirs)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

COLOR_REFERENCE = "#52514e"


def continuous_test_series(data):
    """Join the test trials into one time series without repeating the boundary.

    Each trial file holds both of its end points, so the last sample of a trial is
    the first sample of the next one. Dropping the repeat makes the series uniform
    in time.
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
    period = trial_rows - 1
    onsets = np.arange(test_size) * period
    return series, onsets, period


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


def stretches(onsets, period, shift, num_rows):
    """Start and length of the times whose class is that of each source trial.

    A time t is used when t - shift falls in trial m but t itself does not, which
    leaves t in [onset_m + max(shift, period), onset_m + shift + period). Trials
    whose stretch would run past the end of the recording are marked unusable.
    """
    starts = onsets + max(shift, period)
    length = min(shift, period)
    usable = starts + length <= num_rows
    return starts, length, usable


def accumulate_delay(series, starts, length, usable, labels, folds, num_folds, num_classes):
    """X^T X and X^T Y of every fold at one delay, over all recording sites."""
    num_sites = series.shape[1]
    gram = np.zeros((num_folds, num_sites, num_sites))
    cross = np.zeros((num_folds, num_sites, num_classes))
    for m in np.flatnonzero(usable):
        stretch = np.asarray(series[starts[m]:starts[m] + length], dtype=np.float64)
        gram[folds[m]] += stretch.T @ stretch
        # Every time in the stretch is labelled with trial m's class, so its product
        # with the one-hot target is the column sum placed in that class.
        cross[folds[m], :, labels[m]] += stretch.sum(axis=0)
    return gram, cross


def accumulate_all(series, onsets, period, shifts, labels, folds, num_folds, num_classes,
                   n_jobs):
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(accumulate_delay)(series, *stretches(onsets, period, shift, series.shape[0]),
                                  labels, folds, num_folds, num_classes)
        for shift in shifts)
    return np.stack([g for g, _ in parts]), np.stack([c for _, c in parts])


def solve(gram, cross, fold, reg):
    """Weights fitted on every fold but one, from the per-fold sums."""
    return np.linalg.inv(gram.sum(axis=0) - gram[fold] + reg * np.eye(gram.shape[-1])) \
        @ (cross.sum(axis=0) - cross[fold])


def score_draw(series, onsets, period, shifts, labels, folds, fold_gram, fold_cross,
               columns, reg):
    """Correct and scored times at every delay for one subset of sites."""
    num_folds = fold_gram.shape[1]
    correct = np.zeros(len(shifts), dtype=np.int64)
    scored = np.zeros(len(shifts), dtype=np.int64)
    # Copied out once per draw so that each stretch is a contiguous slice; indexing
    # the columns out of every stretch reads all sites of each row.
    projected = np.ascontiguousarray(series[:, columns])
    every_fold = np.arange(num_folds)
    for i, shift in enumerate(shifts):
        starts, length, usable = stretches(onsets, period, shift, series.shape[0])
        gram = fold_gram[i][np.ix_(every_fold, columns, columns)]
        cross = fold_cross[i][:, columns, :]
        for fold in range(num_folds):
            weights = solve(gram, cross, fold, reg)
            for m in np.flatnonzero(usable & (folds == fold)):
                answers = np.argmax(projected[starts[m]:starts[m] + length] @ weights, axis=1)
                correct[i] += int(np.count_nonzero(answers == labels[m]))
                scored[i] += length
    return correct, scored


def score_all(series, onsets, period, shifts, labels, folds, fold_gram, fold_cross,
              columns, reg, n_jobs, verbose=0):
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
        delayed(score_draw)(series, onsets, period, shifts, labels, folds, fold_gram,
                            fold_cross, columns[d], reg)
        for d in range(len(columns)))
    correct = np.stack([c for c, _ in results], axis=1)
    scored = np.stack([s for _, s in results], axis=1)
    return correct, scored


def check_against_fit_readout(series, onsets, period, shift, labels, folds, num_folds,
                              num_classes, fold_gram, fold_cross, columns, reg):
    """Refit one delay with fit_readout and compare weights and per-time answers."""
    starts, length, usable = stretches(onsets, period, shift, series.shape[0])
    gram = fold_gram[np.ix_(np.arange(num_folds), columns, columns)]
    cross = fold_cross[:, columns, :]
    worst_weight_gap, differing = 0.0, 0
    for fold in range(num_folds):
        train = np.flatnonzero(usable & (folds != fold))
        states = np.concatenate([series[starts[m]:starts[m] + length][:, columns] for m in train])
        target = np.zeros((len(train) * length, num_classes))
        for i, m in enumerate(train):
            target[i * length:(i + 1) * length, labels[m]] = 1.0
        reference = fit_readout(states, target, reg)
        accumulated = solve(gram, cross, fold, reg)
        worst_weight_gap = max(worst_weight_gap, float(
            np.abs(reference - accumulated).max() / np.abs(reference).max()))
        for m in np.flatnonzero(usable & (folds == fold)):
            stretch = series[starts[m]:starts[m] + length][:, columns]
            differing += int(np.count_nonzero(np.argmax(stretch @ reference, axis=1)
                                              != np.argmax(stretch @ accumulated, axis=1)))
    return worst_weight_gap, differing


def _process_one_run(run_index, run_dir, cfg):
    from threadpoolctl import threadpool_limits

    data, _ = load_run(run_dir)
    reg = float(data["reg"]) if cfg.reg is None else float(cfg.reg)
    config = OmegaConf.load(os.path.join(run_dir, HYDRA_CONFIG_FILENAME))

    series, onsets, period = continuous_test_series(data)
    del data["test_state_vars"], data["train_state_vars"]
    labels = np.asarray(data["test_label"], dtype=int)
    num_classes = int(np.max(data["train_label"])) + 1
    num_sites = series.shape[1]
    num_folds = int(cfg.num_folds)
    folds = assign_folds(labels, num_folds)

    step_ms = float(config.task.pattern_duration_ms) / period
    delays_ms = np.arange(float(cfg.delay_step_ms), float(cfg.delay_max_ms) + float(cfg.delay_step_ms) / 2,
                          float(cfg.delay_step_ms))
    shifts = np.rint(delays_ms / step_ms).astype(int)
    if not np.allclose(shifts * step_ms, delays_ms):
        raise ValueError(f"delay step {cfg.delay_step_ms} ms is not a multiple of the "
                         f"sampling step {step_ms} ms")
    logger.info(f"{run_dir}: {len(labels)} test trials, {num_sites} sites, "
                f"{len(delays_ms)} delays of {cfg.delay_step_ms} ms, step {step_ms} ms")

    # Drawn exactly as readout_subset.py draws one subset size.
    rng = np.random.default_rng(int(cfg.seed) + run_index)
    columns = np.array([rng.choice(num_sites, size=int(cfg.subset_size), replace=False)
                        for _ in range(int(cfg.num_draws))])

    n_jobs = int(cfg.n_jobs)
    check_shift = int(np.rint(float(cfg.check_delay_ms) / step_ms))
    if check_shift not in shifts:
        raise ValueError(f"check_delay_ms={cfg.check_delay_ms} is not one of the delays")
    check_index = int(np.flatnonzero(shifts == check_shift)[0])

    # BLAS is held to one thread so that the worker threads do not oversubscribe.
    with threadpool_limits(limits=1):
        fold_gram, fold_cross = accumulate_all(series, onsets, period, shifts, labels, folds,
                                               num_folds, num_classes, n_jobs)
        logger.info("Accumulated X^T X and X^T Y for every delay and fold")

        checks = [check_against_fit_readout(series, onsets, period, check_shift, labels, folds,
                                            num_folds, num_classes, fold_gram[check_index],
                                            fold_cross[check_index], columns[d], reg)
                  for d in range(min(int(cfg.check_draws), len(columns)))]
        for d, (gap, differing) in enumerate(checks):
            logger.info(f"check draw {d} at {cfg.check_delay_ms} ms: largest weight gap relative "
                        f"to fit_readout {gap:.2e}, answers that differ {differing}")
        if any(differing for _, differing in checks):
            logger.warning("accumulated fit and fit_readout disagree on some answers")

        correct, scored = score_all(series, onsets, period, shifts, labels, folds, fold_gram,
                                    fold_cross, columns, reg, n_jobs, verbose=5)
        accuracy_per_draw = correct / scored
        logger.info("Scored the true class sequence")

        # Chance: shuffle the class sequence, rebuild the folds from the shuffled
        # classes and repeat, so that the folding is the same kind as above.
        shuffle_rng = np.random.default_rng(int(cfg.shuffle_seed) + run_index)
        shuffle_columns = columns[:int(cfg.shuffle_draws)]
        chance_per_draw = []
        for s in range(int(cfg.num_shuffles)):
            shuffled = shuffle_rng.permutation(labels)
            shuffled_folds = assign_folds(shuffled, num_folds)
            gram_s, cross_s = accumulate_all(series, onsets, period, shifts, shuffled,
                                             shuffled_folds, num_folds, num_classes, n_jobs)
            c, n = score_all(series, onsets, period, shifts, shuffled, shuffled_folds, gram_s,
                             cross_s, shuffle_columns, reg, n_jobs)
            chance_per_draw.append(c / n)
            del gram_s, cross_s
            logger.info(f"Scored shuffle {s + 1}/{cfg.num_shuffles}")
        chance_per_draw = np.stack(chance_per_draw)

    accuracy = accuracy_per_draw.mean(axis=1)
    chance = chance_per_draw.mean(axis=2)
    readable = accuracy > chance.max(axis=0)
    readable_delays = delays_ms[readable]
    logger.info(f"Accuracy at {delays_ms[0]:.0f} ms = {accuracy[0]:.4f} "
                f"(chance max {chance[:, 0].max():.4f}); delays above every shuffle: "
                f"{len(readable_delays)} of {len(delays_ms)}")

    sample_id = int(config.sample_id) if config.get("sample_id") is not None else run_index
    os.makedirs(cfg.data_dir, exist_ok=True)
    decay_path = os.path.join(cfg.data_dir, f"memory_decay_s{sample_id:03d}.npz")
    np.savez_compressed(decay_path, delays_ms=delays_ms, correct=correct, scored=scored,
                        accuracy_per_draw=accuracy_per_draw, accuracy=accuracy,
                        chance_per_draw=chance_per_draw, chance=chance, readable=readable,
                        test_label=labels.astype(np.int16), folds=folds.astype(np.int16),
                        columns=columns.astype(np.int16), source_run=run_dir, reg=reg,
                        sample_id=sample_id)
    logger.info(f"Saved {decay_path}")

    return {"source_run": run_dir, "sample_id": sample_id, "run_index": run_index, "reg": reg,
            "first_delay_ms": float(delays_ms[0]), "accuracy_at_first_delay": float(accuracy[0]),
            "chance_max_at_first_delay": float(chance[:, 0].max()),
            "num_delays_above_every_shuffle": int(readable.sum()),
            "delays_above_every_shuffle_ms": readable_delays.tolist(),
            "check_against_fit_readout": [{"draw": d, "delay_ms": float(cfg.check_delay_ms),
                                           "weight_gap": gap, "answers_differing": m}
                                          for d, (gap, m) in enumerate(checks)],
            "decay_path": os.path.abspath(decay_path)}


def run_compute(cfg, original_cwd):
    run_dirs = resolve_run_dirs(cfg.run_dir, original_cwd)
    logger.info(f"Analysing {len(run_dirs)} run(s)")
    # One run at a time: each already uses every core through its threads.
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

    delays_s = decay["delays_ms"] / 1000
    mean = decay["accuracy"]
    spread = decay["accuracy_per_draw"].std(axis=1)
    chance = decay["chance"]

    os.makedirs(cfg.figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(delays_s, chance.min(axis=0), chance.max(axis=0), color=COLOR_REFERENCE,
                    alpha=0.18, linewidth=0, zorder=1, label="Shuffled classes (range)")
    ax.plot(delays_s, chance.mean(axis=0), color=COLOR_REFERENCE, linewidth=1, linestyle="--",
            zorder=2, label="Shuffled classes (mean)")
    ax.fill_between(delays_s, mean - spread, mean + spread, color=COLOR_TEST, alpha=0.18,
                    linewidth=0, zorder=3)
    ax.plot(delays_s, mean, color=COLOR_TEST, linewidth=2, zorder=4, label="True classes")

    ax.set_xlabel("Delay d between the state and the class asked for (s)")
    ax.set_ylabel("Accuracy per time point")
    ax.set_title(f"Class of an earlier trial read from one instant, sample "
                 f"{int(decay['sample_id']):03d}", pad=20)
    ax.text(0.5, 1.03, "Line: mean   ·   Band: ± 1 sd across random subsets",
            transform=ax.transAxes, ha="center", fontsize=9, color=COLOR_REFERENCE)
    ax.set_xlim(0, delays_s[-1])
    ax.set_ylim(bottom=0)
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
