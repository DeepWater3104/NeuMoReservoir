"""Dynamics of the raw simulation traces, computed one target at a time.

mode=compute, target=
    rates               detect bAPs and dSpikes in the test trials and count them per
                        second; the somatic rate is read from firing_rate.txt
    effective_rank      Effective Rank of calcium in a short window around each bAP /
                        dSpike, per event and as a run mean
    separation          squared distance between every pair of test trials at every
                        1 ms, for calcium (raw and standardised per site) and inputs; SR
    compartment_spikes  threshold crossing times of every compartment in every test
                        trial, labelled bAP-related or not
    consistency         coincidence of compartment spikes between pairs of trials
                        (reads the compartment_spikes output; cheap, rerun to change Δ)
mode=plot, target=
    clusters            per-event Effective Rank, one panel per run ordered by mu
    separation          SR against accuracy and sparsity, time courses, input distances

Every invocation writes its outputs into the Hydra run directory created for it,
so no run overwrites another.

Usage:
    uv run python dynamics.py mode=compute target=rates 'run_dir=multirun/2026-09-30/local-sweep-caracal/*'
    uv run python dynamics.py mode=compute target=consistency 'spikes_path=[a/compartment_spikes.npz,b/compartment_spikes.npz]'
    uv run python dynamics.py mode=plot target=clusters 'events_path=[a/events.npz,b/events.npz]'
"""

import gc
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

READ_ERRORS = (zipfile.BadZipFile, ValueError, KeyError, IndexError, EOFError, OSError)


# ==============================================================================
# 1. Shared: runs, trials, spike detection
# ==============================================================================

def resolve_job_dirs(cfg):
    if not cfg.run_dir:
        raise ValueError("run_dir must be specified (e.g. 'multirun/.../*')")
    orig_cwd = hydra.utils.get_original_cwd()
    pattern = cfg.run_dir if os.path.isabs(cfg.run_dir) else os.path.join(orig_cwd, cfg.run_dir)
    job_dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not job_dirs:
        raise FileNotFoundError(f"No run directories matched pattern: {pattern}")
    return job_dirs


def run_meta(job_dir):
    """Identity of a run, from its Hydra snapshot."""
    meta = {"run_dir": job_dir, "job_id": os.path.basename(job_dir)}
    try:
        config = OmegaConf.load(os.path.join(job_dir, ".hydra", "config.yaml"))
        meta["syn_loc_mean"] = config.get("syn_loc_mean", None)
        meta["syn_loc_std"] = config.get("syn_loc_std", None)
        meta["seed"] = config.get("seed", None)
        meta["sample_id"] = config.get("sample_id", None)
    except OSError as e:
        logger.warning(f"Failed to read config in {job_dir}: {e}")
    return meta


def test_buffers(job_dir):
    """Paths of the test-trial buffers, in simulation order."""
    paths = []
    for path in sorted(glob.glob(os.path.join(job_dir, "data", "buffer*.npz"))):
        try:
            with np.load(path, allow_pickle=True) as npz:
                if str(npz["mode"]) == "test":
                    paths.append(path)
        except READ_ERRORS as e:
            logger.warning(f"Corrupted buffer file {path}: {e}")
    return paths


def load_trial(path):
    """Time, calcium and membrane potential of one trial, each (T, N) oriented."""
    with np.load(path, allow_pickle=True) as npz:
        t = npz["t_rec"]
        ca = npz["variables"][0]
        vm = npz["variables"][1]
        label = int(npz["TrueLabel"])
        index = int(npz["data_idx"])
    if ca.shape[1] > ca.shape[0]:
        ca, vm = ca.T, vm.T
    return t, ca, vm, label, index


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
        diffs = np.abs(all_local_spikes[:, None] - bap_indices[None, :])
        min_dist_to_bap = np.min(diffs, axis=1)
        dspike_indices = all_local_spikes[min_dist_to_bap >= exclusion_samples]

    num_unmerged = len(dspike_indices)
    merge_samples = int(np.round(merge_ms / dt))
    if merge_samples > 0 and len(dspike_indices) > 1:
        keep = np.r_[True, np.diff(dspike_indices) >= merge_samples]
        dspike_indices = dspike_indices[keep]

    return bap_indices, dspike_indices, num_unmerged


def map_runs(cfg, fn, **kwargs):
    """Apply fn to every run in parallel; returns the non-empty results and the run count."""
    job_dirs = [os.path.abspath(d) for d in resolve_job_dirs(cfg)]
    logger.info(f"{cfg.target}: {len(job_dirs)} runs")
    results = Parallel(n_jobs=cfg.n_jobs, verbose=10)(delayed(fn)(jd, **kwargs) for jd in job_dirs)
    return [r for r in results if r is not None], len(job_dirs)


def write_json(records, name):
    path = os.path.join(HydraConfig.get().runtime.output_dir, name)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved {len(records)} runs to: {path}")


# ==============================================================================
# 2. target=rates
# ==============================================================================

def rates_single_run(job_dir, v_threshold, exclusion_window_ms, merge_ms):
    meta = run_meta(job_dir)
    # Somatic firing rate as the simulation wrote it: somatic spikes over all
    # training and test trials divided by their total duration. Runs cut short
    # leave the file empty; they get None and drop out of the rates.
    rate_path = os.path.join(job_dir, "data", "firing_rate.txt")
    rate_text = open(rate_path).read().strip() if os.path.exists(rate_path) else ""
    meta["soma_rate"] = float(rate_text) if rate_text else None

    paths = test_buffers(job_dir)
    if not paths:
        logger.warning(f"No test buffers in {job_dir}")
        return None
    n_bap = n_dspike = n_unmerged = 0
    duration_s = 0.0
    for path in paths:
        try:
            t, _, vm, _, _ = load_trial(path)
        except READ_ERRORS as e:
            logger.warning(f"Corrupted buffer file {path}: {e}")
            continue
        bap, dspike, unmerged = get_spike_indices(vm, t, v_threshold, exclusion_window_ms, merge_ms)
        n_bap += len(bap)
        n_dspike += len(dspike)
        n_unmerged += unmerged
        duration_s += float(t[-1] - t[0]) / 1000.0
    meta.update({
        "total_duration_s": duration_s,
        "bap_count": n_bap,
        "dspike_count": n_dspike,
        "dspike_count_unmerged": n_unmerged,
        "bap_rate": n_bap / duration_s if duration_s > 0 else None,
        "dspike_rate": n_dspike / duration_s if duration_s > 0 else None,
    })
    return meta


def compute_rates(cfg):
    results, total = map_runs(cfg, rates_single_run, v_threshold=cfg.v_threshold,
                              exclusion_window_ms=cfg.exclusion_window_ms, merge_ms=cfg.merge_ms)
    write_json(results, "rates.json")
    logger.info(f"rates: {len(results)}/{total} runs")


# ==============================================================================
# 3. target=effective_rank
# ==============================================================================

def effective_rank_single_run(job_dir, window_ms, v_threshold, exclusion_window_ms, merge_ms):
    meta = run_meta(job_dir)
    paths = test_buffers(job_dir)
    if not paths:
        logger.warning(f"No test buffers in {job_dir}")
        return None
    pairs = {0: [], 1: []}  # kind 0 = bAP, 1 = dSpike
    # Per-event raw values: (kind, trial, t_ms, er_all, er_pc1_excluded)
    events = []
    for path in paths:
        trial = int(os.path.basename(path)[len("buffer"):-len(".npz")])
        try:
            t, ca, vm, _, _ = load_trial(path)
        except READ_ERRORS as e:
            logger.warning(f"Corrupted buffer file {path}: {e}")
            continue
        half_w = int(np.round(window_ms / float(t[1] - t[0]))) // 2
        bap, dspike, _ = get_spike_indices(vm, t, v_threshold, exclusion_window_ms, merge_ms)
        # PC1 is the global wave of the whole trial; the residual keeps the local dynamics.
        pca = PCA(n_components=1)
        ca_res = ca - pca.inverse_transform(pca.fit_transform(ca))
        for kind, indices in ((0, bap), (1, dspike)):
            for idx in indices:
                if idx < half_w or idx > ca.shape[0] - half_w:
                    continue
                window = slice(idx - half_w, idx + half_w)
                er_all = calculate_effective_rank(ca[window, :])
                er_res = calculate_effective_rank(ca_res[window, :])
                pairs[kind].append((er_all, er_res))
                events.append((kind, trial, float(t[idx]), er_all, er_res))
        del ca, vm, ca_res
        gc.collect()
    # The run mean is left undefined below two events.
    for kind, name in ((0, "bap"), (1, "dspike")):
        values = np.array(pairs[kind]).reshape(-1, 2)
        meta[f"{name}_er_events"] = len(values)
        meta[f"mean_{name}_er_orig"] = float(values[:, 0].mean()) if len(values) >= 2 else None
        meta[f"mean_{name}_er_res"] = float(values[:, 1].mean()) if len(values) >= 2 else None
    return meta, events


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


def compute_effective_rank(cfg):
    results, total = map_runs(cfg, effective_rank_single_run, window_ms=cfg.window_ms,
                              v_threshold=cfg.v_threshold, exclusion_window_ms=cfg.exclusion_window_ms,
                              merge_ms=cfg.merge_ms)
    _save_events(results, os.path.join(HydraConfig.get().runtime.output_dir, "events.npz"))
    write_json([meta for meta, _ in results], "effective_rank.json")
    logger.info(f"effective_rank: {len(results)}/{total} runs")


# ==============================================================================
# 4. target=compartment_spikes
# ==============================================================================

def compartment_spikes_single_run(job_dir, v_threshold, exclusion_window_ms):
    """Threshold crossings of every compartment in every test trial.

    A crossing is taken at the last sample below threshold, the convention of the
    bAP and dSpike detection. It is labelled bAP-related when it lies within
    exclusion_window_ms of a bAP of the same trial.
    """
    meta = run_meta(job_dir)
    paths = test_buffers(job_dir)
    if not paths:
        logger.warning(f"No test buffers in {job_dir}")
        return None
    trials, spikes, baps = [], [], []
    for path in paths:
        try:
            t, _, vm, label, index = load_trial(path)
        except READ_ERRORS as e:
            logger.warning(f"Corrupted buffer file {path}: {e}")
            continue
        t_rel = t - t[0]
        bap, _, _ = get_spike_indices(vm, t, v_threshold, exclusion_window_ms)
        step, comp = np.nonzero((vm[:-1, :] < v_threshold) & (vm[1:, :] >= v_threshold))
        t_spike = t_rel[step]
        t_bap = t_rel[bap]
        if len(t_bap):
            nearest = np.min(np.abs(t_spike[:, None] - t_bap[None, :]), axis=1)
            is_bap = nearest < exclusion_window_ms
        else:
            is_bap = np.zeros(len(t_spike), dtype=bool)
        trials.append((index, label, float(t_rel[-1])))
        spikes.append((index, comp, t_spike, is_bap))
        baps.append((index, t_bap))
        del vm
        gc.collect()
    return meta, trials, spikes, baps


def compute_compartment_spikes(cfg):
    results, total = map_runs(cfg, compartment_spikes_single_run, v_threshold=cfg.v_threshold,
                              exclusion_window_ms=cfg.exclusion_window_ms)
    cols = {k: [] for k in ("trial_run", "trial_index", "trial_label", "trial_duration_ms",
                            "spike_run", "spike_trial", "spike_comp", "spike_t_ms", "spike_is_bap",
                            "bap_run", "bap_trial", "bap_t_ms")}
    run_dir, mu, sigma = [], [], []
    for r, (meta, trials, spikes, baps) in enumerate(results):
        run_dir.append(meta["run_dir"])
        mu.append(np.nan if meta.get("syn_loc_mean") is None else float(meta["syn_loc_mean"]))
        sigma.append(np.nan if meta.get("syn_loc_std") is None else float(meta["syn_loc_std"]))
        for index, label, duration in trials:
            cols["trial_run"].append(r)
            cols["trial_index"].append(index)
            cols["trial_label"].append(label)
            cols["trial_duration_ms"].append(duration)
        for index, comp, t_spike, is_bap in spikes:
            cols["spike_run"].append(np.full(len(comp), r, dtype=np.int32))
            cols["spike_trial"].append(np.full(len(comp), index, dtype=np.int16))
            cols["spike_comp"].append(comp.astype(np.int16))
            cols["spike_t_ms"].append(t_spike.astype(np.float32))
            cols["spike_is_bap"].append(is_bap)
        for index, t_bap in baps:
            cols["bap_run"].append(np.full(len(t_bap), r, dtype=np.int32))
            cols["bap_trial"].append(np.full(len(t_bap), index, dtype=np.int16))
            cols["bap_t_ms"].append(t_bap.astype(np.float32))
    out = {k: (np.concatenate(v) if v and isinstance(v[0], np.ndarray) else np.array(v)) for k, v in cols.items()}
    path = os.path.join(HydraConfig.get().runtime.output_dir, "compartment_spikes.npz")
    np.savez(path, run_dir=np.array(run_dir), syn_loc_mean=np.array(mu), syn_loc_std=np.array(sigma), **out)
    logger.info(f"compartment_spikes: {len(results)}/{total} runs, {len(out['spike_run'])} spikes -> {path}")


# ==============================================================================
# 5. target=consistency
# ==============================================================================

def coincidence_factor(trial, comp, t, n_trials, delta_ms, duration_ms):
    """Γ for every pair of trials, counting coincidences only within the same compartment.

    trial: 0..n_trials-1, comp, t: one entry per spike. Returns an (n_trials, n_trials)
    matrix; NaN where neither trial of the pair has a spike.
    """
    n_comp = int(comp.max()) + 1 if len(comp) else 1
    counts = np.zeros((n_trials, n_comp))
    np.add.at(counts, (trial, comp), 1)
    # Chance coincidences of trial n's spikes with trial m's, compartment by compartment.
    expected = counts @ counts.T * (2.0 * delta_ms / duration_ms)

    coincident = np.zeros((n_trials, n_trials))
    if len(t):
        # One sorted key per spike: compartments are spaced far enough apart that a
        # ±Δ window never reaches into the next compartment.
        key = comp.astype(np.float64) * (duration_ms + 4.0 * delta_ms) + t
        order = np.argsort(key, kind="stable")
        key, trial = key[order], trial[order]
        lo = np.searchsorted(key, key - delta_ms, side="left")
        hi = np.searchsorted(key, key + delta_ms, side="right")
        width = hi - lo
        src = np.repeat(np.arange(len(key)), width)
        nbr = np.arange(width.sum()) - np.repeat(np.cumsum(width) - width, width) + np.repeat(lo, width)
        other = trial[nbr] != trial[src]
        # A spike of trial n counts once per trial m that has a spike within ±Δ.
        matched = np.unique(src[other] * n_trials + trial[nbr[other]])
        np.add.at(coincident, (trial[matched // n_trials], matched % n_trials), 1)

    c_bar = (coincident + coincident.T) / 2.0
    e_bar = (expected + expected.T) / 2.0
    k = counts.sum(axis=1)
    denominator = (k[:, None] + k[None, :]) / 2.0 - e_bar
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(denominator > 0, (c_bar - e_bar) / denominator, np.nan)
    return gamma


def _pair_mean(gamma, labels, same):
    i, j = np.triu_indices(len(labels), k=1)
    select = (labels[i] == labels[j]) == same
    values = gamma[i[select], j[select]]
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else None


def compute_consistency(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    paths = [cfg.spikes_path] if isinstance(cfg.spikes_path, str) else list(cfg.spikes_path)
    records, matrices = [], {}
    for p in paths:
        with np.load(p if os.path.isabs(p) else os.path.join(orig_cwd, p)) as npz:
            data = {k: npz[k] for k in npz.files}
        for r, run_dir in enumerate(data["run_dir"]):
            tsel = data["trial_run"] == r
            index = data["trial_index"][tsel]
            order = np.argsort(index)
            index, labels = index[order], data["trial_label"][tsel][order]
            duration = float(np.median(data["trial_duration_ms"][tsel]))
            position = {int(v): k for k, v in enumerate(index)}
            ssel = data["spike_run"] == r
            trial = np.array([position[int(v)] for v in data["spike_trial"][ssel]], dtype=np.int64)
            comp = data["spike_comp"][ssel].astype(np.int64)
            t = data["spike_t_ms"][ssel].astype(np.float64)
            is_bap = data["spike_is_bap"][ssel]
            rec = {"run_dir": str(run_dir), "syn_loc_mean": float(data["syn_loc_mean"][r]),
                   "syn_loc_std": float(data["syn_loc_std"][r]), "delta_ms": float(cfg.delta_ms),
                   "n_trials": int(len(index))}
            for name, select in (("bap", is_bap), ("dsp", ~is_bap)):
                gamma = coincidence_factor(trial[select], comp[select], t[select], len(index),
                                           cfg.delta_ms, duration)
                rec[f"gamma_within_{name}"] = _pair_mean(gamma, labels, same=True)
                rec[f"gamma_between_{name}"] = _pair_mean(gamma, labels, same=False)
                rec[f"n_spikes_{name}"] = int(select.sum())
                matrices[f"{len(records)}_{name}"] = gamma.astype(np.float32)
            records.append(rec)
    out_dir = HydraConfig.get().runtime.output_dir
    np.savez(os.path.join(out_dir, "consistency_pairs.npz"), run_dir=np.array([r["run_dir"] for r in records]), **matrices)
    write_json(records, "consistency.json")


VARIANTS = ("state_raw", "state_z", "input")


# ==============================================================================
# 6. target=separation (mode=compute) and mode=plot target=separation
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
# 6b. separation: single run
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


def separation_single_run(job_dir, step, tau_ms, pairs_dir):
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
# 6c. separation: compute
# ==============================================================================

def run_separation(cfg: DictConfig):
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
        delayed(separation_single_run)(jd, cfg.step, cfg.tau_ms, pairs_dir) for jd in job_dirs)
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
# 6d. separation: plot
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
    ax.scatter(x, y, s=18, color=color, edgecolors="k", linewidth=0.3)
    ax.set_title(f"r = {r:+.3f} (p = {p:.1e})", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.6)


def plot_separation(cfg: DictConfig):
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
# 7. mode=plot target=clusters
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

COMPUTE = {
    "rates": compute_rates,
    "effective_rank": compute_effective_rank,
    "separation": run_separation,
    "compartment_spikes": compute_compartment_spikes,
    "consistency": compute_consistency,
}
PLOT = {
    "clusters": run_clusters,
    "separation": plot_separation,
}


@hydra.main(config_path="conf", config_name="dynamics", version_base=None)
def main(cfg: DictConfig):
    table = {"compute": COMPUTE, "plot": PLOT}.get(cfg.mode)
    if table is None:
        raise ValueError(f"Unknown mode: {cfg.mode}. Must be 'compute' or 'plot'.")
    if cfg.target not in table:
        raise ValueError(f"Unknown target for mode={cfg.mode}: {cfg.target}. Must be one of {sorted(table)}.")
    table[cfg.target](cfg)


if __name__ == "__main__":
    main()
