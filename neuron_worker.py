"""1ジョブぶんのNEURONシミュレーション本体を、使い捨てのサブプロセスとして実行するワーカー。

`main.py` (Hydra側のジョブ制御・並列数管理) から `subprocess` 経由で毎回新規プロセスとして
起動される。NEURON (`libnrnmech`) はプロセス内にC言語側のグローバル状態
(セクション管理テーブル等) を溜め込み、同一プロセスで多数のジョブを連続実行すると
シミュレーション速度が徐々に悪化する問題があるため、ジョブごとにOSプロセスを
使い捨てにすることで状態の持ち越しを断つ。
"""
import argparse
import json
import logging
import os

import numpy as np


def run(params: dict, original_cwd: str, is_multirun: bool) -> None:
    logger = logging.getLogger(__name__)

    seed = params.get('seed', 1234)
    prng = np.random.default_rng(seed)

    logger.info(f"Task name: {params['task']['name']}")
    logger.info(f"Random seed: {seed}")

    # A sample_id draws its own (mu, sigma) from a generator keyed on that id alone,
    # so the sweep covers the plane without a grid and stays extensible: adding ids
    # later leaves every earlier draw where it was. The generator is separate from
    # prng, so which point is drawn does not disturb the simulation.
    if params.get('sample_id') is not None:
        sample_prng = np.random.default_rng([params['sample_seed'], params['sample_id']])
        mean_low, mean_high = params['syn_loc_mean_range']
        std_low, std_high = params['syn_loc_std_range']
        params['syn_loc_mean'] = float(sample_prng.uniform(mean_low, mean_high))
        params['syn_loc_std'] = float(sample_prng.uniform(std_low, std_high))
        logger.info(f"sample_id {params['sample_id']}: "
                    f"syn_loc_mean={params['syn_loc_mean']:.2f}, "
                    f"syn_loc_std={params['syn_loc_std']:.2f}")

    # 1. Compile MOD files (Resolve path dynamically using original_cwd)
    from neuron_simulation import run_nrnivmodl
    cell_dir = os.path.join(original_cwd, "cells", str(params['cell_name']))
    run_nrnivmodl(cell_dir)

    # 2. Load Cell Model and NEURON environment
    from neuron import h as nrn
    from neuron.units import ms, mV
    from neuron_simulation import get_hoc_morph_for_emodel_folder, extract_template_name, check_line_in_file

    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(cell_dir)
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())

    template_name = extract_template_name(hoc_path.as_posix())

    # Instantiate the cell based on its template structure (checking for GID requirement)
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(nrn, template_name)(0, cell_dir + "morphology", morph_path.name)
    else:
        cell = getattr(nrn, template_name)(cell_dir + "morphology", morph_path.name)

    if (params['gcalbar_ratio'] is not None or
        params['gcanbar_ratio'] is not None or
        params['gcatbar_ratio'] is not None or
        params['gcakbar_ratio'] is not None or
        params['gslowcakbar_ratio'] is not None):
        from cell_modifier import Ca_Related_Channels_modifier
        Ca_Related_Channels_modifier(cell, params)

    # Branching logic based on the specific task type
    if params['task']['name'] == "random":
        from DataGenerator import RandomPattern_datagenerator
        if not params['time_integration']:
            params['task']['bin_width'] = nrn.dt

        datagenerator = RandomPattern_datagenerator(params['task'], prng)

        output = params['output']
        save_buffer = output['buffers']
        logger.info(f"Detailed buffer saving is {'ENABLED' if save_buffer else 'DISABLED'}.")

        # plot_timeseries reads the target and output that buffer_io_included adds, so
        # the figures are only possible when both flags are on.
        plot_buffer_figures = output['buffer_figures'] and output['buffer_io_included']
        if output['buffer_figures'] and not output['buffer_io_included']:
            logger.warning("output.buffer_figures needs output.buffer_io_included; no buffer figures will be drawn.")

        # Which trials get buffered is drawn from a generator of its own rather than
        # from prng. Drawing it from prng would advance the shared stream, so
        # switching buffering off would shift every later draw — synapse placement
        # included — and silently change the simulation an output flag is not
        # supposed to touch.
        params['batches_to_save_idx'] = []
        params['batches_to_save_mode'] = []
        if save_buffer:
            buffer_prng = np.random.default_rng([seed, 0xB0FFE2])

            # Register training indices to be saved later
            for data_idx in range(datagenerator.train_dataset_size):
                params['batches_to_save_idx'].append(data_idx)
                params['batches_to_save_mode'].append("training")

            # Select a subset of test indices (up to 60) for visualization/saving
            test_indices = range(datagenerator.test_dataset_size)
            num_test_samples = min(60, len(test_indices))
            selected_test_indices = buffer_prng.choice(test_indices, size=num_test_samples, replace=False)

            for data_idx in selected_test_indices:
                params['batches_to_save_idx'].append(data_idx)
                params['batches_to_save_mode'].append("test")

        from NeuronalReservoir_classification import neuronalreservoir_classification
        from Analysis import get_spike_timings
        neuronalreservoir = neuronalreservoir_classification(cell, prng, params)
        nrn.finitialize(-65 * mV)

        # Every requested quantity is binned at the same segments and over the same
        # intervals as the readout, so a later analysis can relate the dynamics at a
        # site to what the readout made of it. The readout itself is always fitted on
        # train_state_vars, whichever quantity record_target selects.
        quantities = list(output['reservoir_states_quantities']) if output['reservoir_states'] else []
        collected = {q: {"training": [], "test": []} for q in quantities}

        def collect_states(mode, interval_start, num_bins):
            for quantity in quantities:
                collected[quantity][mode].append(
                    neuronalreservoir.get_binned_states(
                        interval_start, num_bins, params['time_integration'],
                        rec_list=neuronalreservoir.rec_list_for(quantity)))

        logger.info("--- Start Training Data Simulation ---")

        from tqdm import tqdm
        # Start training data simulation loop
        current_time = 0.0
        for data_idx in tqdm(range(datagenerator.train_dataset_size), desc="Training Data Simulation", disable=is_multirun):
            spike_trains = datagenerator.get_spike_trains(data_idx, "train", params['time_integration'])

            # Calculate binning indices for the current simulation interval
            start_bin_idx = sum(datagenerator.len_data[:-1])
            end_bin_idx = sum(datagenerator.len_data) - 1
            num_bins = end_bin_idx - start_bin_idx + 1

            interval_start = (params['task']['bin_width'] * start_bin_idx)
            interval_end = (params['task']['bin_width'] * (end_bin_idx + 1))

            # Execute NEURON simulation and extract binned states
            neuronalreservoir.resister_spike_events(spike_trains)
            current_time += datagenerator.pattern_duration_ms
            neuronalreservoir.generate_dynamics(current_time)
            state_vars = neuronalreservoir.get_binned_states(interval_start, num_bins, params['time_integration'])

            # Store binned states for later optimization (readout training)
            shape_before_concatenate = np.shape(neuronalreservoir.train_state_vars)
            neuronalreservoir.train_state_vars = np.concatenate([neuronalreservoir.train_state_vars, state_vars], axis=0)
            collect_states("training", interval_start, num_bins)

            # Save raw simulation data to buffer if index matches selected batches
            if save_buffer:
                if (data_idx, "training") in zip(neuronalreservoir.batches_to_save_idx, neuronalreservoir.batches_to_save_mode):
                    neuronalreservoir.save_to_buffer("training", data_idx, spike_trains, datagenerator, output['buffer_io_included'])

            # Accumulate spike timings for analysis
            v_rec_array = np.array(neuronalreservoir.Vm_at_soma)
            t_rec_array = np.array(neuronalreservoir.t_rec.to_python())
            neuronalreservoir.spike_timings = neuronalreservoir.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-30)

            nrn.frecord_init()

        logger.info("--- End Training Data Simulation ---")

        # Train the readout weights based on simulated reservoir states
        neuronalreservoir.optimize(neuronalreservoir.train_state_vars, datagenerator.trainingdata_target)
        neuronalreservoir.overwrite_buffer_after_optimized(datagenerator, output['buffer_io_included'])
        neuronalreservoir.save_buffer_all(plot_buffer_figures)

        logger.info("--- Start Test Data Simulation ---")

        # Start test data simulation loop
        for data_idx in tqdm(range(datagenerator.test_dataset_size), desc="Testing Data Simulation", disable=is_multirun):
            spike_trains = datagenerator.get_spike_trains(data_idx, "test", params['time_integration'])

            start_bin_idx = sum(datagenerator.len_data[:-1])
            end_bin_idx = sum(datagenerator.len_data) - 1
            num_bins = end_bin_idx - start_bin_idx + 1

            interval_start = (params['task']['bin_width'] * start_bin_idx)
            interval_end = (params['task']['bin_width'] * (end_bin_idx + 1))

            # Execute simulation for test data
            neuronalreservoir.resister_spike_events(spike_trains)
            current_time += datagenerator.pattern_duration_ms
            neuronalreservoir.generate_dynamics(current_time)
            state_vars = neuronalreservoir.get_binned_states(interval_start, num_bins, params['time_integration'])

            neuronalreservoir.test_state_vars = np.concatenate([neuronalreservoir.test_state_vars, state_vars], axis=0)
            collect_states("test", interval_start, num_bins)

            # Save test batch to buffer if index matches
            if save_buffer:
                if (data_idx, "test") in zip(neuronalreservoir.batches_to_save_idx, neuronalreservoir.batches_to_save_mode):
                    neuronalreservoir.save_to_buffer("test", data_idx, spike_trains, datagenerator, output['buffer_io_included'])
                    neuronalreservoir.save_buffer_single(len(neuronalreservoir.data_buffer) - 1, plot_buffer_figures)

            v_rec_array = np.array(neuronalreservoir.Vm_at_soma)
            t_rec_array = np.array(neuronalreservoir.t_rec.to_python())
            neuronalreservoir.spike_timings = neuronalreservoir.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-30)

            nrn.frecord_init()

        logger.info("--- End Test Data Simulation ---")

        logger.info("--- Start Saving Data and Images ---")

        confusion_matrix, confusion_matrix_axis = neuronalreservoir.get_classification_result("test", datagenerator)

        if output['confusion_matrix']:
            from NeuronalReservoir_classification import plot_confusion_matrix
            train_confusion_matrix, train_axis = neuronalreservoir.get_classification_result("training", datagenerator)
            plot_confusion_matrix(
                confusion_matrix=train_confusion_matrix,
                labels=train_axis,
                title='Classification Confusion Matrix (Training Data)',
                filename='./figure/confmat_train.png'
            )
            plot_confusion_matrix(
                confusion_matrix=confusion_matrix,
                labels=confusion_matrix_axis,
                title='Classification Confusion Matrix (Test Data)',
                filename='./figure/confmat_test.png'
            )
            # Save classification numerical results
            np.savez("./data/classification_results.npz",
                     confusion_matrix=confusion_matrix,
                     axis_labels=confusion_matrix_axis)
            logger.info("Saved classification results to ./data/classification_results.npz")

        if output['reservoir_states']:
            # Save the state matrices so the readout can be refitted offline on any
            # subset of the recording sites without rerunning the simulation. Columns
            # correspond to neuronalreservoir.record_segs, described here by section
            # name and distance from the soma, and every requested quantity shares
            # those columns. With time_integration false these hold the raw
            # per-timestep samples rather than bins.
            nrn.distance(0, 0.5, sec=neuronalreservoir.cell.soma[0])

            # float32 throughout: these matrices dominate the run's output, the extra
            # digits of a float64 trace are below anything the model resolves, and the
            # readout refits identically at single precision. The quantity named by
            # record_target is the one the readout was fitted on, so it is stored once
            # under its own name rather than again as a separate readout copy.
            states = {}
            for quantity in quantities:
                states[f"train_states_{quantity}"] = np.concatenate(
                    collected[quantity]["training"], axis=0).astype(np.float32)
                states[f"test_states_{quantity}"] = np.concatenate(
                    collected[quantity]["test"], axis=0).astype(np.float32)

            # Synapse positions are what intra/inter-branch sparsity is computed from.
            # They existed only in memory until now, so no earlier run can yield it.
            exc_segs = [syn.get_segment() for syn in neuronalreservoir.exc_syn_list]

            np.savez("./data/reservoir_states.npz",
                     **states,
                     quantities=np.array(quantities),
                     trainingdata_target=datagenerator.trainingdata_target.astype(np.float32),
                     train_label=datagenerator.train_label,
                     test_label=datagenerator.test_label,
                     len_data=np.array(datagenerator.len_data),
                     train_dataset_size=datagenerator.train_dataset_size,
                     test_dataset_size=datagenerator.test_dataset_size,
                     bin_width=params['task']['bin_width'],
                     time_integration=params['time_integration'],
                     reg=params['reg'],
                     record_target=params['record_target'],
                     syn_loc_condition=params['syn_loc_condition'],
                     syn_loc_mean=params['syn_loc_mean'],
                     syn_loc_std=params['syn_loc_std'],
                     sample_id=(-1 if params.get('sample_id') is None else params['sample_id']),
                     seed=seed,
                     seg_names=np.array([seg.sec.name() for seg in neuronalreservoir.record_segs]),
                     seg_x=np.array([seg.x for seg in neuronalreservoir.record_segs]),
                     seg_distance=np.array([nrn.distance(seg) for seg in neuronalreservoir.record_segs]),
                     syn_names=np.array([seg.sec.name() for seg in exc_segs]),
                     syn_x=np.array([seg.x for seg in exc_segs]),
                     syn_distance=np.array([nrn.distance(seg) for seg in exc_segs]))
            logger.info(f"Saved reservoir states ({', '.join(quantities)}) to ./data/reservoir_states.npz")

        if output['firing_rate']:
            from Analysis import get_firing_rate
            total_duration_sec = params['task']['pattern_duration_ms'] * (params['task']['num_outputs'] * (params['task']['n_repetition'] + 1)) * 0.001
            firing_rate = get_firing_rate(neuronalreservoir.spike_timings, total_duration_sec)
            with open('./data/firing_rate.txt', 'w') as f:
                f.write(f"{firing_rate}\n")
            logger.info("Saved firing rate to ./data/firing_rate.txt")

    elif params['task']['name'] == "sinwave":
        from DataGenerator import sin_datagenerator

        logger.info(f"--- Preparing Sine Wave Data (Frequency: {params['task']['freq']} Hz) ---")

        # Initialize sine wave data generator with specified frequency
        datagenerator = sin_datagenerator(
            params=params['task'],
            freq=params['task']['freq'],
            prng=prng
        )

        # Sync bin_width from generator to simulation parameters
        params['bin_width'] = datagenerator.bin_width
        logger.info(f"DataGenerator initialized for Sine Wave: {params['bin_width']} ms bins")

        from NeuronalReservoir_prediction import neuronalreservoir_prediction
        neuronalreservoir = neuronalreservoir_prediction(cell, prng, params)

        # Perform basic run for dynamics generation and readout optimization
        spike_trains = datagenerator.get_spike_trains()
        nrn.finitialize(-65 * mV)
        neuronalreservoir.resister_spike_events(spike_trains)
        num_bins = params['task']['len_transientdata'] + params['task']['len_trainingdata'] + params['task']['len_testdata']
        neuronalreservoir.generate_dynamics(num_bins * params['bin_width'])
        state_vars = neuronalreservoir.get_binned_states(0, num_bins)

        # Optimize weights using training segment of the binned states
        neuronalreservoir.optimize(state_vars[params['task']['len_transientdata']:params['task']['len_transientdata'] + params['task']['len_trainingdata'], :], datagenerator.trainingdata_target)
        output = neuronalreservoir.readout(state_vars)

        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error

        # 1. Define time axis and segments for visualization
        t_full = datagenerator.t
        len_trans = params['task']['len_transientdata']
        len_train = params['task']['len_trainingdata']
        len_test = params['task']['len_testdata']
        bin_w = params['bin_width']

        # 2. Create 4-panel plot for flow analysis
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # (1) Input Spike Raster
        if spike_trains.size > 0:
            axes[0].scatter(spike_trains[:, 0], spike_trains[:, 1],
                            marker='|', color='royalblue', s=20, linewidths=1.0)

            num_inputs = int(np.max(spike_trains[:, 1])) + 1 if spike_trains.size > 0 else 1
            axes[0].set_ylim(-0.5, num_inputs - 0.5)

        axes[0].set_ylabel('Input ID')
        axes[0].set_title('Input Spike Raster')

        # (2) Reservoir Membrane Potential (Internal Dynamics)
        if hasattr(neuronalreservoir, 'v_rec_list'):
            for v_rec in neuronalreservoir.v_rec_list:
                axes[1].plot(neuronalreservoir.t_rec, v_rec, alpha=0.5, linewidth=0.5)
        axes[1].set_ylabel('V [mV]')
        axes[1].set_title('Reservoir Membrane Potential')

        # (3) Binned States (Input features for readout)
        axes[2].plot(t_full, state_vars[:, :5], alpha=0.8)
        axes[2].set_ylabel('Binned State')
        axes[2].set_title('Readout Input (First 5 units)')

        # (4) Prediction Output vs Ground Truth
        target_all = datagenerator.get_targetdata()
        input_all = datagenerator.get_inputdata()[len_trans:]
        t_output = t_full[len_trans:]

        axes[3].plot(t_output, target_all, 'k-', alpha=0.3, label='Target')
        axes[3].plot(t_output, input_all, 'k-', alpha=0.3, label='Input')
        # Plot training phase predictions
        axes[3].plot(t_output[:len_train], output[len_trans:len_trans + len_train], 'b', label='Train Pred', linewidth=1)
        # Plot test phase predictions
        axes[3].plot(t_output[len_train:], output[len_trans + len_train:], 'r', label='Test Pred', linewidth=1.2)

        axes[3].axvline(x=t_output[len_train], color='green', linestyle='--', label='Train/Test Split')
        axes[3].set_ylabel('Output')
        axes[3].set_xlabel('Time [ms]')
        axes[3].legend(loc='upper right', fontsize='small')

        # 3. Adjust view window (Zoomed in for clarity)
        start_zoom = (len_trans + len_train // 2) * bin_w
        end_zoom = (len_trans + len_train + len_test) * bin_w
        for ax in axes:
            ax.set_xlim(start_zoom, end_zoom)
            ax.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        if params['output']['flow_analysis_figure']:
            plt.savefig("reservoir_flow_analysis.png")
        plt.close(fig)

        # Final metric validation
        mse_test = mean_squared_error(target_all[len_train:len_train + len_test], output[len_trans + len_train:])
        logger.info(f"Test MSE: {mse_test:.8f}")


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Hydra が既に main.log を用意しているので、同じファイルに追記して
    # ジョブの成果物（main.log）を分断しない。
    file_handler = logging.FileHandler("main.log", mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to a JSON file with params/original_cwd/is_multirun")
    args = parser.parse_args()

    with open(args.payload, "r") as f:
        payload = json.load(f)

    _setup_logging()
    run(payload["params"], payload["original_cwd"], payload["is_multirun"])
