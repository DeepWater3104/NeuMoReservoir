import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
import os

# Configure logging to output information to the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the reservoir simulation.
    Hydra handles the configuration loading and output directory management.
    """
    # Convert DictConfig to a standard Python dictionary for compatibility with existing classes
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    is_multirun = hydra_cfg.mode.name == "MULTIRUN"


    params = OmegaConf.to_container(cfg, resolve=True)
   
    # Ensure necessary directories exist for outputs
    os.makedirs("figure", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Log the resolved configuration for verification
    logger.info("--- Simulation Configuration ---")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize the pseudo-random number generator for reproducibility
    seed = params.get('seed', 1234)
    prng = np.random.default_rng(seed)
    
    # Log general experiment metadata
    logger.info(f"Task name: {params['task']['name']}")
    logger.info(f"Random seed: {seed}")
    
    # 1. Compile MOD files (Check environment context for multirun/single run)
    from neuron_simulation import run_nrnivmodl
    if is_multirun:
        cell_dir = "../../../../cells/" + str(params['cell_name'])
    else:
        cell_dir = "../../../cells/" + str(params['cell_name'])
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

    # Branching logic based on the specific task type
    if params['task']['name'] == "random":
        from DataGenerator import RandomPattern_datagenerator
        datagenerator = RandomPattern_datagenerator(params['task'], prng)

        save_buffer = params['task']['save_buffer']
        if not save_buffer:
            logger.info("Detailed buffer saving is DISABLED.")
        else:
            logger.info("Detailed buffer saving is ENABLED.")

            params['batches_to_save_idx']  = []
            params['batches_to_save_mode'] = []

            # Register training indices to be saved later
            for data_idx in range(datagenerator.train_dataset_size):
                params['batches_to_save_idx'].append(data_idx)
                params['batches_to_save_mode'].append("training")
            
            # Select a subset of test indices (up to 60) for visualization/saving
            test_indices = range(datagenerator.test_dataset_size)
            num_test_samples = min(60, len(test_indices)) 
            selected_test_indices = prng.choice(test_indices, size=num_test_samples, replace=False)
            
            for data_idx in selected_test_indices:
                params['batches_to_save_idx'].append(data_idx)
                params['batches_to_save_mode'].append("test")

        params['bin_width'] = datagenerator.bin_width

        from NeuronalReservoir_classification import neuronalreservoir_classification
        from Analysis import get_spike_timings
        neuronalreservoir = neuronalreservoir_classification(cell, prng, params)
        nrn.finitialize(-65 * mV)

        logger.info("--- Start Training Data Simulation ---")

        from tqdm import tqdm
        # Start training data simulation loop
        for data_idx in tqdm(range(datagenerator.train_dataset_size), desc="Training Data Simulation", disable=is_multirun): 
            spike_trains = datagenerator.get_spike_trains(data_idx, "train")
            
            # Calculate binning indices for the current simulation interval
            start_bin_idx = sum(datagenerator.len_data[:-1])
            end_bin_idx   = sum(datagenerator.len_data)-1
            num_bins = end_bin_idx - start_bin_idx + 1
            
            interval_start = (params['bin_width'] * start_bin_idx)
            interval_end   = (params['bin_width'] * (end_bin_idx+1))

            # Execute NEURON simulation and extract binned states
            neuronalreservoir.resister_spike_events(spike_trains)
            neuronalreservoir.generate_dynamics(interval_end)
            state_vars = neuronalreservoir.get_binned_states(interval_start, num_bins)
            
            # Store binned states for later optimization (readout training)
            neuronalreservoir.train_state_vars = np.concatenate([neuronalreservoir.train_state_vars, state_vars], axis=0)
            
            # Save raw simulation data to buffer if index matches selected batches
            if save_buffer:
                if (data_idx, "training") in zip(neuronalreservoir.batches_to_save_idx, neuronalreservoir.batches_to_save_mode):
                    neuronalreservoir.save_to_buffer("training", data_idx, spike_trains, datagenerator)
 
            # Accumulate spike timings for analysis
            v_rec_array    = np.array(neuronalreservoir.Vm_at_soma)
            t_rec_array = np.array(neuronalreservoir.t_rec.to_python())
            neuronalreservoir.spike_timings = neuronalreservoir.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)

            nrn.frecord_init()                   
        
        logger.info("--- End Training Data Simulation ---")

        # Train the readout weights based on simulated reservoir states
        neuronalreservoir.optimize(neuronalreservoir.train_state_vars, datagenerator.trainingdata_target)
        neuronalreservoir.overwrite_buffer_after_optimized(datagenerator)
        neuronalreservoir.save_buffer_all()

        logger.info("--- Start Test Data Simulation ---")
        
        # Start test data simulation loop
        for data_idx in tqdm(range(datagenerator.test_dataset_size), desc="Testing Data Simulation", disable=is_multirun): 
            spike_trains = datagenerator.get_spike_trains(data_idx, "test")
            
            start_bin_idx = sum(datagenerator.len_data[:-1])
            end_bin_idx   = sum(datagenerator.len_data)-1
            num_bins = end_bin_idx - start_bin_idx + 1
            
            interval_start = (neuronalreservoir.bin_width * start_bin_idx)
            interval_end   = (neuronalreservoir.bin_width * (end_bin_idx+1))

            # Execute simulation for test data
            neuronalreservoir.resister_spike_events(spike_trains)
            neuronalreservoir.generate_dynamics(interval_end)
            state_vars = neuronalreservoir.get_binned_states(interval_start, num_bins)
            
            neuronalreservoir.test_state_vars  = np.concatenate([neuronalreservoir.test_state_vars, state_vars], axis=0)
            
            # Save test batch to buffer if index matches
            if save_buffer:
                if (data_idx, "test") in zip(neuronalreservoir.batches_to_save_idx, neuronalreservoir.batches_to_save_mode):
                    neuronalreservoir.save_to_buffer("test", data_idx, spike_trains, datagenerator)
                    neuronalreservoir.save_buffer_single(len(neuronalreservoir.data_buffer) - 1)
 
            v_rec_array    = np.array(neuronalreservoir.Vm_at_soma)
            t_rec_array = np.array(neuronalreservoir.t_rec.to_python())
            neuronalreservoir.spike_timings = neuronalreservoir.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)

            nrn.frecord_init()

        logger.info("--- End Test Data Simulation ---")

        logger.info("--- Start Saving Data and Images ---")

        # Evaluate and plot classification results for Training set
        confusion_matrix, confusion_matrix_axis = neuronalreservoir.get_classification_result("training", datagenerator)
        from NeuronalReservoir_classification import plot_confusion_matrix
        plot_confusion_matrix(
            confusion_matrix=confusion_matrix, 
            labels=confusion_matrix_axis, 
            title='Classification Confusion Matrix (Training Data)',
            filename='./figure/confmat_train.png'
        )
        
        # Evaluate and plot classification results for Test set
        confusion_matrix, confusion_matrix_axis = neuronalreservoir.get_classification_result("test", datagenerator)
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
        neuronalreservoir.optimize(state_vars[params['task']['len_transientdata']:params['task']['len_transientdata']+params['task']['len_trainingdata'], :], datagenerator.trainingdata_target)
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
        axes[3].plot(t_output[:len_train], output[len_trans:len_trans+len_train], 'b', label='Train Pred', linewidth=1)
        # Plot test phase predictions
        axes[3].plot(t_output[len_train:], output[len_trans+len_train:], 'r', label='Test Pred', linewidth=1.2)
        
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
        plt.savefig("reservoir_flow_analysis.png")
        plt.show()
        
        # Final metric validation
        mse_test = mean_squared_error(target_all[len_train:len_train+len_test], output[len_trans+len_train:])
        logger.info(f"Test MSE: {mse_test:.8f}")

if __name__ == "__main__":
    main()
