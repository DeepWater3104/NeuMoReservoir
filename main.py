import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the reservoir simulation.
    Hydra handles the configuration loading and output directory management.
    """
    # Convert DictConfig to a standard Python dictionary for compatibility with existing classes
    params = OmegaConf.to_container(cfg, resolve=True)
    
    # Print the resolved configuration to the console for verification
    print("--- Simulation Configuration ---")
    print(OmegaConf.to_yaml(cfg))

    # Initialize the pseudo-random number generator for reproducibility
    # Using the seed defined in the configuration file
    seed = params.get('seed', 1234)
    prng = np.random.default_rng(seed)
    
    # Check the experimental setup
    print(f"Task name: {params['task']['name']}")
    print(f"Random seed: {seed}")
    
    # The output directory is automatically managed by Hydra (check the 'outputs' folder)
    # You can access the current working directory using hydra.utils.get_original_cwd() if needed

    # 1. Compile MOD files (Only if necessary or environment changed)
    from neuron_simulation import run_nrnivmodl
    cell_dir = "./cells/cell1/" # This path should eventually come from cfg
    run_nrnivmodl(cell_dir)
    
    # 2. Load Cell Model
    from neuron import h as nrn
    from neuron_simulation import get_hoc_morph_for_emodel_folder, extract_template_name, check_line_in_file
    
    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(cell_dir)
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())
    
    template_name = extract_template_name(hoc_path.as_posix())
    
    # Instantiate the cell based on its template structure
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(nrn, template_name)(0, cell_dir + "morphology", morph_path.name)
    else:
        cell = getattr(nrn, template_name)(cell_dir + "morphology", morph_path.name)


    # --- 3. DataGenerator Setup ---
    # Import the speaker-specific dataset codes from TI46Subset.py
    from TI46Subset import trainseq_code, testseq_code
    from DataGenerator import TI46word_datagenerator
    
    # Create copies of the original lists to avoid mutating global variables
    train_code = trainseq_code.copy()
    test_code = testseq_code.copy()
    
    # Implement dataset repetition and shuffling logic here
    # These functions were originally in your classification scripts
    from NeuronalReservoir_classification import repeat_dataset_codes, shuffle_dataset_codes_numpy
    
    # Repeat and shuffle training data
    # Accessing 'n_repetition' from the task-specific YAML configuration
    print(f"--- Preparing training data (Repetition: {params['task']['n_repetition']}x) ---")
    train_code = repeat_dataset_codes(train_code, n_times=params['task']['n_repetition'])
    train_code = shuffle_dataset_codes_numpy(train_code, prng)
    
    # Repeat and shuffle test data
    print(f"--- Preparing test data (Repetition: {params['task']['n_repetition']}x) ---")
    test_code = repeat_dataset_codes(test_code, n_times=params['task']['n_repetition'])
    test_code = shuffle_dataset_codes_numpy(test_code, prng)

    # Instantiate the data generator
    # path_to_dataset should eventually be moved to your YAML config
    datagenerator = TI46word_datagenerator(
        params=params['task'],
        prng=prng,
        trainseq_code=train_code,
        testseq_code=test_code,
        path_to_dataset="./dataset/ti46/ti20/"
    )
    
    # Synchronize bin_width from datagenerator to the global params
    params['bin_width'] = datagenerator.bin_width
    print(f"DataGenerator initialized with bin_width: {params['bin_width']} ms")

    ## --- Debugging Data Access ---
    #import os
    #
    ## Check if the directory itself is visible
    #dataset_path = datagenerator.path_to_dataset
    #print(f"Checking directory: {os.path.abspath(dataset_path)}")
    #if os.path.exists(dataset_path):
    #    print(f"Directory exists. Contents (first 5): {os.listdir(dataset_path)[:5]}")
    #else:
    #    print("ERROR: Directory does not exist at the specified path.")
    #
    ## Check the first file the generator will try to access
    ## Assuming the format is something like 'f1_01_1.wav'
    #print(f'debug: {len(train_code)}')
    #sample_code = train_code[0]
    #print(f"First training sample codes: {sample_code}")


    spike_trains = datagenerator.get_spike_trains(data_idx=0, mode='train')
    print(spike_trains)

if __name__ == "__main__":
    main()
