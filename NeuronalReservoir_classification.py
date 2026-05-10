import logging
from NeuronalReservoir import neuronalreservoir
from neuron import h as nrn
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Configure logging
logger = logging.getLogger(__name__)

class neuronalreservoir_classification(neuronalreservoir):
    def __init__(self, cell, prng, params):
        self.cell = cell
        self.prng = prng
        nrn.celsius = 36

        self.save_buffer          = params['task']['save_buffer']
        if self.save_buffer:
            self.batches_to_save_idx   = params['batches_to_save_idx']
            self.batches_to_save_mode  = params['batches_to_save_mode']

        self.exc_num_syn          = params['task']['exc_num_syn']
        self.exc_syn_weight       = params['task']['exc_syn_weight']

        # used to calculate firing rate
        self.Vm_at_soma = nrn.Vector().record(self.cell.soma[0](0.5)._ref_v)
        self.spike_timings = []

        super().__init__(cell, prng, params)

        self.train_state_vars = np.zeros((0, self.num_states))
        self.test_state_vars = np.zeros((0, self.num_states))

        self.create_records_for_buffer()

        self.data_buffer = []

        
        logger.info("Initialized neuronalreservoir_classification.")

    def create_records_for_buffer(self):
        if self.save_buffer:
            self.buffer_variable_list = []
            if self.record_target == 'potential':
                total_length = 0
                cumulative_length_dict = []
                for sec in self.cell.all:
                    cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                    cumulative_length_dict.append(cumulative_length)
                    total_length += sec.L

                # Randomly select recording locations based on physical length
                while len(self.buffer_variable_list) < self.num_states:
                    rec_loc = total_length * self.prng.random()

                    for index, sec in enumerate(self.cell.all):
                        if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                            rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                            if hasattr(sec(rec_prop), '_ref_cai'):
                                v = nrn.Vector().record(sec(rec_prop)._ref_cai)
                                self.buffer_variable_list.append(v)
                            else:
                                break
            elif self.record_target == 'calcium_acum':
                total_length = 0
                cumulative_length_dict = []
                for sec in self.cell.all:
                    cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                    cumulative_length_dict.append(cumulative_length)
                    total_length += sec.L

                # Record membrane potential at random locations
                for rec in range(self.num_states):
                    rec_loc = total_length * self.prng.random()

                    for index, sec in enumerate(self.cell.all):
                        if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                            rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                            v = nrn.Vector().record(sec(rec_prop)._ref_v)
                            self.buffer_variable_list.append(v)
            
    def save_to_buffer(self, mode, data_idx, spike_train, datagenerator):
        logger.debug(f"Saving data to buffer. Mode: {mode}, Index: {data_idx}")
        buffer = {}
        buffer['mode']         = mode
        buffer['data_idx']     = data_idx

        buffer['variables']    = []
        v_rec_np = np.stack([v_rec.to_python() for v_rec in self.v_rec_list], axis=1)
        buffer['variables'].append(v_rec_np)
        v_rec_np = np.stack([v_rec.to_python() for v_rec in self.buffer_variable_list], axis=1)
        buffer['variables'].append(v_rec_np)
        buffer['t_rec']        = np.array(self.t_rec.to_python())
    
        buffer['input'] = {}
        buffer['input']['spike_times']   = spike_train[:, 0]
        buffer['input']['spike_neurons'] = spike_train[:, 1]

        if mode=="training":
            buffer['TrueLabel']     = datagenerator.train_label[buffer['data_idx']]
            # Calculate indices for slicing state variables corresponding to this batch
            start_bin_idx           = sum(datagenerator.len_data[:-1])
            end_bin_idx             = sum(datagenerator.len_data)-1
            buffer['target']        = datagenerator.trainingdata_target[start_bin_idx:end_bin_idx+1, :]
            buffer['output']        = self.readout(self.train_state_vars[start_bin_idx:end_bin_idx+1, :])
            buffer['time_output']   = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width
            buffer['reservoir_state'] = self.train_state_vars[start_bin_idx:end_bin_idx+1, :]
            #buffer['output']         = self.readout(buffer['reservoir_state'])
            #buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "training", datagenerator)

        elif mode=="test":
            buffer['TrueLabel']     = datagenerator.test_label[buffer['data_idx']]
            # Calculate indices for slicing state variables in test mode
            start_bin_idx           = sum(datagenerator.len_data[datagenerator.train_dataset_size:-1])
            end_bin_idx             = sum(datagenerator.len_data[datagenerator.train_dataset_size:])-1
            buffer['target']        = datagenerator.testdata_target[start_bin_idx:end_bin_idx+1, :]
            buffer['output']        = self.readout(self.test_state_vars[start_bin_idx:end_bin_idx+1, :])
            buffer['time_output']   = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width + sum(datagenerator.len_data[:datagenerator.train_dataset_size]) * self.bin_width
            buffer['reservoir_state'] = self.test_state_vars[start_bin_idx:end_bin_idx+1, :]
            buffer['output']         = self.readout(buffer['reservoir_state'])
            buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "test", datagenerator)


        self.data_buffer.append(buffer)

    def overwrite_buffer_after_optimized(self, datagenerator):
        logger.info("Overwriting buffer with optimized readout results.")
        for buffer in self.data_buffer:
            if buffer['mode']=="training":
                start_bin_idx            = sum(datagenerator.len_data[:buffer['data_idx']])
                end_bin_idx              = sum(datagenerator.len_data[:buffer['data_idx']+1])-1
                buffer['output']         = self.readout(buffer['reservoir_state'])
                buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "training", datagenerator)
            elif buffer['mode']=="test":
                start_bin_idx            = sum(datagenerator.len_data[datagenerator.train_dataset_size:datagenerator.train_dataset_size+buffer['data_idx']])
                end_bin_idx              = sum(datagenerator.len_data[datagenerator.train_dataset_size:datagenerator.train_dataset_size+buffer['data_idx']+1])-1
                buffer['output']         = self.readout(buffer['reservoir_state'])
                buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "test", datagenerator)

    def classify(self, data_idx, mode, datagenerator):
        if mode=="training":
            start_bin_idx = sum(datagenerator.len_data[:data_idx])
            end_bin_idx   = sum(datagenerator.len_data[:data_idx+1])-1
            output_within_batch = self.readout(self.train_state_vars[start_bin_idx:end_bin_idx+1, :])
        elif mode=="test":
            start_bin_idx = sum(datagenerator.len_data[datagenerator.train_dataset_size:datagenerator.train_dataset_size+data_idx])
            end_bin_idx   = sum(datagenerator.len_data[datagenerator.train_dataset_size:datagenerator.train_dataset_size+data_idx+1])-1
            output_within_batch = self.readout(self.test_state_vars[start_bin_idx:end_bin_idx+1, :])
        
        # Determine the class by finding the most frequent winner neuron over time steps
        winner_neuron_history = np.zeros(output_within_batch.shape[0])
        for bin_idx in range(output_within_batch.shape[0]):
            winner_neuron = np.argmax(output_within_batch[bin_idx, :])
            winner_neuron_history[bin_idx] = winner_neuron

        unique, freq = np.unique(winner_neuron_history, return_counts=True)
        mode_val = unique[np.argmax(freq)]
        return mode_val

    def get_classification_result(self, mode, datagenerator):
        logger.info(f"Generating classification result for {mode} mode.")
        confusion_matrix_axis = np.unique(datagenerator.test_label, return_counts=False)
        num_unique_prompts = np.size(confusion_matrix_axis)
        confusion_matrix = np.zeros((num_unique_prompts, num_unique_prompts)) # axis 0 is prediction axis 1 is ground truth
        if mode=="training":
            for data_idx in range(datagenerator.train_dataset_size):
                predicted = self.classify(data_idx, "training", datagenerator)
                confusion_matrix[int(predicted), int(datagenerator.train_label[data_idx])] += 1
        elif mode=="test":
            for data_idx in range(datagenerator.test_dataset_size):
                predicted = self.classify(data_idx, "test", datagenerator)
                confusion_matrix[int(predicted), int(datagenerator.test_label[data_idx])] += 1

        return confusion_matrix, confusion_matrix_axis

    def save_buffer_single(self, buffer_idx):
        buffer = self.data_buffer[buffer_idx]
        filename = "./data/buffer" + str(buffer_idx).zfill(2) + ".npz"
        logger.info(f"Saving buffer to {filename}")
        np.savez(filename, **buffer)
        filename = "./figure/buffer" + str(buffer_idx).zfill(2) + ".png"
        plot_timeseries(self.data_buffer[buffer_idx], filename)
        self.data_buffer[buffer_idx] = {}

    def save_buffer_all(self):
        for buffer_idx, buffer in enumerate(self.data_buffer):
            # Visualize all buffered time-series data
            filename = "./figure/buffer" + str(buffer_idx).zfill(2) + ".png"
            plot_timeseries(self.data_buffer[buffer_idx], filename)
            filename = "./data/buffer" + str(buffer_idx).zfill(2) + ".npz"
            logger.info(f"Saving buffer to {filename}")
            np.savez(filename, **buffer)
            self.data_buffer[buffer_idx] = {}


def plot_confusion_matrix(confusion_matrix, labels, title='Confusion Matrix', filename='confmat.png'):
    """
    Plots the confusion matrix as a heatmap.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix (rows: prediction, columns: ground truth).
        labels (np.ndarray or list): Labels for each axis (class names or codes).
        title (str): Title of the plot.
    """
    
    # Normalize confusion matrix (optional): Uncomment to display as proportions
    # cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ...)

    plt.figure(figsize=(10, 8)) # Set figure size
    
    # Plot heatmap
    # annot=True: Display values in cells
    # fmt='d': Integer format
    # cmap='Blues': Set colormap
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        cmap='Blues',
        cbar=True, # Display color bar
        xticklabels=labels, # x-axis (Ground Truth)
        yticklabels=labels  # y-axis (Predicted)
    )

    # Axis labels and title
    plt.title(title, fontsize=16)
    plt.ylabel('Predicted Label', fontsize=14) # Rows represent prediction
    plt.xlabel('True Label', fontsize=14) # Columns represent ground truth

    # Adjust ticks to prevent overlap
    plt.tick_params(axis='both', which='major', labelsize=10, rotation=45) 
    
    # Adjust layout
    plt.tight_layout() 
    
    # Save graph
    plt.savefig(filename)


def plot_timeseries(buffer, filename, detailed_reservoir_layer_plot=True):
    # --- 1. Configuration parameters ---
    WIDTH_RATIOS = [1, 1, 1]
    NUM_OUTPUT_NEURONS = buffer['output'].shape[1]
    
    # --- 2. Create Figure and Axes ---
    fig = plt.figure(figsize=(15, 8)) # Slightly larger figure for visibility
    
    gs = gridspec.GridSpec(1, 3, width_ratios=WIDTH_RATIOS)
    
    # --- 3. Define Axes for each layer ---
    
    # 1. Input Layer (Left)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_title('Input Layer Raster Plot', fontsize=14)
    ax_input.set_xlabel('Time (ms)', fontsize=12)
    ax_input.set_ylabel('Neuron Index', fontsize=12)
    
    ax_input.scatter(buffer['input']['spike_times'], buffer['input']['spike_neurons'], s=10, alpha=0.7)

    # 2. Reservoir Layer (Center)
    if detailed_reservoir_layer_plot:
        # Create vertically split subplots for reservoir states
        gs_reservoir = gridspec.GridSpecFromSubplotSpec(
            3, 1, 
            subplot_spec=gs[0, 1], 
            hspace=0.1 # Increase vertical spacing slightly
        )
        
        ax_reservoir_list = []
        ax_reservoir = fig.add_subplot(gs_reservoir[0, 0])
        for i in range(buffer['variables'][0].shape[1]):
            ax_reservoir.plot(buffer['t_rec'], buffer['variables'][0][:, i], linewidth=1.2)
        
        ax_reservoir_list.append(ax_reservoir)

        ax_reservoir = fig.add_subplot(gs_reservoir[1, 0])
        for i in range(buffer['reservoir_state'].shape[1]):
            ax_reservoir.plot(buffer['time_output'], buffer['reservoir_state'][:, i], linewidth=1.2)

        ax_reservoir_list.append(ax_reservoir)

        ax_reservoir = fig.add_subplot(gs_reservoir[2, 0])
        for i in range(buffer['variables'][1].shape[1]):
            ax_reservoir.plot(buffer['t_rec'], buffer['variables'][1][:, i], linewidth=1.2)
            ax_reservoir.set_xlabel('Time [ms]')

        ax_reservoir_list.append(ax_reservoir)
    else:
        ax_reservoir = fig.add_subplot(gs[0, 1])
        ax_reservoir.set_title('Reservoir Neuron States', fontsize=14)
        ax_reservoir.set_xlabel('Time (ms)', fontsize=12)
        ax_reservoir.set_ylabel('Membrane Potential (mV)', fontsize=12)

        # Ensure buffer['variables'][0] exists and has appropriate shape
        if buffer['variables'][0].shape[0] > 0:
            for state_idx in range(buffer['variables'][0].shape[1]):
                ax_reservoir.plot(buffer['t_rec'], buffer['variables'][0][:, state_idx], linewidth=0.8)
        else:
            ax_reservoir.text(0.5, 0.5, 'No Reservoir Data', transform=ax_reservoir.transAxes, 
                              ha='center', va='center', fontsize=12, color='gray')

 
    # 3. Output Layer (Right) - Split vertically
    gs_output = gridspec.GridSpecFromSubplotSpec(
        NUM_OUTPUT_NEURONS, 1, 
        subplot_spec=gs[0, 2], 
        hspace=0.1
    )
    
    ax_output_list = []
    for i in range(NUM_OUTPUT_NEURONS):
        ax_output = fig.add_subplot(gs_output[i, 0])
        ax_output.plot(buffer['time_output'], buffer['output'][:, i], label="Output", linewidth=1.2)
        ax_output.plot(buffer['time_output'], buffer['target'][:, i], label="Ground Truth", linestyle='--', linewidth=1.2)

        # Adjust Y-axis labels and ticks
        if i == NUM_OUTPUT_NEURONS - 1: # X-axis label only on the bottom subplot
            ax_output.set_xlabel('Time Steps', fontsize=12)
        else: # Hide X-axis tick labels for others
            ax_output.tick_params(labelbottom=False)

        if i == 0: # Title and legend for the first subplot
            ax_output.set_title('Output Neuron Activity', fontsize=14)
            ax_output.legend(loc='upper right', fontsize=10)
        
        # Y-axis label for each output neuron
        ax_output.set_ylabel(f'Neuron {i+1}', fontsize=10, rotation=0, ha='right')
        ax_output.tick_params(axis='y', labelsize=10)
        ax_output.grid(True, linestyle=':', alpha=0.6)

        ax_output_list.append(ax_output)
        
    # --- 4. Final adjustments and display ---
    title = f"Time Series: True {buffer['TrueLabel']} Predicted {buffer['PredictedLabel']}"
    fig.suptitle(title, fontsize=16, y=1.005)

    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def repeat_dataset_codes(seq_code, n_times=5):
    """
    Function to duplicate dataset index codes (list) n_times while maintaining consistency.
    """
    keys = list(seq_code.keys())
    
    # Get list length
    original_length = len(seq_code[keys[0]])
    if original_length == 0:
        logger.warning("Dataset is empty. Skipping duplication.")
        return seq_code

    # Concatenate each list n_times
    for key in keys:
        original_list = seq_code[key]
        # Link [a, b, c] to [a, b, c, a, b, c, ...]
        seq_code[key] = original_list * n_times 
        
    new_length = len(seq_code[keys[0]])
    logger.info(f"Dataset duplication complete. Original size: {original_length}, New size: {new_length} ({n_times}x)")
    return seq_code


def shuffle_dataset_codes_numpy(seq_code, prng: np.random.Generator):
    """
    Function to shuffle dataset code lists in a random yet reproducible way 
    using NumPy's Generator, maintaining key correspondence.
    """
    keys = list(seq_code.keys())
    
    # Get list length (all should be the same)
    data_length = len(seq_code[keys[0]])
    if data_length == 0:
        logger.warning("Dataset is empty. Skipping shuffle.")
        return seq_code

    # 1. Generate index array for shuffling
    # prng.permutation() returns a random permutation of indices from 0 to N-1
    shuffled_indices = prng.permutation(data_length)

    # 2. Shuffle each list using the generated indices
    for key in keys:
        # Convert to NumPy array and reorder based on shuffled indices
        original_array = np.array(seq_code[key])
        shuffled_array = original_array[shuffled_indices]
        
        # Store result back in dictionary as list format
        seq_code[key] = shuffled_array.tolist()
    
    logger.info(f"Dataset shuffle complete. Keys: {keys}, Size: {data_length}")
    return seq_code
