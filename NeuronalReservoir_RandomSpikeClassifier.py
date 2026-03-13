from NeuronalReservoir import neuronalreservoir
from neuron import h as nrn
from neuron.units import ms, mV

class neuronalreservoir_classification(neuronalreservoir):
    def __init__(self, cell, datagenerator_classification, prng, params, calcium=None, ip3=None, cyt=None):
        self.cell = cell
        self.calcium=calcium
        self.ip3=ip3
        self.cyt=cyt

        self.prng = prng
        nrn.celsius = 36
        self.datagenerator = datagenerator_classification

        # store parameters as members
        self.bin_width   = params['bin_width']
        self.num_states = params['num_states']
        self.num_outputs = params['num_outputs']
        self.exc_num_syns         = params['exc_num_syns']
        self.exc_syn_weight       = params['exc_syn_weight']
        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.syn_mechanisms         = params['syn_mechanisms']
        self.record_target = params['record_target']
        self.reg = params['reg']
        self.plot_all              = params['plot_all']

        self.W = self.prng.random((self.num_outputs, self.num_states+1)) # readout weight
        self.training_state_vars = np.zeros((self.num_states, 0))
        self.test_state_vars = np.zeros((self.num_states, 0))
        self.create_synapses()
        self.connect_synapses()
        self.v_rec_list = []
        self.excsyncurrent_rec_list = []
        self.inhsyncurrent_rec_list = []
        self.create_records(calcium, cyt)

        # used to calculate firing rate
        self.Vm_at_soma = nrn.Vector().record(self.cell.soma[0](0.5)._ref_v)
        self.spike_timings = []

        self.data_buffer = []
        self.batches_to_save_idx   = params['batches_to_save_idx']
        self.batches_to_save_mode  = params['batches_to_save_mode']

    def create_records(self, calcium=None, cyt=None):
        super().create_records(calcium, cyt)
        if self.plot_all:
            self.buffer_variable_list = []
            if self.record_target == 'potential':
                if calcium == None: # if RxD calcium variable is not given, record calcium accumulation
                    total_length = 0
                    cumulative_length_dict = []
                    for sec in self.cell.all:
                        cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                        cumulative_length_dict.append(cumulative_length)
                        total_length += sec.L

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
                else: # record RxD  calcium variable
                    node_list = calcium[cyt].nodes
                    selected_nodes_array = self.prng.choice(node_list, size=self.num_states, replace=False)
                    selected_nodes_list = selected_nodes_array.tolist()
                    for node in selected_nodes_list:
                         v = nrn.Vector().record(node._ref_concentration)
                         self.buffer_variable_list.append(v)
            elif self.record_target == 'calcium_rxd' or self.record_target == 'calcium_acum':
                total_length = 0
                cumulative_length_dict = []
                for sec in self.cell.all:
                    cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                    cumulative_length_dict.append(cumulative_length)
                    total_length += sec.L

                for rec in range(self.num_states):
                    rec_loc = total_length * self.prng.random()

                    for index, sec in enumerate(self.cell.all):
                        if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                            rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                            v = nrn.Vector().record(sec(rec_prop)._ref_v)
                            self.buffer_variable_list.append(v)

    def sampling(self, start_bin_idx, end_bin_idx):
        state_vars_within_batch = np.zeros((self.num_states, end_bin_idx - start_bin_idx + 1))
        v_rec_np = np.array([np.array(v_rec.to_python()) for v_rec in self.v_rec_list])
        t_rec_array = np.array(self.t_rec.to_python())
        for bin_idx in range(start_bin_idx, end_bin_idx+1):
            bin_start_time = self.bin_width * bin_idx 
            bin_end_time   = self.bin_width * (bin_idx+1)
            time_idx_within_bin = np.where((bin_start_time < t_rec_array) & (t_rec_array < bin_end_time))[0]
            if time_idx_within_bin[-1]+1 < len(t_rec_array): # not the last bin in this batch
                for time_idx in time_idx_within_bin:
                    state_vars_within_batch[:, bin_idx - start_bin_idx] += (v_rec_np[:, time_idx] * (t_rec_array[time_idx+1] - t_rec_array[time_idx]))
                state_vars_within_batch[:, bin_idx - start_bin_idx] = state_vars_within_batch[:, bin_idx - start_bin_idx] / (t_rec_array[time_idx_within_bin[-1]+1] - t_rec_array[time_idx_within_bin[0]])
                    
            elif time_idx_within_bin[-1]+1 == len(t_rec_array): # the last bin in this batch
                for time_idx in time_idx_within_bin[:-1]:
                    state_vars_within_batch[:, bin_idx - start_bin_idx] += (v_rec_np[:, time_idx] * (t_rec_array[time_idx+1] - t_rec_array[time_idx]))
                state_vars_within_batch[:, bin_idx - start_bin_idx] += (v_rec_np[:, time_idx_within_bin[-1]] * (t_rec_array[time_idx_within_bin[-1]] - t_rec_array[time_idx_within_bin[-1]-1]))
                state_vars_within_batch[:, bin_idx - start_bin_idx] = state_vars_within_batch[:, bin_idx - start_bin_idx] / (t_rec_array[time_idx_within_bin[-1]] - t_rec_array[time_idx_within_bin[-2]] + (t_rec_array[time_idx_within_bin[-1]] - t_rec_array[time_idx_within_bin[0]]))

        return state_vars_within_batch

    def save_to_buffer(self, mode, data_idx, spike_train):
        buffer = {}
        buffer['mode']         = mode
        buffer['data_idx']     = data_idx

        v_rec_np = np.array([np.array(v_rec.to_python()) for v_rec in self.v_rec_list])
        buffer['variables']    = []
        buffer['variables'].append(v_rec_np)
        v_rec_np = np.array([np.array(v_rec.to_python()) for v_rec in self.buffer_variable_list])
        buffer['variables'].append(v_rec_np)
        buffer['t_rec']        = np.array(self.t_rec.to_python())

        buffer['input'] = {}
        buffer['input']['spike_times']   = spike_train[:, 0]
        buffer['input']['spike_neurons'] = spike_train[:, 1]

        if mode=="training":
            buffer['TrueLabel']= self.datagenerator.train_labels[buffer['data_idx']]
            start_bin_idx      = sum(self.datagenerator.len_data[:-1])
            end_bin_idx        = sum(self.datagenerator.len_data)-1
            buffer['target']   = self.datagenerator.trainingdata_target[:, start_bin_idx:end_bin_idx+1]
            buffer['output']   = self.readout_training()[:, start_bin_idx:end_bin_idx+1]
            buffer['time_output'] = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width
            buffer['reservoir_state'] = self.training_state_vars[:, start_bin_idx:end_bin_idx+1]
        elif mode=="test":
            buffer['TrueLabel']= self.datagenerator.test_labels[buffer['data_idx']]
            start_bin_idx      = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:-1])
            end_bin_idx        = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:])-1
            buffer['target']   = self.datagenerator.testdata_target[:, start_bin_idx:end_bin_idx+1]
            buffer['output']   = self.readout_test()[:, start_bin_idx:end_bin_idx+1]
            buffer['time_output'] = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width + sum(self.datagenerator.len_data[:self.datagenerator.train_dataset_size]) * self.bin_width
            buffer['reservoir_state'] = self.test_state_vars[:, start_bin_idx:end_bin_idx+1]

        self.data_buffer.append(buffer)

    def overwrite_buffer_after_optimized(self):
        for buffer in self.data_buffer:
            if buffer['mode']=="training":
                start_bin_idx      = sum(self.datagenerator.len_data[:buffer['data_idx']])
                end_bin_idx        = sum(self.datagenerator.len_data[:buffer['data_idx']+1])-1
                buffer['output'] = self.readout_training()[:, start_bin_idx:end_bin_idx+1]
                buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "training")
            elif buffer['mode']=="test":
                start_bin_idx      = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:self.datagenerator.train_dataset_size+buffer['data_idx']])
                end_bin_idx        = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:self.datagenerator.train_dataset_size+buffer['data_idx']+1])-1
                buffer['output'] = self.readout_test()[:, start_bin_idx:end_bin_idx+1]
                buffer['PredictedLabel'] = self.classify(buffer['data_idx'], "test")

    def generate_response(self, show_progress=True):
        from Analysis import get_spike_timings

        nrn.finitialize(-65 * mV)

        if show_progress == False:
            for data_idx in range(self.datagenerator.train_dataset_size): # requirement for data_idx is to specify single data within training or test set
                spike_trains = self.datagenerator.resister_inputevent_toNetCon(data_idx, "training")
                start_bin_idx = sum(self.datagenerator.len_data[:-1])
                end_bin_idx   = sum(self.datagenerator.len_data)-1
                # bin_width in neuronalreservoir does not neccesarrily correspond to that in datagenerator 
                interval_start = (self.bin_width * start_bin_idx)
                interval_end   = (self.bin_width * (end_bin_idx+1))
                nrn.continuerun( interval_end * ms )
                shape_before_concatenate = self.training_state_vars.shape
                self.training_state_vars = np.concatenate([self.training_state_vars, self.sampling(start_bin_idx, end_bin_idx)], axis=1)
                if (data_idx, "training") in zip(self.batches_to_save_idx, self.batches_to_save_mode):
                    self.save_to_buffer("training", data_idx, spike_trains)

                v_rec_array    = np.array(self.Vm_at_soma)
                t_rec_array = np.array(self.t_rec.to_python())
                self.spike_timings = self.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)

                nrn.frecord_init()              

            for data_idx in range(self.datagenerator.test_dataset_size): # requirement for data_idx is to specify single data within training or test set
                spike_trains = self.datagenerator.resister_inputevent_toNetCon(data_idx, "test")
                start_bin_idx = sum(self.datagenerator.len_data[:-1])
                end_bin_idx   = sum(self.datagenerator.len_data)-1
                # bin_width in neuronalreservoir does not neccesarrily correspond to that in datagenerator 
                interval_start = (self.bin_width * start_bin_idx)
                interval_end   = (self.bin_width * (end_bin_idx+1))
                nrn.continuerun( interval_end * ms )
                self.test_state_vars = np.concatenate([self.test_state_vars, self.sampling(start_bin_idx, end_bin_idx)], axis=1)
                if (data_idx, "test") in zip(self.batches_to_save_idx, self.batches_to_save_mode):
                    self.save_to_buffer("test", data_idx, spike_trains)

                v_rec_array    = np.array(self.Vm_at_soma)
                t_rec_array = np.array(self.t_rec.to_python())
                self.spike_timings = self.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)

                nrn.frecord_init()    

        elif show_progress == True:
            from tqdm import tqdm
            for data_idx in tqdm(range(self.datagenerator.train_dataset_size), desc="Training Data Simulation"): 
                # requirement for data_idx is to specify single data within training or test set
                spike_trains = self.datagenerator.resister_inputevent_toNetCon(data_idx, "training")
                
                # datagenerator.len_data はリストや配列であると仮定
                start_bin_idx = sum(self.datagenerator.len_data[:-1])
                end_bin_idx   = sum(self.datagenerator.len_data)-1
                
                # bin_width in neuronalreservoir does not neccesarrily correspond to that in datagenerator 
                interval_start = (self.bin_width * start_bin_idx)
                interval_end   = (self.bin_width * (end_bin_idx+1))
                
                # nrn (NEURON) の実行
                nrn.continuerun( interval_end * ms )
                
                shape_before_concatenate = self.training_state_vars.shape
                self.training_state_vars = np.concatenate([self.training_state_vars, self.sampling(start_bin_idx, end_bin_idx)], axis=1)
                
                # zip の引数の順序を調整
                if (data_idx, "training") in zip(self.batches_to_save_idx, self.batches_to_save_mode):
                    self.save_to_buffer("training", data_idx, spike_trains)
 
                v_rec_array    = np.array(self.Vm_at_soma)
                t_rec_array = np.array(self.t_rec.to_python())
                self.spike_timings = self.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)
           
                nrn.frecord_init()                   
            
            # ---
            
            # test data loop
            # tqdmで range() をラップし、descで進捗バーの説明を設定
            for data_idx in tqdm(range(self.datagenerator.test_dataset_size), desc="Testing Data Simulation"): 
                # requirement for data_idx is to specify single data within training or test set
                spike_trains = self.datagenerator.resister_inputevent_toNetCon(data_idx, "test")
                
                # datagenerator.len_data はリストや配列であると仮定
                start_bin_idx = sum(self.datagenerator.len_data[:-1])
                end_bin_idx   = sum(self.datagenerator.len_data)-1
                
                # bin_width in neuronalreservoir does not neccesarrily correspond to that in datagenerator 
                interval_start = (self.bin_width * start_bin_idx)
                interval_end   = (self.bin_width * (end_bin_idx+1))
                
                # nrn (NEURON) の実行
                nrn.continuerun( interval_end * ms )
                
                self.test_state_vars = np.concatenate([self.test_state_vars, self.sampling(start_bin_idx, end_bin_idx)], axis=1)
                
                # zip の引数の順序を調整
                if (data_idx, "test") in zip(self.batches_to_save_idx, self.batches_to_save_mode):
                    self.save_to_buffer("test", data_idx, spike_trains)
 
                v_rec_array    = np.array(self.Vm_at_soma)
                t_rec_array = np.array(self.t_rec.to_python())
                self.spike_timings = self.spike_timings + get_spike_timings(t_rec_array, v_rec_array, threshold=-20)

                nrn.frecord_init()

    def optimize(self):
        state_vars = np.concatenate( (self.training_state_vars, np.ones((1, self.training_state_vars.shape[1]))), axis=0 )
        self.W = self.datagenerator.trainingdata_target @ np.transpose(state_vars) @ np.linalg.inv(state_vars @ state_vars.transpose() + self.reg*np.eye(self.num_states+1))

    def readout_training(self):
        state_vars = np.concatenate( (self.training_state_vars, np.ones((1, self.training_state_vars.shape[1]))), axis=0 )
        return  self.W @ state_vars

    def readout_test(self):
        state_vars = np.concatenate( (self.test_state_vars, np.ones((1, self.test_state_vars.shape[1]))), axis=0 )
        return self.W @ state_vars

    def classify(self, data_idx, mode):
        if mode=="training":
            start_bin_idx = sum(self.datagenerator.len_data[:data_idx])
            end_bin_idx   = sum(self.datagenerator.len_data[:data_idx+1])-1
            output_within_batch = self.readout_training()[:, start_bin_idx:end_bin_idx+1]
        elif mode=="test":
            start_bin_idx = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:self.datagenerator.train_dataset_size+data_idx])
            end_bin_idx   = sum(self.datagenerator.len_data[self.datagenerator.train_dataset_size:self.datagenerator.train_dataset_size+data_idx+1])-1
            output_within_batch = self.readout_test()[:, start_bin_idx:end_bin_idx+1]
        
        winner_neuron_history = np.zeros(output_within_batch.shape[1])
        for bin_idx in range(output_within_batch.shape[1]):
            winner_neuron = np.argmax(output_within_batch[:, bin_idx])
            winner_neuron_history[bin_idx] = winner_neuron

        unique, freq = np.unique(winner_neuron_history, return_counts=True)
        mode = unique[np.argmax(freq)]
        return mode

    def get_classification_result(self, mode):
        confusion_matrix_axis = np.arange(self.datagenerator.num_classes)
        confusion_matrix = np.zeros((self.datagenerator.num_classes, self.datagenerator.num_classes)) # axis 0 is prediction axis 1 is ground truth
        if mode=="training":
            for data_idx in range(self.datagenerator.train_dataset_size):
                predicted = self.classify(data_idx, "training")
                confusion_matrix[int(predicted), self.datagenerator.train_labels[data_idx]] += 1
        elif mode=="test":
            for data_idx in range(self.datagenerator.test_dataset_size):
                predicted = self.classify(data_idx, "test")
                confusion_matrix[int(predicted), self.datagenerator.test_labels[data_idx]] += 1

        return confusion_matrix, confusion_matrix_axis

    def save_buffer_all(self):
        for buffer_idx, buffer in enumerate(self.data_buffer):
            filename = "../data/buffer" + str(buffer_idx).zfill(2) + ".npz"
            np.savez(filename, **buffer)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, labels, title='Confusion Matrix', filename='confmat.png'):
    """
    混同行列をヒートマップとして描画します。

    Args:
        confusion_matrix (np.ndarray): 混同行列（行: 予測、列: 真値）。
        labels (np.ndarray or list): 各軸のラベル（クラス名、またはコード）。
        title (str): グラフのタイトル。
    """
    
    # 混同行列を標準化（オプション）：全体に対する割合で表示したい場合にコメントを外す
    # cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ...)

    plt.figure(figsize=(10, 8)) # グラフのサイズを設定
    
    # ヒートマップの描画
    # annot=True: セルに値を表示
    # fmt='d': 値のフォーマットを整数に設定
    # cmap='Blues': カラーマップを設定（様々な色があります 'YlGnBu', 'Reds', 'viridis'など）
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        cmap='Blues',
        cbar=True, # カラーバーを表示
        xticklabels=labels, # x軸（真値）のラベル
        yticklabels=labels  # y軸（予測）のラベル
    )

    # 軸ラベルとタイトル
    plt.title(title, fontsize=16)
    plt.ylabel('Predicted Label', fontsize=14) # 行は予測
    plt.xlabel('True Label', fontsize=14) # 列は真値

    # ラベルが重ならないように調整
    plt.tick_params(axis='both', which='major', labelsize=10, rotation=45) 
    
    # グラフのレイアウトを調整
    plt.tight_layout() 
    
    # グラフの表示
    plt.savefig(filename)


import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np # npが定義されていない場合のために追加

def plot_timeseries(buffer, filename, detailed_reservoir_layer_plot=True):
    # --- 1. 図の構成パラメータ ---
    WIDTH_RATIOS = [1, 1, 1]
    NUM_OUTPUT_NEURONS = buffer['output'].shape[0]
    
    # --- 2. FigureとAxesの作成 ---
    fig = plt.figure(figsize=(15, 8)) # Figureサイズを少し大きくして見やすくする
    
    gs = gridspec.GridSpec(1, 3, width_ratios=WIDTH_RATIOS)
    
    # --- 3. 各層のAxesの定義 ---
    
    # 1. Input Layer (左端)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_title('Input Layer Raster Plot', fontsize=14) # 英語タイトル
    ax_input.set_xlabel('Time (ms)', fontsize=12) # X軸ラベル
    ax_input.set_ylabel('Neuron Index', fontsize=12) # Y軸ラベル
    
    ax_input.scatter(buffer['input']['spike_times'], buffer['input']['spike_neurons'], s=10, alpha=0.7) # マーカーサイズと透明度を調整

    # 2. Reservoir Layer (中央)
    if detailed_reservoir_layer_plot:
        gs_reservoir = gridspec.GridSpecFromSubplotSpec(
            3, 1, 
            subplot_spec=gs[0, 1], 
            hspace=0.1 # 縦方向のスペースを少し広げる
        )
        
        ax_reservoir_list = []
        ax_reservoir = fig.add_subplot(gs_reservoir[0, 0])
        for i in range(buffer['variables'][0].shape[0]):
            ax_reservoir.plot(buffer['t_rec'], buffer['variables'][0][i, :], linewidth=1.2) # 線幅を調整
        
        ax_reservoir_list.append(ax_reservoir)

        ax_reservoir = fig.add_subplot(gs_reservoir[1, 0])
        for i in range(buffer['reservoir_state'].shape[0]):
            ax_reservoir.plot(buffer['time_output'], buffer['reservoir_state'][i, :], linewidth=1.2) # 線幅を調整

        ax_reservoir_list.append(ax_reservoir)

        ax_reservoir = fig.add_subplot(gs_reservoir[2, 0])
        for i in range(buffer['variables'][1].shape[0]):
            ax_reservoir.plot(buffer['t_rec'], buffer['variables'][1][i, :], linewidth=1.2) # 線幅を調整
            ax_reservoir.set_xlabel('Time [ms]')

        ax_reservoir_list.append(ax_reservoir)
    else:
        ax_reservoir = fig.add_subplot(gs[0, 1])
        ax_reservoir.set_title('Reservoir Neuron States', fontsize=14) # 英語タイトル
        ax_reservoir.set_xlabel('Time (ms)', fontsize=12) # X軸ラベル
        ax_reservoir.set_ylabel('Membrane Potential (mV)', fontsize=12) # Y軸ラベル

        # buffer['v_rec']が存在し、形状が適切であることを確認
        if buffer['variables'][0].shape[0] > 0:
            for state_idx in range(buffer['variables'][0].shape[0]):
                ax_reservoir.plot(buffer['t_rec'], buffer['variables'][0][state_idx, :], linewidth=0.8) # 線幅を調整
        else:
            ax_reservoir.text(0.5, 0.5, 'No Reservoir Data', transform=ax_reservoir.transAxes, 
                              ha='center', va='center', fontsize=12, color='gray')

 
    # 3. Output Layer (右端) - 縦に分割
    gs_output = gridspec.GridSpecFromSubplotSpec(
        NUM_OUTPUT_NEURONS, 1, 
        subplot_spec=gs[0, 2], 
        hspace=0.1 # 縦方向のスペースを少し広げる
    )
    
    ax_output_list = []
    for i in range(NUM_OUTPUT_NEURONS):
        ax_output = fig.add_subplot(gs_output[i, 0])
        ax_output.plot(buffer['time_output'], buffer['output'][i, :], label="Output", linewidth=1.2) # 線幅を調整
        ax_output.plot(buffer['time_output'], buffer['target'][i, :], label="Ground Truth", linestyle='--', linewidth=1.2) # ターゲットを点線に

        # Y軸のラベルと目盛りを調整
        if i == NUM_OUTPUT_NEURONS - 1: # 一番下のサブプロットにX軸ラベル
            ax_output.set_xlabel('Time Steps', fontsize=12)
        else: # それ以外のサブプロットはX軸の目盛りを非表示
            ax_output.tick_params(labelbottom=False)

        if i == 0: # 最初のサブプロットにタイトル
            ax_output.set_title('Output Neuron Activity', fontsize=14)
            ax_output.legend(loc='upper right', fontsize=10) # 凡例を追加
        
        # 各出力ニューロンのY軸ラベル
        ax_output.set_ylabel(f'Neuron {i+1}', fontsize=10, rotation=0, ha='right') # 回転させて右揃え
        ax_output.tick_params(axis='y', labelsize=10) # Y軸の目盛りラベルサイズ調整
        ax_output.grid(True, linestyle=':', alpha=0.6) # グリッドを追加

        #ax_output.set_ylim(-0.5, 2.0)
        ax_output_list.append(ax_output)
        
    # --- 4. 図の調整と表示 ---
    title = f"Time Series: True {buffer['TrueLabel']} Predicted {buffer['PredictedLabel']}"
    fig.suptitle(title, fontsize=16, y=1.005) # 全体タイトルを調整

    plt.tight_layout() # 全体タイトルを考慮して調整
    
    plt.savefig(filename, dpi=300) # 高解像度で保存
    plt.close(fig)




def repeat_dataset_codes(seq_code, n_times=5):
    """
    データセットのインデックスコード（リスト）を、対応を保ったままn_times回複製する関数。
    """
    keys = list(seq_code.keys())
    
    # リストの長さを取得
    original_length = len(seq_code[keys[0]])
    if original_length == 0:
        print("データが空です。複製をスキップします。")
        return seq_code

    # 各キーのリストをn_times回複製（連結）する
    for key in keys:
        # リストを複製
        original_list = seq_code[key]
        # [a, b, c] が [a, b, c, a, b, c, ...] となるように連結
        seq_code[key] = original_list * n_times 
        
    new_length = len(seq_code[keys[0]])
    print(f"データセットの複製が完了しました。元のサイズ: {original_length}, 新しいサイズ: {new_length} ({n_times}倍)")
    return seq_code


def shuffle_dataset_codes_numpy(seq_code, prng: np.random.Generator):
    """
    NumPyのGeneratorを使用して、データセットのコードリストを
    対応を保ったままランダムかつ再現性のある方法でシャッフルする関数。
    """
    keys = list(seq_code.keys())
    
    # リストの長さを取得（すべて同じ長さのはず）
    data_length = len(seq_code[keys[0]])
    if data_length == 0:
        print("データが空です。シャッフルをスキップします。")
        return seq_code

    # 1. シャッフル用のインデックス配列を生成
    # prng.permutation() は、0から N-1 までのランダムな順列（インデックス）を返します
    shuffled_indices = prng.permutation(data_length)

    # 2. 生成されたインデックスを使って各リストをシャッフル
    for key in keys:
        # NumPy配列に変換し、シャッフルされたインデックスで要素を並び替える
        original_array = np.array(seq_code[key])
        shuffled_array = original_array[shuffled_indices]
        
        # 結果を元のリスト形式に戻して辞書に格納
        seq_code[key] = shuffled_array.tolist()
    
    print(f"データセットのシャッフルが完了しました。Keys: {keys}, サイズ: {data_length}")
    return seq_code


if __name__ == '__main__':
    import numpy as np
    import NeuronalReservoir_params
    from DataGenerator import sin_datagenerator, MackeyGlass_datagenerator, TI46word_datagenerator, RandomPattern_datagenerator
    from sklearn.metrics import mean_squared_error
    params = NeuronalReservoir_params.RandomSpike_params
    prng = np.random.default_rng(1234)

    # CA3 pyramidal
    #from CA3pyramidal import CA3Pyramidal
    #cell = CA3Pyramidal(STDP=False)

    #from TI46Subset import trainseq_code, testseq_code

    ## 訓練データ
    #print(f"--- 訓練データを5倍に複製 ---")
    #trainseq_code = repeat_dataset_codes(trainseq_code, n_times=params['n_repetition'])
    #print(f"--- 訓練データをシャッフル ---")
    #trainseq_code = shuffle_dataset_codes_numpy(trainseq_code, prng)
    #
    ## テストデータ
    #print(f"\n--- テストデータを5倍に複製 ---")
    #testseq_code = repeat_dataset_codes(testseq_code, n_times=params['n_repetition'])
    #print(f"--- テストデータをシャッフル ---")
    #testseq_code = shuffle_dataset_codes_numpy(testseq_code, prng)
    datagenerator = RandomPattern_datagenerator(params, prng)

    params['batches_to_save_idx']  = []
    params['batches_to_save_mode'] = []

    for data_idx in range(datagenerator.train_dataset_size):
        params['batches_to_save_idx'].append(data_idx)
        params['batches_to_save_mode'].append("training")
    # ---
    
    # testseq_codeの処理
    test_indices = range(datagenerator.test_dataset_size)
    num_test_samples = min(60, len(test_indices)) # 最大60個、データ数を超えないように
    #num_test_samples = len(test_indices)
    # ランダムにインデックスを30個（またはそれ以下）選択
    selected_test_indices = prng.choice(test_indices, size=num_test_samples, replace=False)
    
    for data_idx in selected_test_indices:
        params['batches_to_save_idx'].append(data_idx)
        params['batches_to_save_mode'].append("test")

    #datagenerator = TI46word_datagenerator(params, prng, trainseq_code, testseq_code, "../../../NEURON_implementation/dataset/ti46/ti20/")
    #datagenerator = TI46word_datagenerator(params, prng, trainseq_code, testseq_code, "../dataset/ti46/ti20/")
    params['bin_width'] = datagenerator.bin_width

    import sys
    #sys.path.append("../cells/cell1/")
    sys.path.append("../cells/1672/")
    import os
    from neuron_simulation import run_nrnivmodl, extract_template_name, get_hoc_morph_for_emodel_folder, check_line_in_file
    import logging
    
    #relativepath_cell1 = "../cells/cell1/"
    relativepath_cell1 = "../cells/1672/"
    
    # Check if the mecahnisms folder exsists
    if not(os.path.exists(relativepath_cell1 + "mechanisms")):
        logging.error("No mechanisms directory found.")
    # Run the command
    run_nrnivmodl(relativepath_cell1)
    #from neuron import rxd
    
    # Get the hoc file
    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(relativepath_cell1)
    #Load the standard hoc file and the custom hoc file for the model
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())
    #Extract the template name from the hoc file, and create a cell instance
    method_name = extract_template_name(hoc_path.as_posix())
    #Based on the number of arguments in the template, initialize the cell.
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(nrn, method_name)(0, relativepath_cell1 + "morphology",morph_path.name )
    else:
        cell = getattr(nrn, method_name)( relativepath_cell1 + "morphology",morph_path.name )


    #Setup the parameters for the cell
    nrn.celsius = 34.0
    nrn.v_init = -80.

    reservoir = neuronalreservoir_classification(cell, datagenerator, prng, params)
    reservoir.generate_response()
    reservoir.optimize()

    import sys
    reservoir.overwrite_buffer_after_optimized()

    confusion_matrix, confusion_matrix_axis = reservoir.get_classification_result("training")
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix, 
        labels=confusion_matrix_axis, 
        title='Classification Confusion Matrix (Training Data)',
        filename='../figure/confmat_train.png'
    )
    confusion_matrix, confusion_matrix_axis = reservoir.get_classification_result("test")
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix, 
        labels=confusion_matrix_axis, 
        title='Classification Confusion Matrix (Test Data)',
        filename='../figure/confmat_test.png'
    )
    np.savez("../data/classification_results.npz",
             confusion_matrix=confusion_matrix,
             axis_labels=confusion_matrix_axis)

    for buffer_idx in range(len(reservoir.batches_to_save_idx)):
        filename = "../figure/buffer" + str(buffer_idx).zfill(2) + ".png"
        plot_timeseries(reservoir.data_buffer[buffer_idx], filename)

    reservoir.save_buffer_all()
