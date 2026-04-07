from NeuronalReservoir import neuronalreservoir
from neuron import h as nrn
from neuron.units import ms, mV

class neuronalreservoir_classification(neuronalreservoir):
    def __init__(self, cell, prng, params):
        self.cell = cell

        self.prng = prng
        nrn.celsius = 36

        self.plot_all              = params['plot_all']
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
        self.batches_to_save_idx   = params['batches_to_save_idx']
        self.batches_to_save_mode  = params['batches_to_save_mode']

    def create_records_for_buffer(self):
        if self.plot_all:
            self.buffer_variable_list = []
            if self.record_target == 'potential':
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
            elif self.record_target == 'calcium_acum':
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

    def save_to_buffer(self, mode, data_idx, spike_train, datagenerator):
        buffer = {}
        buffer['mode']         = mode
        buffer['data_idx']     = data_idx

        buffer['variables']    = []
        v_rec_np = np.array([np.array(v_rec.to_python()) for v_rec in self.v_rec_list])
        buffer['variables'].append(v_rec_np)
        v_rec_np = np.stack([v_rec.to_python() for v_rec in self.buffer_variable_list], axis=1)
        buffer['variables'].append(v_rec_np)
        buffer['t_rec']        = np.array(self.t_rec.to_python())
    
        num_input_neurons = len(spike_train)
        num_spikes = 0
        for neuron_idx in range(num_input_neurons):
            num_spikes += len(spike_train[neuron_idx])
        
        # スパイクがない場合のNaN処理を考慮 (例: 1以上のスパイクがある場合のみ実行)
        if num_spikes > 0:
            spike_times = np.zeros(num_spikes)
            spike_neurons = np.zeros(num_spikes)
            spike_idx = 0
            for neuron_idx, spike_train in enumerate(spike_train):
                for spike_time in spike_train:
                    spike_times[spike_idx]    = spike_time
                    spike_neurons[spike_idx] = neuron_idx
                    spike_idx += 1

            buffer['input'] = {}
            buffer['input']['spike_times']   = spike_times
            buffer['input']['spike_neurons'] = spike_neurons

        else:
            buffer['input'] = None

        if mode=="training":
            buffer['TrueLabel']     = datagenerator.train_label[buffer['data_idx']]
            start_bin_idx           = sum(datagenerator.len_data[:-1])
            end_bin_idx             = sum(datagenerator.len_data)-1
            buffer['target']        = datagenerator.trainingdata_target[start_bin_idx:end_bin_idx+1, :]
            buffer['output']        = self.readout(self.train_state_vars[start_bin_idx:end_bin_idx+1, :])
            buffer['time_output']   = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width
            buffer['reservoir_state'] = self.train_state_vars[start_bin_idx:end_bin_idx+1, :]
        elif mode=="test":
            buffer['TrueLabel']     = datagenerator.test_label[buffer['data_idx']]
            start_bin_idx           = sum(datagenerator.len_data[datagenerator.train_dataset_size:-1])
            end_bin_idx             = sum(datagenerator.len_data[datagenerator.train_dataset_size:])-1
            buffer['target']        = datagenerator.testdata_target[start_bin_idx:end_bin_idx+1, :]
            buffer['output']        = self.readout(self.test_state_vars[start_bin_idx:end_bin_idx+1, :])
            buffer['time_output']   = np.arange(start_bin_idx, end_bin_idx+1) * self.bin_width + sum(datagenerator.len_data[:datagenerator.train_dataset_size]) * self.bin_width
            buffer['reservoir_state'] = self.test_state_vars[start_bin_idx:end_bin_idx+1, :]

        self.data_buffer.append(buffer)

    def overwrite_buffer_after_optimized(self, datagenerator):
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
        
        winner_neuron_history = np.zeros(output_within_batch.shape[0])
        for bin_idx in range(output_within_batch.shape[0]):
            winner_neuron = np.argmax(output_within_batch[bin_idx, :])
            winner_neuron_history[bin_idx] = winner_neuron

        unique, freq = np.unique(winner_neuron_history, return_counts=True)
        mode = unique[np.argmax(freq)]
        return mode

    def get_classification_result(self, mode, datagenerator):
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

    def save_buffer_all(self):
        for buffer_idx, buffer in enumerate(self.data_buffer):
            filename = "./data/buffer" + str(buffer_idx).zfill(2) + ".npz"
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
    NUM_OUTPUT_NEURONS = buffer['output'].shape[1]
    
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
        for i in range(buffer['reservoir_state'].shape[1]):
            ax_reservoir.plot(buffer['time_output'], buffer['reservoir_state'][:, i], linewidth=1.2) # 線幅を調整

        ax_reservoir_list.append(ax_reservoir)

        ax_reservoir = fig.add_subplot(gs_reservoir[2, 0])
        for i in range(buffer['variables'][1].shape[1]):
            ax_reservoir.plot(buffer['t_rec'], buffer['variables'][1][:, i], linewidth=1.2) # 線幅を調整
            ax_reservoir.set_xlabel('Time [ms]')

        ax_reservoir_list.append(ax_reservoir)
    else:
        ax_reservoir = fig.add_subplot(gs[0, 1])
        ax_reservoir.set_title('Reservoir Neuron States', fontsize=14) # 英語タイトル
        ax_reservoir.set_xlabel('Time (ms)', fontsize=12) # X軸ラベル
        ax_reservoir.set_ylabel('Membrane Potential (mV)', fontsize=12) # Y軸ラベル

        # buffer['v_rec']が存在し、形状が適切であることを確認
        if buffer['variables'][0].shape[0] > 0:
            for state_idx in range(buffer['variables'][0].shape[1]):
                ax_reservoir.plot(buffer['t_rec'], buffer['variables'][0][:, state_idx], linewidth=0.8) # 線幅を調整
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
        ax_output.plot(buffer['time_output'], buffer['output'][:, i], label="Output", linewidth=1.2) # 線幅を調整
        ax_output.plot(buffer['time_output'], buffer['target'][:, i], label="Ground Truth", linestyle='--', linewidth=1.2) # ターゲットを点線に

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


