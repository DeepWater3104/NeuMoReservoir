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
    from neuron.units import ms, mV
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


    # --- 3. DataGenerator Type Specific Setup ---
    if params['task']['datagenerator_type'] == "TI46word_datagenerator":
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

        spike_trains = datagenerator.get_spike_trains(data_idx=0, mode='train')
    elif params['task']['datagenerator_type'] == "sin_datagenerator":
        from DataGenerator import sin_datagenerator
        
        print(f"--- Preparing Sine Wave Data (Frequency: {params['task']['freq']} Hz) ---")
        
        # 正弦波ジェネレータの初期化
        # params['task'] に freq や bin_width が含まれている想定です
        datagenerator = sin_datagenerator(
            params=params['task'],
            freq=params['task']['freq'],
            prng=prng
        )
        
        # bin_width の同期（必要であれば）
        params['bin_width'] = datagenerator.bin_width
        print(f"DataGenerator initialized for Sine Wave: {params['bin_width']} ms bins")

        from NeuronalReservoir_prediction import neuronalreservoir_prediction
        neuronalreservoir = neuronalreservoir_prediction(cell, prng, params)

        # 動作確認用に最初のスパイクを取得してみる
        spike_trains = datagenerator.get_spike_trains()
        nrn.finitialize(-65 * mV)
        neuronalreservoir.resister_spike_events(spike_trains)
        neuronalreservoir.generate_dynamics((params['task']['len_transientdata']+params['task']['len_trainingdata']+params['task']['len_testdata']) * params['bin_width'])
        state_vars = neuronalreservoir.get_binned_states()
        print(state_vars.shape)
        print(state_vars[params['task']['len_transientdata']:params['task']['len_transientdata']+params['task']['len_trainingdata'], :].shape)
        print(datagenerator.trainingdata_target.shape)
        neuronalreservoir.optimize(state_vars[params['task']['len_transientdata']:params['task']['len_transientdata']+params['task']['len_trainingdata'], :], datagenerator.trainingdata_target)
        output = neuronalreservoir.readout(state_vars)

        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error
        
        # --- 1. 時間軸と区切りの定義 ---
        t_full = datagenerator.t 
        len_trans = params['task']['len_transientdata']
        len_train = params['task']['len_trainingdata']
        len_test = params['task']['len_testdata']
        bin_w = params['bin_width']
        
        # --- 2. プロットの作成（4段構成） ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # (1) 入力スパイクラスター（原因の可視化）
        #print(spike_trains.shape)
        #for spike_times, i in enumerate(spike_trains):
        #    axes[0].vlines(spike_times, i + 0.6, i + 1.4, color='royalblue', linewidth=1.0)
        #axes[0].set_ylabel('Input ID')
        #axes[0].set_title('Input Spike Raster')
        # --- 修正版：(num_spikes, 2) 形式の spike_trains をプロットする ---

        # axes[0] (入力ラスター) の部分を以下に差し替えてください
        if spike_trains.size > 0:
            # spike_trains[:, 0] が時刻、spike_trains[:, 1] がニューロンID
            axes[0].scatter(spike_trains[:, 0], spike_trains[:, 1], 
                            marker='|', color='royalblue', s=20, linewidths=1.0)
            
            # y軸の範囲をニューロン数に合わせる
            num_inputs = int(np.max(spike_trains[:, 1])) + 1 if spike_trains.size > 0 else 1
            axes[0].set_ylim(-0.5, num_inputs - 0.5)
        
        axes[0].set_ylabel('Input ID')
        axes[0].set_title('Input Spike Raster')
        
        # (2) リザーバ膜電位（内部ダイナミクスの可視化）
        if hasattr(neuronalreservoir, 'v_rec_list'):
            # 全細胞プロットすると重い場合は [::n] で間引いてください
            for v_rec in neuronalreservoir.v_rec_list:
                axes[1].plot(neuronalreservoir.t_rec, v_rec, alpha=0.5, linewidth=0.5)
        axes[1].set_ylabel('V [mV]')
        axes[1].set_title('Reservoir Membrane Potential')
        
        # (3) ビニングされた状態（特徴量の可視化）
        # state_vars は [全期間, ユニット数] の形状を想定
        axes[2].plot(t_full, state_vars[:, :5], alpha=0.8) 
        axes[2].set_ylabel('Binned State')
        axes[2].set_title('Readout Input (First 5 units)')
        
        # (4) 出力（予測） vs 教師データ
        target_all = datagenerator.get_targetdata()
        input_all = datagenerator.get_inputdata()[len_trans:]
        t_output = t_full[len_trans:]
        
        axes[3].plot(t_output, target_all, 'k-', alpha=0.3, label='Target')
        axes[3].plot(t_output, input_all, 'k-', alpha=0.3, label='Input')
        # 学習フェーズ
        axes[3].plot(t_output[:len_train], output[len_trans:len_trans+len_train], 'b', label='Train Pred', linewidth=1)
        # テストフェーズ（インデックス修正済み）
        axes[3].plot(t_output[len_train:], output[len_trans+len_train:], 'r', label='Test Pred', linewidth=1.2)
        
        axes[3].axvline(x=t_output[len_train], color='green', linestyle='--', label='Train/Test Split')
        axes[3].set_ylabel('Output')
        axes[3].set_xlabel('Time [ms]')
        axes[3].legend(loc='upper right', fontsize='small')
        
        # --- 3. 表示範囲の調整（拡大表示） ---
        start_zoom = (len_trans + len_train // 2) * bin_w
        end_zoom = (len_trans + len_train + len_test) * bin_w
        for ax in axes:
            ax.set_xlim(start_zoom, end_zoom)
            ax.grid(axis='x', alpha=0.2)
        
        plt.tight_layout()
        plt.savefig("reservoir_flow_analysis.png")
        plt.show()
        
        # 数値の確認も忘れずに
        mse_test = mean_squared_error(target_all[len_train:len_train+len_test], output[len_train:len_train+len_test])
        print(f"Test MSE: {mse_test:.8f}")

if __name__ == "__main__":
    main()
