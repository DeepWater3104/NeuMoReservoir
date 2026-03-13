import numpy as np
from neuron import h as nrn
from neuron.units import ms, mV

def create_datasyn_correspondance_toydata(self):
    self.exc_dataresolution = (max(self.get_inputdata()) - min(self.get_inputdata())) / (self.exc_num_syns - self.exc_num_synchro_syns+1)
    for i in range(self.exc_num_syns):
        minimum = i * self.exc_dataresolution + min(self.get_inputdata()) - (self.exc_num_synchro_syns-1)*self.exc_dataresolution
        maximum = (i+1)*self.exc_dataresolution + min(self.get_inputdata())

        # care about the boundary condition
        data_range = {'min': minimum, 'max': maximum}
        self.dataexcsyn_dict.append(data_range)

    self.inh_dataresolution = (max(self.get_inputdata()) - min(self.get_inputdata())) / (self.inh_num_syns - self.inh_num_synchro_syns+1)
    for i in range(self.inh_num_syns):
        minimum = i * self.inh_dataresolution + min(self.get_inputdata()) - (self.inh_num_synchro_syns-1)*self.inh_dataresolution
        maximum = (i+1)*self.inh_dataresolution + min(self.get_inputdata())

        # care about the boundary condition
        data_range = {'min': minimum, 'max': maximum}
        self.datainhsyn_dict.append(data_range)

def get_dend_and_soma(cell):
    all_dend_soma = []
    for sec in cell.somatic:
        all_dend_soma.append(sec)
    for sec in cell.basal:
        all_dend_soma.append(sec)
    for sec in cell.apical:
        all_dend_soma.append(sec)
    return all_dend_soma
     

def create_synapses_toydata(self, cell):
    total_length = 0
    cumulative_length_dict = []
    all_dend_soma = get_dend_and_soma(cell)
    for sec in all_dend_soma:
        cumulative_length = {'min': total_length, 'max': total_length+sec.L}
        cumulative_length_dict.append(cumulative_length)
        total_length += sec.L

    self.exc_syn_list = []
    self.inh_syn_list = []
    for i in range(self.exc_num_syns):
        syn_loc = total_length * self.prng.random()

        for index, sec in enumerate(all_dend_soma):
            if cumulative_length_dict[index]['min'] <= syn_loc and syn_loc < cumulative_length_dict[index]['max']:
                syn = nrn.Exp2Syn(sec((syn_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])))
                syn.tau1 = self.exc_syn_tau1 * ms
                syn.tau2 = self.exc_syn_tau2 * ms
                syn.e = -10. * ms
                self.exc_syn_list.append(syn)

    for i in range(self.inh_num_syns):
        syn_loc = total_length * self.prng.random()

        for index, sec in enumerate(all_dend_soma):
            if cumulative_length_dict[index]['min'] <= syn_loc and syn_loc < cumulative_length_dict[index]['max']:
                syn = nrn.Exp2Syn(sec((syn_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])))
                syn.tau1 = self.inh_syn_tau1 * ms
                syn.tau2 = self.inh_syn_tau2 * ms
                syn.e = -80. * ms
                self.inh_syn_list.append(syn)

    self.dataexcsyn_dict = []
    self.datainhsyn_dict = []

def test_distance_accuracy(self, cell):
    from neuron import h
    # 1. 基点を再設定（念のため）
    nrn.distance(0, 0.5, sec=cell.soma[0])
    
    print("--- NEURON Distance Validation ---")
    
    # テストA: 細胞体自身の距離（0付近になるはず）
    d_soma = h.distance(cell.soma[0](0.5))
    print(f"Distance at Soma(0.5): {d_soma:.4f} um (Expected: 0.0)")

    # テストB: 細胞体の端（L/2 になるはず）
    d_soma_end = h.distance(cell.soma[0](1.0))
    print(f"Distance at Soma(1.0): {d_soma_end:.4f} um (Expected: ~{cell.soma[0].L/2:.1f})")

    # テストC: 最初の樹状突起の根元
    # get_dend_and_somaが返す最初のセクションの開始点
    first_dend = next(iter(get_dend_and_soma(cell)))
    d_dend_start = h.distance(first_dend(0.0))
    print(f"Distance at First Dendrite start: {d_dend_start:.4f} um")

def create_synapses_hetero(self, cell, condition):
    test_distance_accuracy(self, cell)
    # 距離計算の基準点（細胞体）を設定
    nrn.distance(0, 0.5, sec=cell.soma[0]) 
    #nrn.distance(cell.soma[0](0.5))
    all_segs = []
    areas = []
    distances = []

    # 1. 全セグメントの情報を収集
    for sec in get_dend_and_soma(cell):
        for seg in sec:
            all_segs.append(seg)
            areas.append(seg.area())
            distances.append(nrn.distance(seg))

    areas = np.array(areas)
    weights = np.zeros(areas.shape)
    distances = np.array(distances)

    # 2. 条件に応じた重み付け
    if condition == "distal-dense":  # 遠位に密集
        mu = 600.0  # 遠く（細胞の最大長に合わせて調整）
        sigma = 100.0
        weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
        
    elif condition == "proximal-dense":  # 近位に密集
        mu = 0.0   # 細胞体に近い
        sigma = 100.0
        weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
    elif condition == "proximal-sparse":  # 近位に密集
        mu = 000.0   # 細胞体に近い
        sigma = 300.0
        weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
    elif condition == "distal-sparse":  # 遠位に密集
        mu = 600.0  # 遠く（細胞の最大長に合わせて調整）
        sigma = 300.0
        weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
        
    elif condition == "random":  # 面積に比例した一様分布
        weights = areas

    # 3. 重みの正規化（合計を1にする）
    prob = weights / np.sum(weights)

    # 4. シナプスの配置（興奮性を例に）
    # 重みに基づいてセグメントをランダムに選択
    chosen_indices = self.prng.choice(len(all_segs), size=self.exc_num_syns, p=prob)

    self.exc_syn_list = []
    for idx in chosen_indices:
        seg = all_segs[idx]
        syn = nrn.Exp2Syn(seg)
        # --- パラメータ設定 ---
        syn.tau1 = self.exc_syn_tau1
        syn.tau2 = self.exc_syn_tau2
        syn.e = -10.0
        self.exc_syn_list.append(syn)

    chosen_indices = self.prng.choice(len(all_segs), size=self.inh_num_syns, p=prob)

    self.inh_syn_list = []
    for idx in chosen_indices:
        seg = all_segs[idx]
        syn = nrn.Exp2Syn(seg)
        # --- パラメータ設定 ---
        syn.tau1 = self.inh_syn_tau1
        syn.tau2 = self.inh_syn_tau2
        syn.e = -80.0
        self.inh_syn_list.append(syn)

    check_synapse_stats_v2(self)

def check_synapse_stats_v2(self):
    from neuron import h
    import numpy as np
    from collections import Counter

    # 1. 各シナプスのセグメントオブジェクトを直接取得
    exc_segs = [syn.get_segment() for syn in self.exc_syn_list]
    
    # 2. セグメントごとの重複数をカウント
    seg_counts = Counter(exc_segs)
    occ_segs_count = len(seg_counts) # シナプスが存在するユニークなセグメント数
    max_overlap = max(seg_counts.values()) if seg_counts else 0

    # 3. 距離データの取得
    exc_dists = [h.distance(seg) for seg in exc_segs]

    print(f"\n--- Detailed Placement Check ({self.condition}) ---")
    print(f"Total Synapses: {len(exc_dists)}")
    print(f"Occupied Segments: {occ_segs_count}")
    print(f"Max Overlap in one segment: {max_overlap} synapses")
    print(f"Average Density: {len(exc_dists)/occ_segs_count:.2f} syns/segment (among occupied)")

    # 4. 距離ごとの「シナプス数」vs「セグメント数」
    print("\n[Distance: Synapses vs Occupied Segments]")
    bins = np.arange(0, 1100, 100)
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        
        # この範囲にある全シナプス
        syns_in_range = [d for d in exc_dists if lower <= d < upper]
        # この範囲にある、シナプスを持つユニークなセグメント
        segs_in_range = {s for s in exc_segs if lower <= h.distance(s) < upper}
        
        if len(syns_in_range) > 0:
            bar = "#" * int(len(syns_in_range) / len(self.exc_syn_list) * 40)
            print(f"{lower:4.0f}-{upper:4.0f} um: {bar}")
            print(f"    -> Synapses: {len(syns_in_range)}, Segments: {len(segs_in_range)}")

def check_synapse_stats(self):
    from neuron import h
    import numpy as np

    # 1. 各シナプスの距離を計算してリスト化
    exc_dists = [h.distance(syn.get_segment()) for syn in self.exc_syn_list]
    inh_dists = [h.distance(syn.get_segment()) for syn in self.inh_syn_list]

    # 2. 統計量を表示
    print(f"\n--- Synapse Placement Check ({self.condition}) ---")
    print(f"Excitation: count={len(exc_dists)}, mean={np.mean(exc_dists):.1f}um, std={np.std(exc_dists):.1f}um")
    print(f"            min={np.min(exc_dists):.1f}um, max={np.max(exc_dists):.1f}um")
    print(f"Inhibition: count={len(inh_dists)}, mean={np.mean(inh_dists):.1f}um, std={np.std(inh_dists):.1f}um")
    
    # 3. 簡易的な分布の目視チェック（テキストベース）
    print("\n[Distance Distribution (Excitation)]")
    hist, bin_edges = np.histogram(exc_dists, bins=10, range=(0, 1000))
    for i in range(len(hist)):
        bar = "#" * int(hist[i] / max(hist) * 20) if max(hist) > 0 else ""
        print(f"{bin_edges[i]:4.0f}-{bin_edges[i+1]:4.0f} um: {bar} ({hist[i]})")



def connect_synapses_toydata(self):
    self.exc_nc_list = []
    for syn in self.exc_syn_list:
        nc_tosyn = nrn.NetCon(None, syn)
        nc_tosyn.weight[0] = self.exc_syn_weight
        self.exc_nc_list.append(nc_tosyn)

    self.inh_nc_list = []
    for syn in self.inh_syn_list:
        nc_tosyn = nrn.NetCon(None, syn)
        nc_tosyn.weight[0] = self.inh_syn_weight
        self.inh_nc_list.append(nc_tosyn)


def resister_inputevent_toNetCon_toydata(self):
    for data_index, data in enumerate(self.get_inputdata()):
        time = self.bin_width * data_index  
        
        # search for corresponding synapse
        for nc_index, nc in enumerate(self.exc_nc_list):
            if self.dataexcsyn_dict[nc_index]['min'] <= data and data < self.dataexcsyn_dict[nc_index]['max']:
                # only when the data is within range, resister event at time
                #print(time)
                nc.event(time) # need to be verified

        for nc_index, nc in enumerate(self.inh_nc_list):
            if self.datainhsyn_dict[nc_index]['min'] <= data and data < self.datainhsyn_dict[nc_index]['max']:
                # only when the data is within range, resister event at time
                #print(time)
                nc.event(time) # need to be verified


class datagenerator():
    def __init__(self, params):
        # hold parameters as instance's variables
        self.bin_width         = params['bin_width'] # bin_width determines the sampling frequency
        self.len_transientdata = params['len_transientdata']
        self.len_trainingdata  = params['len_trainingdata']
        self.len_testdata      = params['len_testdata']
        self.exc_num_syns         = params['exc_num_syns']
        self.exc_num_synchro_syns = params['exc_num_synchro_syns']
        self.exc_syn_weight       = params['exc_syn_weight']
        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.inh_num_syns         = params['inh_num_syns']
        self.inh_num_synchro_syns = params['inh_num_synchro_syns']
        self.inh_syn_weight       = params['inh_syn_weight']
        self.inh_syn_tau1         = params['inh_syn_tau1']
        self.inh_syn_tau2         = params['inh_syn_tau2']

        self.t = self.bin_width*np.arange(self.len_transientdata+self.len_trainingdata+self.len_testdata)


        if hasattr(self, 'time_delay'): # true when the instance generate continuos time system's timeseries
            self.transientdata_input  = self.generate_data(0, self.len_transientdata, 0)
            self.trainingdata_input   = self.generate_data(self.len_transientdata, self.len_transientdata+self.len_trainingdata, 0)
            self.trainingdata_target  = self.generate_data(self.len_transientdata, self.len_transientdata+self.len_trainingdata, self.time_delay)
            self.testdata_input       = self.generate_data(self.len_transientdata+self.len_trainingdata, self.len_transientdata+self.len_trainingdata+self.len_testdata, 0)
            self.testdata_target      = self.generate_data(self.len_transientdata+self.len_trainingdata, self.len_transientdata+self.len_trainingdata+self.len_testdata, self.time_delay)
   
    def generate_data(self, start_binindex, end_binindex, delay):
        raise NotImplementedError("generate_data() is not implemented.")

    def get_targetdata(self):
        return np.concatenate([self.trainingdata_target, self.testdata_target])

    def get_inputdata(self):
        return np.concatenate([self.transientdata_input, self.trainingdata_input, self.testdata_input])

    def create_synapses(self, cell):
        raise NotImplementedError("create_synapses() is not implemented.")

    def connect_synapses(self):
        raise NotImplementedError("connect_synapses() is not implemented.")

    def resister_inputevent_toNetCon(self, netcon_list):
        # this function must be called after nrn.finitialize()
        raise NotImplementedError("resister_spikes() is not implemented.")


class sin_datagenerator(datagenerator):
    def __init__(self, params, freq, prng):
        self.time_delay        = params['time_delay'] 
        self.freq              = freq
        self.prng = prng
        super().__init__(params)

    def generate_data(self, start_binindex, end_binindex, delay):
        data = np.sin(2*np.pi*self.freq*(self.t[start_binindex:end_binindex]-delay))
        return data

    def get_targetdata(self):
        return np.concatenate([self.trainingdata_target, self.testdata_target])

    def get_inputdata(self):
        return np.concatenate([self.transientdata_input, self.trainingdata_input, self.testdata_input])

    def create_synapses(self, cell):
        create_synapses_toydata(self, cell)
        create_datasyn_correspondance_toydata(self)

    def connect_synapses(self):
        connect_synapses_toydata(self)
       
    def resister_inputevent_toNetCon(self):
        resister_inputevent_toNetCon_toydata(self)


class MackeyGlass_datagenerator(datagenerator):
    def __init__(self, params, prng):
        self.len_delay = params['len_delay']
        super().__init__(params)
        self.prng = prng
        
        len_transientdata_mg = 1000
        import signalz
        self.temp_data = signalz.mackey_glass(len_transientdata_mg+self.len_transientdata+self.len_trainingdata+self.len_testdata+self.len_delay, a=0.2 , b=0.8, c=0.9, d=23, e=10, initial=0.5*self.prng.random()+0.5)[len_transientdata_mg:]

        self.transientdata_input = self.generate_data(0, self.len_transientdata, 0)
        self.trainingdata_input   = self.generate_data(self.len_transientdata, self.len_transientdata+self.len_trainingdata, 0)
        self.trainingdata_target  = self.generate_data(self.len_transientdata, self.len_transientdata+self.len_trainingdata, self.len_delay)
        self.testdata_input       = self.generate_data(self.len_transientdata+self.len_trainingdata, self.len_transientdata+self.len_trainingdata+self.len_testdata, 0)
        self.testdata_target      = self.generate_data(self.len_transientdata+self.len_trainingdata, self.len_transientdata+self.len_trainingdata+self.len_testdata, self.len_delay)

    def generate_data(self, start_binindex, end_binindex, delay):
        return self.temp_data[start_binindex+delay:end_binindex+delay]

    def get_targetdata(self):
        return np.concatenate([self.trainingdata_target, self.testdata_target])

    def get_inputdata(self):
        return np.concatenate([self.transientdata_input, self.trainingdata_input, self.testdata_input])

    def create_synapses(self, cell):
        create_synapses_toydata(self, cell)
        create_datasyn_correspondance_toydata(self)

    def connect_synapses(self):
        connect_synapses_toydata(self)
       
    def resister_inputevent_toNetCon(self):
        resister_inputevent_toNetCon_toydata(self)

from lyon.calc import LyonCalc
import librosa

class TI46word_datagenerator(datagenerator):
    def __init__(self, params, prng, trainseq_code=None, testseq_code=None, path_to_dataset="../dataset/ti46/ti20/"):
        self.prng = prng

        self.exc_num_syns         = params['exc_num_syns']
        self.exc_syn_weight       = params['exc_syn_weight']
        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.syn_mechanisms       = params['syn_mechanisms']
        self.IP3stim              = params['IP3stim']

        self.maximum_firing_rate = params['maximum_firing_rate']

        self.inh_num_syns         = params['inh_num_syns']
        self.inh_syn_weight       = params['inh_syn_weight']
        self.inh_syn_tau1         = params['inh_syn_tau1']
        self.inh_syn_tau2         = params['inh_syn_tau2']

        self.mGluR_stim_train_idx = 0
        self.mGluR_stim_train = []

        self.path_to_dataset = path_to_dataset
        self.num_outputs = params['num_outputs']
        self.decimation_factor = params['decimation_factor']
        self.sr = params['sampling_frequency']
        self.bin_width = self.decimation_factor * (10**3/self.sr) # this parameter is determined using decimation_factor

        self.prompts = [i for i in range(10)]
        self.prompt_codes = [f"{i:02d}" for i in self.prompts]
        self.train_session_codes = ["se"]
        self.test_session_codes  = ["s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        self.train_token_codes    = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
        self.test_token_codes   = ["t0", "t1"]
        self.speaker_codes = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]

        if trainseq_code == None:
            self.trainseq_promptcode   = [prompt 
                                          for prompt in self.prompts
                                          for i in range(len(self.speaker_codes)) 
                                          for j in range(len(self.train_session_codes)) 
                                          for k in range(len(self.train_token_codes))] 
            self.trainseq_speakercode = [speaker
                                          for i in range(len(self.prompts))
                                          for speaker in self.speaker_codes
                                          for j in range(len(self.train_session_codes)) 
                                          for k in range(len(self.train_token_codes))] 
            self.trainseq_sessioncode  = [session
                                          for i in range(len(self.prompts))
                                          for j in range(len(self.speaker_codes)) 
                                          for session in self.train_session_codes
                                          for k in range(len(self.train_token_codes))] 
            self.trainseq_tokencode    =   [token
                                          for i in range(len(self.prompts))
                                          for j in range(len(self.speaker_codes)) 
                                          for k in self.train_session_codes
                                          for token in self.train_token_codes]
        else:
            self.trainseq_promptcode  = trainseq_code['trainseq_promptcode']
            self.trainseq_speakercode = trainseq_code['trainseq_speakercode']
            self.trainseq_sessioncode = trainseq_code['trainseq_sessioncode'] 
            self.trainseq_tokencode   = trainseq_code['trainseq_tokencode']
                           
        if testseq_code == None:
            self.testseq_promptcode   = [prompt 
                                         for prompt in self.prompts
                                         for i in range(len(self.speaker_codes)) 
                                         for j in range(len(self.test_session_codes)) 
                                         for k in range(len(self.test_token_codes))] 
            self.testseq_speaker_code = [speaker
                                         for i in range(len(self.prompts))
                                         for speaker in self.speaker_codes
                                         for j in range(len(self.test_session_codes)) 
                                         for k in range(len(self.test_token_codes))] 
            self.testseq_sessioncode  = [session
                                         for i in range(len(self.prompts))
                                         for j in range(len(self.speaker_codes)) 
                                         for session in self.test_session_codes
                                         for k in range(len(self.test_token_codes))] 
            self.testseq_tokencode    = [token
                                         for i in range(len(self.prompt_codes))
                                         for j in range(len(self.speaker_codes)) 
                                         for k in range(len(self.test_session_codes))
                                         for token in self.test_token_codes]
        else:
            self.testseq_promptcode  = testseq_code['testseq_promptcode']
            self.testseq_speakercode = testseq_code['testseq_speakercode']
            self.testseq_sessioncode = testseq_code['testseq_sessioncode'] 
            self.testseq_tokencode   = testseq_code['testseq_tokencode']


        self.train_dataset_size    = len(self.trainseq_promptcode)
        self.test_dataset_size     = len(self.testseq_promptcode)
        self.total_dataset_size    = self.train_dataset_size + self.test_dataset_size

        self.len_data = [] # this parameter is determined using decimation_factor
        self.trainingdata_target = np.zeros((self.num_outputs, 0))
        self.testdata_target     = np.zeros((self.num_outputs, 0))

    def generate_target_within_batch(self, data_idx, prompt):
        target_within_batch = np.zeros((self.num_outputs, self.len_data[data_idx]))
        target_within_batch[prompt, :] = np.ones(self.len_data[data_idx])
        return target_within_batch # should be two-dimensional np array

    def get_filepath(self, prompt, speaker, session, token):
        if session in self.train_session_codes:
            # train
            return self.path_to_dataset + "train/" + speaker + "/" + prompt+speaker+session+token + ".sph"
        elif session in self.test_session_codes:
            # test
            return self.path_to_dataset + "test/"  + speaker + "/" + prompt+speaker+session+token + ".sph"
    
    def read_dataset(self, prompt, speaker, session, token):
        clip_dir = self.get_filepath(prompt, speaker, session, token)
        waveform, sr = librosa.load(clip_dir,sr=None) # load sph file with original sampling frequency
        waveform = waveform.astype(np.float64) #  This process is required for filtering with lyon passive cochlear filter
        return np.arange(len(waveform)) * (10**3 / sr), sr, waveform
    
    def get_cochleogram(self, prompt, speaker, session, token):
        time, sr, waveform = self.read_dataset(prompt, speaker, session, token)
        bin_width = self.decimation_factor * (10**3 / sr)
        calc = LyonCalc()
        coch = calc.lyon_passive_ear(waveform, sr, self.decimation_factor)
        time_coch = np.arange(coch.shape[0]) * bin_width
        return time_coch, coch, 

    def generate_data(self, data_idx, mode):
        if mode=="training":
            prompt  = self.trainseq_promptcode[data_idx]
            speaker = self.trainseq_speakercode[data_idx]
            session = self.trainseq_sessioncode[data_idx]
            token   = self.trainseq_tokencode[data_idx]
            time_coch, coch = self.get_cochleogram(self.prompt_codes[prompt], speaker,  session, token)
            self.len_data.append(np.size(time_coch))
            self.trainingdata_target = np.concatenate([self.trainingdata_target, self.generate_target_within_batch(data_idx,  prompt)], axis=1)
            return time_coch, coch
        elif mode=="test":
            prompt  = self.testseq_promptcode[data_idx]
            speaker = self.testseq_speakercode[data_idx]
            session = self.testseq_sessioncode[data_idx]
            token   = self.testseq_tokencode[data_idx]
            time_coch, coch = self.get_cochleogram(self.prompt_codes[prompt], speaker,  session, token)
            self.len_data.append(np.size(time_coch))
            self.testdata_target = np.concatenate([self.testdata_target, self.generate_target_within_batch(self.train_dataset_size+data_idx,  prompt)], axis=1)
            return time_coch, coch

    def create_synapses(self, cell, ip3=None, cyt=None):
        if self.syn_mechanisms=='ionotropic':
            create_synapses_toydata(self, cell)
        elif self.syn_mechanisms=='ionotropic_and_metabotropic':
            create_synapses_ionotoropic_and_metabotropic(self, cell, ip3, cyt)


    def connect_synapses(self):
        connect_synapses_toydata(self)

    def normalize_coch(self, coch):
        maximum_activity = np.max(coch)
        return self.maximum_firing_rate * 10e-3 * coch * self.bin_width / maximum_activity

    def convert_to_spiketrain(self, time_coch, normalized_coch, encoding="poissonian"):
        num_channels = normalized_coch.shape[1]
        spike_trains = []
        for channel_idx in range(num_channels):
            auditory_activity = normalized_coch[:, channel_idx]
            spike_train = []
            for time_idx, activity in enumerate(auditory_activity):
                 if self.prng.random() < activity:
                    spike_train.append(time_coch[time_idx])
            spike_trains.append(spike_train)
        return spike_trains
       
    def resister_inputevent_toNetCon(self, data_idx, mode):
        time_coch, coch = self.generate_data(data_idx, mode)
        spike_trains = self.convert_to_spiketrain(time_coch, self.normalize_coch(coch), "poissonian")
        if self.syn_mechanisms=='ionotropic':
            for synapse_idx, spike_train in enumerate(spike_trains):
                for spike_time in spike_train:
                    spike_time = self.bin_width * sum(self.len_data[:-1]) + spike_time
                    self.exc_nc_list[synapse_idx].event(spike_time)
            return spike_trains


import numpy as np
import random

# 親クラス datagenerator がある前提です
# class datagenerator: pass 

class RandomPattern_datagenerator(datagenerator):
    def __init__(self, params, prng, train_ratio=0.8):
        self.prng = prng
        
        # --- NEURON Synapse Parameters (継承) ---
        self.exc_num_syns         = params['exc_num_syns']
        self.exc_syn_weight       = params['exc_syn_weight']
        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.syn_mechanisms       = params['syn_mechanisms']
        self.IP3stim              = params.get('IP3stim', 0) # keyがない場合の対策

        self.firing_rate = params['firing_rate']

        self.inh_num_syns         = params['inh_num_syns']
        self.inh_syn_weight       = params['inh_syn_weight']
        self.inh_syn_tau1         = params['inh_syn_tau1']
        self.inh_syn_tau2         = params['inh_syn_tau2']

        self.mGluR_stim_train_idx = 0
        self.mGluR_stim_train = []

        # --- Random Task Specific Parameters ---
        self.num_outputs  = params['num_outputs'] # クラス数（例: 10クラス分類なら10）
        self.num_classes  = params['num_outputs'] # クラス数（例: 10クラス分類なら10）
        self.num_channels = params['exc_num_syns'] # 入力次元数（シナプス数と一致させる）
        self.condition    = params['condition'] # 
        self.jitter_std   = params['jitter_std'] # 
        
        self.pattern_duration_ms = params.get('pattern_duration_ms', 1000) # 1パターンの長さ(ms)
        self.poisson_dt= params.get('poisson_dt', 1)
        self.bin_width = params['bin_width']

        # データセットのサイズ定義
        self.num_test_per_class = params['num_test_per_class']
        self.train_dataset_size = self.num_classes
        self.test_dataset_size  = self.num_classes * self.num_test_per_class
        self.total_dataset_size = self.train_dataset_size + self.test_dataset_size

        # インデックスの管理
        # 例: [0, 0, ..., 1, 1, ..., 9, 9] のようなラベル配列を作成しシャッフル
        # ランダムにシャッフルしてTrain/Testに分割
        self.train_labels = np.arange(self.num_classes)
        self.test_labels = np.repeat(np.arange(self.num_classes), self.num_test_per_class)
        permuted_indices = self.prng.permutation(len(self.test_labels))
        self.test_labels = self.test_labels[permuted_indices]

        self.len_data = [] 
        self.trainingdata_target = np.zeros((self.num_classes, 0))
        self.testdata_target     = np.zeros((self.num_classes, 0))

        # --- テンプレートパターンの生成 ---
        # 各クラスに対応する「発火確率マップ（cochleogram相当）」をあらかじめ作成して固定する
        # Shape: [num_classes, num_channels, time_steps]
        num_steps = int(self.pattern_duration_ms / self.poisson_dt)
        p_spikes  = self.firing_rate * self.poisson_dt * 1e-3
        self.class_templates = self._generate_PoissonSpikeTrain(num_steps, p_spikes)
        #self.class_templates = self._generate_OverlappingStructuredTemplates(num_steps, p_spikes) # for distance analysis
        #self.class_templates = self._generate_BoostedTemplates(num_steps, p_spikes) # for distance analysis

    def _generate_PoissonSpikeTrain(self, num_steps, p_spikes):
        """
        各クラスごとの固定されたランダムパターン（テンプレート）を生成する。
        """
        templates = []
        for _ in range(self.num_classes):
            #spike_matrix = np.random.rand(num_channels, num_steps) < p_spike
            spike_matrix = self.prng.random((self.num_channels, num_steps)) < p_spikes

            # 2. True(スパイク)のインデックスを抽出
            # rows=neuron_idx, cols=time_step
            neuron_indices, time_steps = np.where(spike_matrix)
            
            # 3. 時間ステップを物理時間に変換
            spike_times = time_steps * self.poisson_dt
            
            # 4. (spike_time, neuron_idx) の形に結合
            # patternのshapeは (総スパイク数, 2) になります
            pattern = np.column_stack((spike_times, neuron_indices))
            
            # (オプション) 時間順にソートする場合
            pattern = pattern[pattern[:, 0].argsort()]
            
            templates.append(pattern)
        
        return templates


    def _generate_BoostedTemplates(self, num_steps, p_base, num_boost=20, boost_factor=5.0):
        """
        ランダムに選んだ一部のニューロンの発火率を上げ、最後に正規化するシンプルな方式。
        
        Parameters:
        -----------
        num_boost : int
            1クラスあたり何個のニューロンを「贔屓」するか
        boost_factor : float
            贔屓するニューロンの発火率を何倍にするか
        """
        templates = []
        
        for c in range(self.num_classes):
            # 1. すべてのニューロンの重みを1で初期化
            spatial_weights = np.ones(self.num_channels)
            
            # 2. ランダムに指定した数のニューロンをサンプリングしてブースト
            # クラスごとに異なるPRNGシードを使うことで、固定のテンプレートになります
            boost_indices = self.prng.choice(self.num_channels, size=num_boost, replace=False)
            spatial_weights[boost_indices] *= boost_factor
            
            # 3. ★重要：平均発火率が p_base になるように正規化
            # これにより、どのクラスも「総スパイク数」が統計的に等しくなります
            p_profile = spatial_weights * p_base
            
            # 確率が1を超えないようにクリップ
            p_profile = np.clip(p_profile, 0, 0.99)
    
            # 4. スパイク行列の生成
            p_matrix = np.tile(p_profile[:, np.newaxis], (1, num_steps))
            spike_matrix = self.prng.random((self.num_channels, num_steps)) < p_matrix
    
            # 5. 座標形式に変換してソート
            neuron_indices, time_steps = np.where(spike_matrix)
            spike_times = time_steps * self.poisson_dt
            pattern = np.column_stack((spike_times, neuron_indices))
            pattern = pattern[pattern[:, 0].argsort()]
            
            templates.append(pattern)
            
        return templates

    def generate_target_within_batch(self, data_idx, label_idx):
        """
        教師データの生成 (One-hot encoding over time)
        """
        # 現在のデータ長を取得
        current_len = self.len_data[data_idx]
        target_within_batch = np.zeros((self.num_classes, current_len))
        # 正解ラベルの次元を1にする
        target_within_batch[label_idx, :] = 1.0
        return target_within_batch

   # --- 以下、元のクラスからほぼ変更なし（NEURONとのインターフェース） ---

    def create_synapses(self, cell, ip3=None, cyt=None):
        # 外部関数に依存しているようですが、呼び出し構造は維持
        if self.syn_mechanisms=='ionotropic':
            #create_synapses_toydata(self, cell) # 必要に応じてimport
            create_synapses_hetero(self, cell, self.condition)
            #pass 
        elif self.syn_mechanisms=='ionotropic_and_metabotropic':
            create_synapses_ionotoropic_and_metabotropic(self, cell, ip3, cyt)
            #pass

    def connect_synapses(self):
        connect_synapses_toydata(self)
        #pass

    def generate_data(self, data_idx, mode):
        if mode=="training":
            self.len_data.append(int(self.pattern_duration_ms / self.bin_width))
            label_idx = data_idx
            self.trainingdata_target = np.concatenate([self.trainingdata_target, self.generate_target_within_batch(data_idx,  label_idx)], axis=1)
            return self.class_templates[label_idx]
        elif mode=="test":
            self.len_data.append(int(self.pattern_duration_ms / self.bin_width))
            label_idx = self.test_labels[data_idx]
            self.testdata_target = np.concatenate([self.testdata_target, self.generate_target_within_batch(self.train_dataset_size+data_idx,  label_idx)], axis=1)

            # --- ジッターを加える処理 ---
            # 1. テンプレートをコピー（元のテンプレートを壊さないため）
            spiketrain = self.class_templates[label_idx].copy().astype(np.float64)
            
            # 2. ジッター（ガウスノイズ）を生成
            # spiketrain[:, 0] が spike_times です
            noise = self.prng.normal(loc=0, scale=self.jitter_std, size=spiketrain[:, 0].shape)
            
            # 3. 時刻の列にだけノイズを加える
            spiketrain[:, 0] += noise
            
            # 4. 時刻がマイナスにならないようにクリップし、時間順に再ソート
            spiketrain[:, 0] = np.clip(spiketrain[:, 0], 0, self.pattern_duration_ms)
            spiketrain = spiketrain[spiketrain[:, 0].argsort()]
            
            return spiketrain
        
    def resister_inputevent_toNetCon(self, data_idx, mode):
        """
        NEURONのNetConにイベントを登録するメイン関数
        """
        spike_trains = self.generate_data(data_idx, mode)

        # 全体のシミュレーション時間（オフセット）を計算
        # len_dataにはこれまでのデータの長さ（ステップ数ではなく配列サイズ=時間数と仮定されている場合もあるが、
        # 元コードでは np.size(time_coch) を入れているので、これは「サンプル数(点数)」である。
        # spike_time は bin_width * sum(len_data) でオフセットする必要がある。
        
        # 注意: 元コードの sum(self.len_data[:-1]) は直前のデータまでの長さを取得している
        offset_time = 0
        if len(self.len_data) > 1:
            # 今回追加した分(generate_dataでappend済)を除いて合計し、時間に変換
            # len_dataはappend済みなので、今回分を除くには [:-1]
            offset_time = sum(self.len_data[:-1]) * self.bin_width

        if self.syn_mechanisms == 'ionotropic':
            for spike_time, synapse_idx in spike_trains:
                abs_time = offset_time + spike_time
                self.exc_nc_list[int(synapse_idx)].event(abs_time)
            return spike_trains
