from neuron import h as nrn
from neuron.units import ms, mV
import numpy as np
import math
nrn.load_file("stdrun.hoc")

def get_dend_and_soma(cell):
    all_dend_soma = []
    for sec in cell.somatic:
        all_dend_soma.append(sec)
    for sec in cell.basal:
        all_dend_soma.append(sec)
    for sec in cell.apical:
        all_dend_soma.append(sec)
    return all_dend_soma

def check_synapse_stats(self):
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


class neuronalreservoir():
    def __init__(self, cell, prng, params):
        self.cell = cell
        nrn.celsius = 36

        self.bin_width   = params['bin_width']
        self.num_states = params['num_states']
        self.record_target = params['record_target']

        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.condition            = params['condition']

        self.reg = params['reg']
        self.cell = cell

        self.v_rec_list = []

        self.prng = prng
        self.W = self.prng.random(self.num_states)# readout weight
        
        self._build_network()
        self._create_records()

    def _build_network(self):
        self._create_synapses()
        self._connect_synapses()

    def _create_synapses(self):
        test_distance_accuracy(self, self.cell)
        # 距離計算の基準点（細胞体）を設定
        nrn.distance(0, 0.5, sec=self.cell.soma[0]) 
        all_segs = []
        areas = []
        distances = []

        # 1. 全セグメントの情報を収集
        for sec in get_dend_and_soma(self.cell):
            for seg in sec:
                all_segs.append(seg)
                areas.append(seg.area())
                distances.append(nrn.distance(seg))

        areas = np.array(areas)
        weights = np.zeros(areas.shape)
        distances = np.array(distances)

        # 2. 条件に応じた重み付け
        if self.condition == "distal-dense":  # 遠位に密集
            mu = 600.0  # 遠く（細胞の最大長に合わせて調整）
            sigma = 100.0
            weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
            
        elif self.condition == "proximal-dense":  # 近位に密集
            mu = 0.0   # 細胞体に近い
            sigma = 100.0
            weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
        elif self.condition == "proximal-sparse":  # 近位に密集
            mu = 000.0   # 細胞体に近い
            sigma = 300.0
            weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
        elif self.condition == "distal-sparse":  # 遠位に密集
            mu = 600.0  # 遠く（細胞の最大長に合わせて調整）
            sigma = 300.0
            weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
            
        elif self.condition == "random":  # 面積に比例した一様分布
            weights = areas

        # 3. 重みの正規化（合計を1にする）
        prob = weights / np.sum(weights)

        # 4. シナプスの配置（興奮性を例に）
        # 重みに基づいてセグメントをランダムに選択
        chosen_indices = self.prng.choice(len(all_segs), size=self.exc_num_syn, p=prob)

        self.exc_syn_list = []
        for idx in chosen_indices:
            seg = all_segs[idx]
            syn = nrn.Exp2Syn(seg)
            # --- パラメータ設定 ---
            syn.tau1 = self.exc_syn_tau1
            syn.tau2 = self.exc_syn_tau2
            syn.e = -10.0
            self.exc_syn_list.append(syn)

        check_synapse_stats(self)

    def _connect_synapses(self):
        self.exc_nc_list = []
        for syn in self.exc_syn_list:
            nc_tosyn = nrn.NetCon(None, syn)
            nc_tosyn.weight[0] = self.exc_syn_weight
            self.exc_nc_list.append(nc_tosyn)

    def resister_spike_events(self, spike_trains):
        """
        spike_trains: np.ndarray (shape: [N, 2]) 
                      column 0: spike_time, column 1: neuron_idx
        """
        # 1. 時間順にソートされていることを保証 (重要)
        # 既にソート済みであればこのステップはスキップ可能だが、安全のため。
        sorted_spikes = spike_trains[spike_trains[:, 0].argsort()]

        # 2. イベントの登録
        for spike_time, neuron_idx in sorted_spikes:
            # neuron_idx を int にキャスト（numpyのfloat型だとエラーが出るシミュレータがあるため）
            idx = int(neuron_idx)
            
            # 指定されたニューロン(NetCon等)にイベントを投入
            self.exc_nc_list[idx].event(spike_time)

    def _create_records(self):
        self.t_rec = nrn.Vector().record(nrn._ref_t)

        if self.record_target == 'potential':
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
                        self.v_rec_list.append(v)

        elif self.record_target == 'calcium_acum':
            total_length = 0
            cumulative_length_dict = []
            for sec in self.cell.all:
                cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                cumulative_length_dict.append(cumulative_length)
                total_length += sec.L

            while len(self.v_rec_list) < self.num_states:
                rec_loc = total_length * self.prng.random()

                for index, sec in enumerate(self.cell.all):
                    if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                        rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                        if hasattr(sec(rec_prop), '_ref_cai'):
                            v = nrn.Vector().record(sec(rec_prop)._ref_cai)
                            self.v_rec_list.append(v)
                        else:
                            break

    def generate_dynamics(self, total_duration):
        nrn.continuerun( total_duration * ms)

    #def bin_average(self, v_rec, t_rec):
    #    v_rec = np.array(v_rec)
    #    t_rec = np.array(t_rec)
    #    bin_index = 0
    #    temp_sum = np.zeros(self.datagenerator.len_transientdata + self.datagenerator.len_trainingdata+self.datagenerator.len_testdata)

    #    for t_index, t in enumerate(t_rec):
    #        bin_index = math.floor(t/self.bin_width)
    #        if (self.datagenerator.len_transientdata + self.datagenerator.len_trainingdata+self.datagenerator.len_testdata) <= bin_index:
    #            break

    #        if (t_index+1) == len(t_rec):
    #            temp_sum[bin_index] += v_rec[t_index] * (len(self.datagenerator.get_inputdata()) * self.bin_width - t)
    #        else:
    #            temp_sum[bin_index] += v_rec[t_index] * (t_rec[t_index+1]-t)

    #    output = temp_sum / self.bin_width

    #    return np.transpose(output)

    def get_binned_states(self):
        t_rec = np.array(self.t_rec.to_python())
        v_rec = np.column_stack([np.array(v.to_python()) for v in self.v_rec_list])
        t_end = t_rec[-1]

        # 1. 時間刻みの計算 (dt) を先に計算
        # 各点から「次の点」までの時間を重みとする
        dt = np.diff(t_rec, append=t_rec[-1] + (t_rec[-1] - t_rec[-2]))
        
        # 2. ビン・インデックスの計算（丸め誤差対策）
        # t_startからの相対時間で計算
        t_relative = t_rec - t_rec[0]
        bin_indices = (t_relative / self.bin_width).astype(int)
        num_bins = int(t_end / self.bin_width)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # 3. 重み付き状態量の計算
        weighted_v = v_rec * dt[:, np.newaxis]
        
        # 4. 集約 (ループを回避)
        res = np.zeros((num_bins, self.num_states))
        # numpy.add.at は res[bin_indices, :] += weighted_v を高速に行う
        np.add.at(res, bin_indices, weighted_v)
        
        return res / self.bin_width

    #def get_binned_states(self):
    #    """
    #    start_binからend_binまでの状態量を計算する共通メソッド
    #    """
    #    # 1. 必要な時間範囲を特定 (ベクトル演算で高速化)
    #    #t_start = start_bin * self.bin_width
    #    #t_end = (end_bin + 1) * self.bin_width
    #    t_rec = np.array(self.t_rec.to_python())
    #    t_start   = t_rec[0]
    #    t_end     = t_rec[-1]
    #    start_bin = int(t_start / self.bin_width)
    #    print(f'start_bin: {start_bin}')
    #    end_bin   = int(t_end   / self.bin_width)
    #    print(f'end_bin: {end_bin}')

    #    # 範囲内のインデックスを抽出
    #    v_rec = np.column_stack([np.array(v_rec.to_python()) for v_rec in self.v_rec_list])
    #    mask = (t_rec >= t_start) & (t_rec < t_end)
    #    v_slice = v_rec[mask, :]
    #    t_slice = t_rec[mask]
    #
    #    # 2. 各Binへの割り当て（np.digitizeを使用）
    #    bin_indices = ((t_slice - t_start) // self.bin_width).astype(int)
    #    
    #    # 3. 時間重み付き和の計算 (t_{i+1} - t_i)
    #    dt = np.diff(t_slice, append=t_slice[-1] + (t_slice[-1]-t_slice[-2]))
    #    weighted_v = v_slice * dt[:, np.newaxis]
    #    
    #    # 4. Binごとに集約 (np.add.at または bincount)
    #    num_bins = int(end_bin - start_bin + 1)
    #    #res = np.zeros((self.num_states, num_bins))
    #    res = np.zeros((num_bins, self.num_states))
    #    for i in range(self.num_states):
    #        res[:, i] = np.bincount(bin_indices, weights=weighted_v[:, i], minlength=num_bins)[:num_bins]
    #        
    #    return res / self.bin_width

    def readout(self, state_vars):
        return state_vars @ self.W

    def optimize(self, state_vars, target):
        self.W = np.linalg.inv(np.transpose(state_vars) @ state_vars + self.reg*np.eye(self.num_states)) @ np.transpose(state_vars) @ target
