from neuron import h as nrn
from neuron.units import ms, mV
import numpy as np
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

nrn.load_file("stdrun.hoc")

def get_soma_and_basal(cell):
    soma_and_basal = []
    for sec in cell.somatic:
        soma_and_basal.append(sec)
    for sec in cell.basal:
        soma_and_basal.append(sec)
    return soma_and_basal

def get_soma_and_apical(cell):
    soma_and_apical = []
    for sec in cell.somatic:
        soma_and_apical.append(sec)
    for sec in cell.apical:
        soma_and_apical.append(sec)
    return soma_and_apical

def get_soma_and_all_dend(cell):
    soma_and_all_dend = []
    for sec in cell.somatic:
        soma_and_all_dend.append(sec)
    for sec in cell.apical:
        soma_and_all_dend.append(sec)
    for sec in cell.basal:
        soma_and_all_dend.append(sec)
    return soma_and_all_dend

def check_synapse_stats(self):
    from neuron import h
    import numpy as np
    from collections import Counter

    # 1. directly obtain segment objects for each synapse
    exc_segs = [syn.get_segment() for syn in self.exc_syn_list]
    
    # 2. count overlaps per segment
    seg_counts = Counter(exc_segs)
    occ_segs_count = len(seg_counts) # number of unique segments with synapses
    max_overlap = max(seg_counts.values()) if seg_counts else 0

    # 3. obtain distance data
    exc_dists = [h.distance(seg) for seg in exc_segs]

    logger.info(f"\n--- detailed placement check ({self.syn_loc_condition}) ---")
    logger.info(f"total synapses: {len(exc_dists)}")
    logger.info(f"occupied segments: {occ_segs_count}")
    logger.info(f"max overlap in one segment: {max_overlap} synapses")
    logger.info(f"average density: {len(exc_dists)/occ_segs_count:.2f} syns/segment (among occupied)")

    # 4. "number of synapses" vs "number of segments" per distance
    logger.info("\n[distance: synapses vs occupied segments]")
    bins = np.arange(0, 1100, 100)
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        
        # all synapses in this range
        syns_in_range = [d for d in exc_dists if lower <= d < upper]
        # unique segments with synapses in this range
        segs_in_range = {s for s in exc_segs if lower <= h.distance(s) < upper}
        
        if len(syns_in_range) > 0:
            bar = "#" * int(len(syns_in_range) / len(self.exc_syn_list) * 40)
            logger.info(f"{lower:4.0f}-{upper:4.0f} um: {bar}")
            logger.info(f"    -> synapses: {len(syns_in_range)}, segments: {len(segs_in_range)}")


def report_synapse_stats(self):
    """
    基底・尖端樹状突起別にシナプスの距離分布とセグメント占有率を出力する。
    """
    from collections import Counter
    import numpy as np

    # 1. セグメントの取得と部位判定
    # 判定基準: セクション名に 'apic' が含まれれば apical、それ以外（dend, basal等）は basal
    exc_segs = [syn.get_segment() for syn in self.exc_syn_list]
    
    # 部位別のデータ格納用
    stats = {
        "apical": {"dists": [], "segs": set()},
        "basal":  {"dists": [], "segs": set()}
    }

    for seg in exc_segs:
        sec_name = seg.sec.name().lower()
        dist = nrn.distance(seg)
        
        target = "apical" if "apic" in sec_name else "basal"
        stats[target]["dists"].append(dist)
        stats[target]["segs"].add(seg)

    # 2. ログ出力
    logger.info(f"\n--- detailed placement check ({self.syn_loc_condition}) ---")
    logger.info(f"total synapses: {len(exc_segs)}")

    bins = np.arange(0, 1100, 100)
    
    for domain in ["basal", "apical"]:
        dists = stats[domain]["dists"]
        segs = stats[domain]["segs"]
        
        if not dists:
            logger.info(f"\n[{domain.upper()}] No synapses placed.")
            continue

        # ドメインごとの統計
        seg_counts = Counter([syn.get_segment() for syn in self.exc_syn_list 
                              if (domain == "apical" and "apic" in syn.get_segment().sec.name().lower()) 
                              or (domain == "basal" and "apic" not in syn.get_segment().sec.name().lower())])
        
        max_overlap = max(seg_counts.values()) if seg_counts else 0
        
        logger.info(f"\n[{domain.upper()} dendrites]")
        logger.info(f"  synapses: {len(dists)}, unique segments: {len(segs)}")
        logger.info(f"  max overlap: {max_overlap}, avg density: {len(dists)/len(segs):.2f} syns/seg")

        # 距離ビンごとの分布表示
        for i in range(len(bins)-1):
            lower, upper = bins[i], bins[i+1]
            syns_in_range = [d for d in dists if lower <= d < upper]
            segs_in_range = {s for s in segs if lower <= nrn.distance(s) < upper}

            if syns_in_range:
                # 全シナプス数に対する割合でバーを表示
                bar_len = int(len(syns_in_range) / len(self.exc_syn_list) * 40)
                bar = "#" * bar_len
                logger.info(f"  {lower:4.0f}-{upper:4.0f} um: {bar}")
                logger.info(f"    -> syns: {len(syns_in_range)}, segments: {len(segs_in_range)}")


def test_distance_accuracy(self, cell):
    from neuron import h
    # 1. reset origin (just in case)
    nrn.distance(0, 0.5, sec=cell.soma[0])
    
    logger.info("--- neuron distance validation ---")
    
    # test a: distance of the soma itself (should be near 0)
    d_soma = h.distance(cell.soma[0](0.5))
    logger.info(f"distance at soma(0.5): {d_soma:.4f} um (expected: 0.0)")

    # test b: edge of the soma (should be l/2)
    d_soma_end = h.distance(cell.soma[0](1.0))
    logger.info(f"distance at soma(1.0): {d_soma_end:.4f} um (expected: ~{cell.soma[0].L/2:.1f})")

class neuronalreservoir():
    def __init__(self, cell, prng, params):
        self.cell = cell
        nrn.celsius = 36

        self.num_states    = params['num_states']
        self.record_target = params['record_target']

        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.syn_loc_condition    = params['syn_loc_condition']
        self.syn_loc_mean         = params['syn_loc_mean']
        self.syn_loc_std          = params['syn_loc_std']

        self.reg = params['reg']
        self.cell = cell

        self.v_rec_list = []

        self.prng = prng
        self.W = self.prng.random(self.num_states) # readout weight
        
        self._build_network()
        self._create_records()

    def _build_network(self):
        self._create_synapses()
        self._connect_synapses()

    def _create_synapses(self):
        test_distance_accuracy(self, self.cell)
        # set the reference point for distance calculation (soma)
        nrn.distance(0, 0.5, sec=self.cell.soma[0]) 
        all_segs = []
        areas = []
        distances = []

        # 1. collect information for all segments
        if self.syn_loc_condition == "random" or self.syn_loc_condition == "gaussian":
            for sec in get_soma_and_all_dend(self.cell):
                for seg in sec:
                    all_segs.append(seg)
                    areas.append(seg.area())
                    distances.append(nrn.distance(seg))
        elif self.syn_loc_condition == "gaussian-apical":
            for sec in get_soma_and_apical(self.cell):
                for seg in sec:
                    all_segs.append(seg)
                    areas.append(seg.area())
                    distances.append(nrn.distance(seg))
        elif self.syn_loc_condition == "gaussian-basal":
            for sec in get_soma_and_basal(self.cell):
                for seg in sec:
                    all_segs.append(seg)
                    areas.append(seg.area())
                    distances.append(nrn.distance(seg))

        areas = np.array(areas)
        weights = np.zeros(areas.shape)
        distances = np.array(distances)

        ## 2. Weighting based on conditions
        if self.syn_loc_condition == "gaussian-apical" or self.syn_loc_condition == "gaussian-basal" or self.syn_loc_condition == "gaussian":
            mu    = self.syn_loc_mean
            sigma = self.syn_loc_std
            weights = areas * np.exp(-((distances - mu)**2) / (2 * sigma**2))
        elif self.syn_loc_condition == "random":
            weights = areas

        # 3. Normalize weights (sum to 1)
        prob = weights / np.sum(weights)

        # 4. Placement of synapses (using excitatory as an example)
        # Randomly select segments based on weights
        chosen_indices = self.prng.choice(len(all_segs), size=self.exc_num_syn, p=prob)

        self.exc_syn_list = []
        for idx in chosen_indices:
            seg = all_segs[idx]
            syn = nrn.Exp2Syn(seg)
            # --- Parameter settings ---
            syn.tau1 = self.exc_syn_tau1
            syn.tau2 = self.exc_syn_tau2
            syn.e = -10.0

            self.exc_syn_list.append(syn)

        logger.info(f"Synapses created for condition: {self.syn_loc_condition}")
        #check_synapse_stats(self)
        report_synapse_stats(self)

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
        # 1. Guarantee chronological sorting (Important)
        # Skip if already sorted, but kept for safety.
        sorted_spikes = spike_trains[spike_trains[:, 0].argsort()]

        # 2. Register events
        for spike_time, neuron_idx in sorted_spikes:
            # Cast neuron_idx to int (some simulators error with numpy float)
            idx = int(neuron_idx)
            
            # Inject event to specific neuron (NetCon, etc.)
            self.exc_nc_list[idx].event(spike_time)
        
        logger.debug(f"Registered {len(sorted_spikes)} spike events.")

    def _create_records(self):
        self.t_rec = nrn.Vector().record(nrn._ref_t)
        self.record_segs = []

        if self.record_target == 'potential':
            total_length = 0
            cumulative_length_dict = []
            for sec in get_soma_and_all_dend(self.cell):
                cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                cumulative_length_dict.append(cumulative_length)
                total_length += sec.L

            for rec in range(self.num_states):
                # Randomly pick a location along the total length
                rec_loc = total_length * self.prng.random()

                for index, sec in enumerate(get_soma_and_all_dend(self.cell)):
                    if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                        # Calculate proportional position within the section
                        rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                        v = nrn.Vector().record(sec(rec_prop)._ref_v)
                        self.record_segs.append(sec(rec_prop))
                        self.v_rec_list.append(v)

        elif self.record_target == 'calcium_acum':
            total_length = 0
            cumulative_length_dict = []
            for sec in get_soma_and_all_dend(self.cell):
                cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                cumulative_length_dict.append(cumulative_length)
                total_length += sec.L

            while len(self.v_rec_list) < self.num_states:
                rec_loc = total_length * self.prng.random()

                for index, sec in enumerate(get_soma_and_all_dend(self.cell)):
                    if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                        rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                        # Check if calcium concentration pointer exists
                        if hasattr(sec(rec_prop), '_ref_cai'):
                            v = nrn.Vector().record(sec(rec_prop)._ref_cai)
                            self.record_segs.append(sec(rec_prop))
                            self.v_rec_list.append(v)
                        else:
                            break
        
        logger.info(f"Recording setup complete for target: {self.record_target}")

    def generate_dynamics(self, total_duration):
        nrn.continuerun( total_duration * ms)

    def get_binned_states(self, interval_start, num_bins, time_integration):
        if time_integration:
            t_rec = np.array(self.t_rec.to_python())
            v_rec = np.column_stack([np.array(v.to_python()) for v in self.v_rec_list])
            t_start = interval_start
            t_end   = t_rec[-1]

            # 1. Calculate time steps (dt) first
            # Use time until "next point" as the weight for each point
            dt = np.diff(t_rec, append=t_rec[-1] + (t_rec[-1] - t_rec[-2]))
            
            # 2. Calculate bin indices (accounting for rounding errors)
            # Calculate relative time from t_start
            t_relative = t_rec - t_start
            bin_indices = (t_relative / self.bin_width).astype(int)
            #num_bins = int((t_end - t_start) / self.bin_width)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            
            # 3. Calculate weighted state variables
            weighted_v = v_rec * dt[:, np.newaxis]
            
            # 4. Aggregation (avoiding loops)
            res = np.zeros((num_bins, self.num_states))
            # numpy.add.at performs res[bin_indices, :] += weighted_v efficiently
            np.add.at(res, bin_indices, weighted_v)
            
            return res / self.bin_width

        elif not time_integration:
            t_rec = np.array(self.t_rec.to_python())
            v_rec = np.column_stack([np.array(v.to_python()) for v in self.v_rec_list])
            return v_rec

    def readout(self, state_vars):
        return state_vars @ self.W

    def optimize(self, state_vars, target):
        logger.info("Optimizing readout weights...")
        self.W = np.linalg.inv(np.transpose(state_vars) @ state_vars + self.reg*np.eye(self.num_states)) @ np.transpose(state_vars) @ target
