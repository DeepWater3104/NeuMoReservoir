import numpy as np

def get_spike_timings(time, Vm_trace, threshold):
    spike_timings = []
    for current_time_idx, current_time in enumerate(time[1:]):
        if Vm_trace[current_time_idx-1] < threshold and threshold < Vm_trace[current_time_idx]:
            spike_timings.append(current_time)

    return spike_timings

def get_firing_rate(spike_timings, duration):
    # spike_timings: list of spike timings
    # duration: Temporal duration recorded membrane potential in sec
    return len(spike_timings) / duration

def get_spike_indices(vm, t, v_threshold=-20):
    """
    vm: (N_time_steps, N_compartments)
    """
    dt = t[1] - t[0]
    mean_vm = np.mean(vm, axis=1) # calculate mean membrane potential for each time steps
    
    # bAP
    spike_cond = (mean_vm[:-1] < v_threshold) & (mean_vm[1:] >= v_threshold)
    bap_indices = np.where(spike_cond)[0]
    
    # dAP
    all_local_spikes = []
    for i in range(vm.shape[1]): # iteration about each compartments
        spks = np.where((vm[:-1, i] < v_threshold) & (vm[1:, i] >= v_threshold))[0]
        all_local_spikes.extend(spks)
    all_local_spikes = np.unique(all_local_spikes)

    dap_indices = []
    exclusion_window = int(10.0 / dt) 
    for d_idx in all_local_spikes:
        if not np.any(np.abs(bap_indices - d_idx) < exclusion_window):
            dap_indices.append(d_idx)
                
    return bap_indices, np.array(dap_indices)


# ==============================================
# Effective Rank defined in Roy & Vetterli, 2007
# ==============================================
def calculate_effective_rank(sub_data):
    """
    sub_data: (N_time_steps, N_compartments)
    """
    if np.all(sub_data == 0) or sub_data.size == 0:
        return 1.0
        
    try:
        # 特異値分解
        # 形状が (T, N) の場合、s は min(T, N) 個の特異値を返す
        _, s, _ = np.linalg.svd(sub_data, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    sigma_sum = np.sum(s)
    if sigma_sum == 0:
        return 1.0
        
    p = s / sigma_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)
