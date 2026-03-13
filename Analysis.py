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
