import numpy as np
import ibl_io
import firing_rates as fr
from iblatlas.atlas import AllenAtlas

def load_spike_window(one, atlas, eid):
    trials = ibl_io.load_trials(one, eid)
    stim_on = np.asarray(trials['stimOn_times'], float)
    stim_on = stim_on[np.isfinite(stim_on)]
    first_stim = np.min(stim_on)
    pid = ibl_io.pick_best_insertion(one=one, atlas=atlas, eid=eid, target_prefix="VISp")
    spikes, clusters = ibl_io.load_spikes_and_clusters(one=one, atlas=atlas, pid=pid)
    spike_times = np.asarray(spikes['times'], float)
    rec_start = np.finite_min(spike_times)
    rec_end = np.finite_max(spike_times[spike_times < first_stim])
    spike_times = spike_times[(spike_times >= rec_start) & (spike_times < rec_end)]
    spike_clusters = np.asarray(spikes['clusters'])
    spike_clusters = spike_clusters[(spike_times >= rec_start) & (spike_times <= rec_end)]
    trimmed_spikes = {'times': spike_times, 'clusters': spike_clusters}
    return rec_start, rec_end, trimmed_spikes

def make_bin_edges(start, end, bin_size):
    starts = np.arange(start, end - bin_size, bin_size)
    ends = starts + bin_size
    return starts, ends

def compute_null_firing_rates(spikes, starts, ends, region_cluster_ids, eps=1e-12):
    fr, unit_ids = fr.compute_static_firing_rates(spikes, starts, ends, region_cluster_ids, eps)
    return fr, unit_ids

def null_filter_active_units(X, eps=1e-10, min_units=None):
    X_filtered, unit_mask = fr.filter_active_units(X, eps=eps, min_units=min_units)
    return X_filtered, unit_mask

def compute_null_residual(X):
    X = np.asarray(X, float)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    return X_centered

def compute_null_covariance(X):
    C = np.cov(X.T, bias=False)
    C = 0.5 * (C + C.T)
    return C









