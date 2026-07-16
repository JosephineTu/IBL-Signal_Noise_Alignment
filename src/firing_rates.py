# firing_rates.py
import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins

def compute_static_firing_rates(spikes, region_cluster_ids, start, end, eps=1e-12):
    stim_intervals = np.c_[start, end]
    n_trials = stim_intervals.shape[0]
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], stim_intervals)
    cluster_ids = np.asarray(cluster_ids)
    if counts.shape[0] == cluster_ids.shape[0] and counts.shape[1] == n_trials:
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[keep, :].T
        unit_ids = cluster_ids[keep]
    elif counts.shape[1] == cluster_ids.shape[0] and counts.shape[0] == n_trials:
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[:, keep]
        unit_ids = cluster_ids[keep]
    else:
        raise RuntimeError(f"Counts shape {counts.shape} and cluster_ids shape {cluster_ids.shape} are incompatible.")
    
    dur = np.diff(stim_intervals, axis=1)
    fr = counts / (dur + eps)
    return fr, unit_ids

def make_time_windows(t_start, t_end, bin_size, step_size):
    starts = np.arange(t_start, t_end - bin_size, step_size)
    ends = starts + bin_size
    if len(starts) < 1:
        raise ValueError("No time windows can be created with the given parameters.")
    return np.c_[starts, ends]

def compute_time_resolved_firing_rates(spikes, stim_on, region_cluster_ids, windows):
    stim_on = np.asarray(stim_on)
    windows = np.asarray(windows, float)
    n_trials = stim_on.shape[0]
    n_bins = windows.shape[0]
    n_intervals = n_trials * n_bins

    starts = (stim_on[:, None] + windows[None, :, 0]).reshape(-1)
    ends = (stim_on[:, None] + windows[None, :, 1]).reshape(-1)
    intervals = np.c_[starts, ends]
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], intervals)
    cluster_ids = np.asarray(cluster_ids)
    if counts.shape[0] == cluster_ids.shape[0] and counts.shape[1] == n_intervals:
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[keep, :].T
        unit_ids = cluster_ids[keep]
    elif counts.shape[1] == cluster_ids.shape[0] and counts.shape[0] == n_intervals:
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[:, keep]
        unit_ids = cluster_ids[keep]
    else:
        raise RuntimeError(
            f"Unexpected counts shape {counts.shape}, "
            f"cluster_ids {cluster_ids.shape}, n_intervals={n_intervals}"
        )

    counts = counts.reshape(n_trials, n_bins, -1)
    dur = (windows[:,1] - windows[:,0])[None, :, None]
    fr_tb = counts / (dur + 1e-12)
    return fr_tb, unit_ids

def filter_active_units(X, eps=1e-10, min_units=None):
    X = np.asarray(X, dtype=float)
    if X.ndim not in (2, 3):
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
    n_trials = X.shape[0]

    if X.ndim == 2:
        n_units = X.shape[1]
        std = np.nanstd(X, axis=0)
        unit_mask = (std > eps)
        X_filtered = X[:, unit_mask]
    else:
        n_trials, n_bins, n_units = X.shape
        std = np.nanstd(X, axis=(0, 1))
        unit_mask = (std > eps)
        X_filtered = X[:, :, unit_mask]
    
    if min_units is not None and np.sum(unit_mask) < min_units:
        raise RuntimeError(f"Not enough active units: {np.sum(unit_mask)} < {min_units}")
    
    return X_filtered, unit_mask
    


