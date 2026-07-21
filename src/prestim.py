# prestim.py
import numpy as np
from . import alignment_metrics as am
from . import firing_rates as fr

def get_prestim_residuals(spikes, region_cluster_ids,stim_on, condition_masks, unit_mask, lapse=0.1, eps=1e-12):
    stim_on = np.asarray(stim_on, float)
    unit_mask = np.asarray(unit_mask, bool)
    starts = stim_on - lapse
    ends = stim_on
    firing_rate, unit_ids = fr.compute_static_firing_rates(spikes, region_cluster_ids, starts, ends, eps=eps)
    fr_filtered = firing_rate[:, unit_mask]
    R, _, residual_mask = am.noise_residuals_by_condition(fr_filtered, condition_masks)
    return R, residual_mask








