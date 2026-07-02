from one.api import ONE
from iblatlas.atlas import AllenAtlas as ba
from brainbox.io.one import SpikeSortingLoader
import json
import numpy as np

def one_setup(cache_dir: str):
    ONE.setup(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
    )

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=cache_dir, 
        )
    return one

def build_eid_from_results(json_path: str):
    eids=[]
    with open(json_path, 'r') as f:
        results = json.load(f)
        print(f"Loaded {len(results)} results from {json_path}")
        for lab_name in results.keys():
            for subject in results[lab_name].keys():
                eids.extend(results[lab_name][subject]["VIS_eids"])
    return eids

def load_trials(one, eid:str):
    trials = one.load_object(eid, 'trials', collection='alf')

    stim_on = np.asarray(trials['stimOn_tims'])
    stim_off = np.asarray(trials['stimOff_times'])

    valid = (~np.isnan(stim_on) & (~np.isnan(stim_off)))
    num_trials = len(stim_on)
    for k in list(trials.keys()):
        v = np.asarray(trials[k])
        if v.ndim >= 1 and v.shape[0] == num_trials:
            trials[k] = v[valid]
        else:
            trials[k] = v
    trials['stimOn_times'] = stim_on[valid]
    trials['stimOff_times'] = stim_off[valid]
    return trials

def pick_best_insertion(one, atlas, eid:str, target_prefix="VISp"):
    insertion = one.alyx.rest('insertions', 'list', session=eid)
    best_pid, best_n = None, -1
    errors = []
    for ins in insertion:
        pid = ins['id']
        try:
            sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
            acr = clusters.get('acronym', None)
            if acr is None:
                continue
            acr = np.asarray(acr)
            n = int(np.sum([a.startswith(target_prefix) for a in acr]))
            if n > best_n:
                best_n = n
                best_pid = pid
        except Exception as e:
            errors.append((pid, repr(e)))
            continue
    if best_pid is None:
        msg = f"No valid insertion found for eid={eid}."
        if errors:
            msg += f"Example loader errors: {errors[:1]}"
        raise RuntimeError(msg)
    return best_pid

def load_spikes_and_clusters(one, atlas, pid:str):
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    return spikes, clusters

def get_region_cluster_ids(clusters, target_prefix='VISp'):
    acr = np.asarray(clusters['acronym'])
    region_mask = np.array([a.startswith(target_prefix) for a in acr])
    region_cluster_ids = np.asarray(clusters['cluster_id'])[region_mask]
    return region_cluster_ids

