# run_condition_geometry.py
import src.ibl_io as ibl_io
import src.firing_rates as fr
import src.alignment_metrics as am
import src.trial_selection as ts
from itertools import combinations

def signal_manifold_overlap(X, condition_masks, n_components=3):
    components, explained_variance_ratio = am.compute_condition_mean_pca(X, condition_masks, n_components=n_components)
    return components, explained_variance_ratio

def compute_signal_axis_pair_similarity(X, trials, pos_mask, neg_mask, min_trial=5, eps=1e-12):
    u_sig = am.compute_signal_axis(X, pos_mask, neg_mask, min_trial=min_trial, eps=eps)
    axes = am.compute_contrast_pair_axes(X, trials, eps=eps)
    out_pairwise, out_global = am.summarize_contrast_pair(axes, u_sig, eps=eps)
    return out_pairwise, out_global

def noise_subspace_similarity(X, condition_masks, k=3, eps=1e-12):
    noise_subspace_similarities, mean_similarity, min_similarity, trial_counts = am.condition_noise_subspaces(X, condition_masks, k=k, eps=eps)
    random_similarity = am.random_noise_subspace_similarity(X, condition_masks, k=k, eps=eps)
    return {
        'noise_subspace_similarities': noise_subspace_similarities,
        'mean_similarity': mean_similarity,
        'min_similarity': min_similarity,
        'random_similarity': random_similarity,
        'trial_counts': trial_counts,
    }


