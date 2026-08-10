# run_condition_geometry.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import src.ibl_io as ibl_io
import src.firing_rates as fr
import src.alignment_metrics as am
import src.trial_selection as ts
import src.prestim as prestim
# from itertools import combinations
import numpy as np
import pandas as pd
import pickle
import argparse
from iblatlas.atlas import AllenAtlas 

def signal_manifold_overlap(X, condition_masks, pos_mask, neg_mask, n_components=3):
    components, explained_variance_ratio = am.compute_condition_mean_pca(X, condition_masks, n_components=n_components)
    u_sig = am.compute_signal_axis(X, pos_mask, neg_mask, min_trials=5, eps=1e-12)
    overlap_by_k = {}
    U = components.T
    for k in range(1, U.shape[1]+1):
        U_k = U[:, :k]
        overlap = float(np.sum((U_k.T @ u_sig) ** 2))
        overlap_by_k[k] = overlap
    # only for plotting
    R, condition_means, _ = am.noise_residuals_by_condition(X, condition_masks)
    condition_order = sorted(condition_means.keys())
    condition_mean_matrix = np.asarray([condition_means[cond] for cond in condition_order], float, )
    condition_mean_centered = condition_mean_matrix - np.mean(condition_mean_matrix, axis=0, keepdims=True)
    condition_mean_scores = condition_mean_centered @ U

    return {'components': components, 
            'explained_variance_ratio': explained_variance_ratio,
            'u_sig': u_sig,
            'overlap_by_k': overlap_by_k,
            'condition_order': condition_order,
            'condition_mean_scores': condition_mean_scores,}

# def compute_signal_axis_pair_similarity(X, trials, u_sig, min_trials=5, eps=1e-12):
#     axes = am.compute_contrast_pair_axes(X, trials, eps=eps)
#     out_pairwise, out_global = am.summarize_contrast_pair_axes(axes, u_sig, eps=eps)
#     return out_pairwise, out_global

def noise_subspace_similarity(X, condition_masks, k=3, eps=1e-12):
    noise_subspace_similarities, mean_similarity, min_similarity, trial_counts = am.condition_noise_subspaces(X, condition_masks, k=k, eps=eps)
    random_similarity = am.random_subspace_similarity(X, condition_masks, k=k, eps=eps)
    return {
        'noise_subspace_similarities': noise_subspace_similarities,
        'mean_similarity': mean_similarity,
        'min_similarity': min_similarity,
        'random_similarity': random_similarity,
        'trial_counts': trial_counts,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run condition-geometry analysis for one brain region."
    )
    parser.add_argument(
        "--target-prefix",
        required=True,
        help="Allen region acronym or prefix, for example VISp or VISl.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    target_prefix = args.target_prefix.strip()
    if not target_prefix:
        raise ValueError("--target-prefix cannot be empty")
    if Path(target_prefix).name != target_prefix:
        raise ValueError(
            f"Invalid --target-prefix path component: {target_prefix!r}"
        )
    one = ibl_io.one_setup(
        cache_dir="/scratch/midway3/xiaorantu/ONE"
    )
    data_dir = (
        REPO_ROOT
        / "results"
        / "region_scan"
        / target_prefix
    )
    data_path = (
        data_dir
        / f"{target_prefix}_subjects_by_lab.json"
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Region-scan input JSON not found: {data_path}"
        )
    eids = ibl_io.build_eids_from_results(data_path)
    eids_to_run = eids[:9]
    atlas = AllenAtlas()
    rows = []
    details = {}

    output_dir = (
        REPO_ROOT
        / "results"
        / "condition_geometry"
        / target_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target prefix: {target_prefix}")
    print(f"Input JSON: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of selected EIDs: {len(eids_to_run)}")

    for eid in eids_to_run:
        print(f'Processing eid: {eid}')
        trials = ibl_io.load_trials(one=one, eid=eid)
        best_pid = ibl_io.pick_best_insertion(one=one, atlas=atlas, eid=eid, target_prefix=target_prefix)
        spikes, clusters = ibl_io.load_spikes_and_clusters(one=one, pid=best_pid, atlas=atlas)
        region_cluster_ids = ibl_io.get_region_cluster_ids(clusters, target_prefix=target_prefix)
        stim_on = np.asarray(trials["stimOn_times"], dtype=float)
        start = stim_on + 0.04 # VISp maximal population trajectory distance and modulation latency
        end = start + 0.1 # try out 100 ms time window
        X, unit_ids = fr.compute_static_firing_rates(spikes, region_cluster_ids, start=start, end=end)
        X_filtered, unit_mask = fr.filter_active_units(X, eps=1e-10, min_units=5)

        signed_contrast = ts.get_signed_contrast(trials)
        condition_masks = ts.make_condition_masks(signed_contrast, min_trials=5)
        high_mask = ts.get_high_masks(signed_contrast, min_trials=5, threshold=0.5)
        pos_mask, neg_mask = ts.get_pos_neg_masks(signed_contrast, high_mask=high_mask, min_trials=5)
        pos_1_mask = high_mask & pos_mask
        neg_1_mask = high_mask & neg_mask
        R, condition_means, residual_mask = am.noise_residuals_by_condition(X_filtered, condition_masks)
        prestim_R, _ = prestim.get_prestim_residuals(spikes, region_cluster_ids, stim_on, condition_masks, unit_mask, lapse=0.1, eps=1e-12)
        prestim_noise_covariance_high = am.compute_noise_covariance(prestim_R, high_mask)
        prestim_noise_top_variance, prestim_a = am.compute_noise_top1_variance(prestim_noise_covariance_high, n_components=1)   
        noise_covariance_high = am.compute_noise_covariance(R, high_mask)
        noise_top_variance, a = am.compute_noise_top1_variance(noise_covariance_high, n_components=1)
        effective_units = 1.0 / np.sum(a ** 4)
        null_eigenspectrum_results = am.test_null_eigenspectrum(X_filtered, high_mask, pos_mask, neg_mask, n_iter=500, eps=1e-12, seed=0)
        true_ratio = null_eigenspectrum_results['pc12_true']
        null_mean_ratio = null_eigenspectrum_results['null_mean']
        p_val = null_eigenspectrum_results['p_value']
        eigenspectrum = null_eigenspectrum_results['eigenspectrum']
        max_loading_sq = np.max(a ** 2)
        pos_1_covariance = am.compute_noise_covariance(R, pos_1_mask)
        neg_1_covariance = am.compute_noise_covariance(R, neg_1_mask)
        _, a_pos_1 = am.compute_noise_top1_variance(pos_1_covariance, n_components=1)
        _, a_neg_1 = am.compute_noise_top1_variance(neg_1_covariance, n_components=1)
        a_similarity = float(np.sum((a_pos_1.T @ a_neg_1) ** 2))
        null_a_results = am.null_a_similarity(a_pos_1, a_neg_1, n_iter=1000, seed=0)
        null_a_p_val = null_a_results['p_value']
        signal_manifold_results = signal_manifold_overlap(X_filtered, condition_masks, pos_mask, neg_mask, n_components=3)
        u_sig = signal_manifold_results['u_sig']
        # pairwise_results, global_results = compute_signal_axis_pair_similarity(X_filtered, trials, u_sig, min_trials=5, eps=1e-12)
        noise_subspace_results = noise_subspace_similarity(X_filtered, condition_masks, k=3, eps=1e-12)
        evr = np.asarray(signal_manifold_results['explained_variance_ratio'], float,)
        overlap = signal_manifold_results['overlap_by_k']
        prestim_a_cosine_similarity = float(np.sum((prestim_a.T @ a) ** 2))
        row = {
            "eid": eid,
            "pid": best_pid,
            "n_trials": int(X_filtered.shape[0]),
            "n_units": int(X_filtered.shape[1]),
            "n_conditions": int(len(condition_masks)),
            # high contrast u_sig for use
            "u_sig": np.asarray(u_sig, float),
            "stim_pc1_var": float(evr[0]) if len(evr) > 0 else np.nan,
            "stim_pc2_var": float(evr[1]) if len(evr) > 1 else np.nan,
            "stim_pc3_var": float(evr[2]) if len(evr) > 2 else np.nan,
            "stim_pc123_var": float(np.nansum(evr[:3])),

            "sig_overlap_stim_pc1": overlap.get(1, np.nan),
            "sig_overlap_stim_pc2": overlap.get(2, np.nan),
            "sig_overlap_stim_pc3": overlap.get(3, np.nan),

            "noise_mean_condition_similarity": noise_subspace_results["mean_similarity"],
            "noise_min_condition_similarity": noise_subspace_results["min_similarity"],
            # "pairwise_signal_axes": pairwise_results,
            # "global_signal_axis_summary": global_results,
            "noise_top1_variance": noise_top_variance,
            "effective_units": effective_units,
            "max_loading_sq": max_loading_sq,
            "noise_pc12_ratio": true_ratio,
            "pc12_null_pval": p_val,
            "eigenspectrum": eigenspectrum,
            "a_similarity": a_similarity,
            "null_a_p_val": null_a_p_val,
            "prestim_a_cosine_similarity": prestim_a_cosine_similarity,
        }
        random_similarity = noise_subspace_results["random_similarity"]
        if isinstance(random_similarity, dict):
            row["random_noise_similarity_mean"] = random_similarity.get("mean", np.nan)
            row["random_noise_similarity_p95"] = random_similarity.get("p95", np.nan)
        else:
            row["random_noise_similarity_mean"] = float(random_similarity)
        rows.append(row)
        details[eid] = {
            "signal_manifold": signal_manifold_results,
            "noise_subspace": noise_subspace_results,
            "unit_ids": unit_ids[unit_mask],
            "condition_trial_counts": {
                float(c): int(mask.sum())
                for c, mask in condition_masks.items()
            },
        }
        print(
            f"  PC1={row['stim_pc1_var']:.3f}, "
            f"sig-PC3={row['sig_overlap_stim_pc3']:.3f}, "
            f"noise-sim={row['noise_mean_condition_similarity']:.3f}"
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "condition_geometry_summary.csv"
    details_pkl = output_dir / "condition_geometry_details.pkl"
    summary_df.to_csv(summary_csv, index=False)

    with open(details_pkl, "wb") as f:
        pickle.dump(details, f)

    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved details to {details_pkl}")

if __name__ == "__main__":
    main()

