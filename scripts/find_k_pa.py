from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.atlas import AllenAtlas


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent
    if (SCRIPT_DIR.parent / "src").is_dir()
    else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

import src.alignment_metrics as am
import src.firing_rates as fr
import src.ibl_io as ibl_io
import src.trial_selection as ts


def find_K_pa(R, pos_mask, neg_mask, num_iter=500):
    rng = np.random.default_rng(seed=42)
    n_neurons = R.shape[1]
    noise_mask = pos_mask | neg_mask

    C_orig = am.compute_noise_covariance(R, noise_mask)
    _, _, observed_eigenvalues = am.get_eigenspectrum(C_orig)
    observed_eigenvalues = np.asarray(
        observed_eigenvalues,
        dtype=float,
    )

    null_eigenvalues = np.empty(
        (num_iter, n_neurons),
        dtype=float,
    )
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)

    for iteration in range(num_iter):
        R_null = R.copy()

        for neuron_idx in range(n_neurons):
            pos_draw = rng.permutation(pos_idx)
            neg_draw = rng.permutation(neg_idx)

            R_null[pos_idx, neuron_idx] = R[
                pos_draw,
                neuron_idx,
            ]
            R_null[neg_idx, neuron_idx] = R[
                neg_draw,
                neuron_idx,
            ]

        C_null = am.compute_noise_covariance(
            R_null,
            noise_mask,
        )
        _, _, eigenvalues = am.get_eigenspectrum(C_null)
        null_eigenvalues[iteration] = eigenvalues

    null_p95 = np.percentile(
        null_eigenvalues,
        95,
        axis=0,
    )

    k_pa = 0
    while (
        k_pa < n_neurons
        and observed_eigenvalues[k_pa] > null_p95[k_pa]
    ):
        k_pa += 1

    return k_pa


def run_one_eid(
    one,
    atlas,
    eid,
    target_prefix,
    min_trials=5,
    min_units=5,
    num_iter=500,
):
    print(f"Processing {eid}...", flush=True)

    trials = ibl_io.load_trials(one, eid)
    pid = ibl_io.pick_best_insertion(
        one,
        atlas,
        eid,
        target_prefix=target_prefix,
    )
    spikes, clusters = ibl_io.load_spikes_and_clusters(
        one,
        atlas,
        pid=pid,
    )
    region_cluster_ids = ibl_io.get_region_cluster_ids(
        clusters,
        target_prefix=target_prefix,
    )

    stim_on = trials["stimOn_times"]
    signed_contrast = ts.get_signed_contrast(trials)

    start = stim_on + 0.04
    end = start + 0.1
    X, unit_ids = fr.compute_static_firing_rates(
        spikes,
        region_cluster_ids,
        start=start,
        end=end,
    )
    X_filtered, unit_mask = fr.filter_active_units(
        X,
        min_units=min_units,
    )
    if X_filtered is None:
        raise RuntimeError(
            f"Not enough active units for eid={eid}: "
            f"{int(np.sum(unit_mask))} < {min_units}"
        )

    high_mask = ts.get_high_masks(
        signed_contrast,
        min_trials=min_trials,
    )
    pos_mask, neg_mask = ts.get_pos_neg_masks(
        signed_contrast,
        high_mask=high_mask,
        min_trials=min_trials,
    )
    condition_masks = {
        "pos": pos_mask,
        "neg": neg_mask,
    }
    R, condition_means, residual_mask = (
        am.noise_residuals_by_condition(
            X_filtered,
            condition_masks,
        )
    )

    k_pa = find_K_pa(
        R,
        pos_mask,
        neg_mask,
        num_iter=num_iter,
    )

    print(
        f"Finished {eid} | pid={pid} | "
        f"n_trials={X_filtered.shape[0]} | "
        f"n_units={X_filtered.shape[1]} | "
        f"k_pa={k_pa}",
        flush=True,
    )

    return {
        "eid": str(eid),
        "k_pa": int(k_pa),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="/scratch/midway3/xiaorantu/ONE",
    )
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--session-index", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    target_prefix = args.target_prefix.strip()
    if not target_prefix:
        parser.error("--target-prefix cannot be empty")

    data_path = (
        REPO_ROOT
        / "results"
        / "region_scan"
        / target_prefix
        / f"{target_prefix}_subjects_by_lab.json"
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Region-scan input not found: {data_path}"
        )

    if args.output_csv is None:
        output_csv = (
            REPO_ROOT
            / "results"
            / "k_pa"
            / target_prefix
            / f"{target_prefix}_kpa_summary.csv"
        )
    else:
        output_csv = Path(args.output_csv)
        if not output_csv.is_absolute():
            output_csv = REPO_ROOT / output_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {data_path}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print("Setting up ONE...", flush=True)
    one = ibl_io.one_setup(cache_dir=args.cache_dir)
    print(f"ONE cache directory: {one.cache_dir}", flush=True)

    atlas = AllenAtlas()
    eids = ibl_io.build_eids_from_results(data_path)
    if args.session_index is not None:
        eids_to_run = [eids[args.session_index]]
    else:
        eids_to_run = eids[: args.max_sessions]

    rows = []
    for eid in eids_to_run:
        try:
            row = run_one_eid(
                one=one,
                atlas=atlas,
                eid=eid,
                target_prefix=target_prefix,
                min_trials=args.min_trials,
                min_units=args.min_units,
                num_iter=args.num_iter,
            )
            rows.append(row)
        except Exception as exc:
            print(
                f"FAILED eid {eid}: {exc!r}",
                flush=True,
            )

    summary_df = pd.DataFrame(
        rows,
        columns=["eid", "k_pa"],
    )
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved {len(summary_df)} sessions to {output_csv}")


if __name__ == "__main__":
    main()