from __future__ import annotations

import argparse
import pickle
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
import src.conditional_fisher_null as cf
import src.firing_rates as fr
import src.ibl_io as ibl_io
import src.trial_selection as ts


def format_tag_value(value):
    return str(float(value)).replace("-", "m").replace(".", "p")


def make_time_tag(t_start, t_end, bin_size, step_size):
    return (
        f"t{format_tag_value(t_start)}"
        f"to{format_tag_value(t_end)}"
        f"_bin{format_tag_value(bin_size)}"
        f"_step{format_tag_value(step_size)}"
    )


def compute_both_nulls(
    X,
    *,
    signed_contrast,
    k_pa,
    min_trials,
    num_iter,
):
    """Compute finite-trial-corrected Haar and pairing nulls."""
    X = np.asarray(X, dtype=float)
    signed_contrast = np.asarray(signed_contrast, dtype=float)

    high_mask = ts.get_high_masks(
        signed_contrast,
        min_trials=min_trials,
    )
    if high_mask is None:
        raise RuntimeError("Not enough high-contrast trials")

    pos_mask, neg_mask = ts.get_pos_neg_masks(
        signed_contrast,
        high_mask=high_mask,
        min_trials=min_trials,
    )
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    u_sig, signal_axis = am.compute_signal_axis(
        X,
        pos_mask,
        neg_mask,
        min_trials=min_trials,
    )

    condition_masks = {
        "pos": pos_mask,
        "neg": neg_mask,
    }
    residuals, _, residual_mask = am.noise_residuals_by_condition(
        X,
        condition_masks,
    )
    noise_mask = pos_mask | neg_mask
    C = am.compute_noise_covariance(
        residuals,
        noise_mask & residual_mask,
    )

    if k_pa < 2:
        raise ValueError(f"k_pa must be at least 2, got {k_pa}")
    if k_pa > C.shape[0]:
        raise ValueError(
            f"k_pa={k_pa} exceeds the number of active units {C.shape[0]}"
        )

    haar = cf.k_constrained_haar_null(
        k_pa=k_pa,
        signal_axis=signal_axis,
        C=C,
        n_pos=n_pos,
        n_neg=n_neg,
        num_iter=num_iter,
    )
    pairing = cf.pairing_permutation_null(
        k_pa=k_pa,
        signal_axis=signal_axis,
        C=C,
        n_pos=n_pos,
        n_neg=n_neg,
        num_iter=num_iter,
    )

    if not np.isclose(
        haar["T_obs_raw"],
        pairing["T_obs_raw"],
    ):
        raise RuntimeError(
            "The Haar and pairing functions returned different T_obs_raw"
        )
    if not np.isclose(
        haar["T_obs_corrected"],
        pairing["T_obs_corrected"],
    ):
        raise RuntimeError(
            "The Haar and pairing functions returned different "
            "T_obs_corrected"
        )

    haar_null = np.asarray(
        haar["null_Ts_corrected"],
        dtype=float,
    )
    pairing_null = np.asarray(
        pairing["null_Ts_corrected"],
        dtype=float,
    )

    if haar["status"] == "ok":
        haar_null_mean = float(np.mean(haar_null))
        haar_null_median = float(np.median(haar_null))
        haar_null_p05 = float(np.percentile(haar_null, 5))
        haar_null_p95 = float(np.percentile(haar_null, 95))
    else:
        haar_null_mean = np.nan
        haar_null_median = np.nan
        haar_null_p05 = np.nan
        haar_null_p95 = np.nan

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "k_pa": int(k_pa),
        "signal_norm": float(np.linalg.norm(signal_axis)),
        "T_obs_raw": float(haar["T_obs_raw"]),
        "T_obs_corrected": float(haar["T_obs_corrected"]),
        "mean_noise_scale": float(haar["mean_noise_scale"]),
        "haar_status": haar["status"],
        "haar_null_mean_corrected": haar_null_mean,
        "haar_null_median_corrected": haar_null_median,
        "haar_null_p05_corrected": haar_null_p05,
        "haar_null_p95_corrected": haar_null_p95,
        "haar_p_lower_corrected": float(
            haar["p_lower_corrected"]
        ),
        "haar_p_upper_corrected": float(
            haar["p_upper_corrected"]
        ),
        "pairing_null_mean_corrected": float(
            np.mean(pairing_null)
        ),
        "pairing_null_median_corrected": float(
            np.median(pairing_null)
        ),
        "pairing_null_p05_corrected": float(
            np.percentile(pairing_null, 5)
        ),
        "pairing_null_p95_corrected": float(
            np.percentile(pairing_null, 95)
        ),
        "pairing_p_lower_corrected": float(
            pairing["p_lower_corrected"]
        ),
        "pairing_p_upper_corrected": float(
            pairing["p_upper_corrected"]
        ),
        "u_sig": u_sig,
        "signal_axis": signal_axis,
        "C": C,
        "haar": haar,
        "pairing": pairing,
    }

def run_metric_by_bin(
    fr_tb,
    times,
    *,
    signed_contrast,
    k_pa,
    min_trials,
    min_units,
    num_iter,
):
    n_bins = fr_tb.shape[1]
    rows = []
    full_outputs = [None] * n_bins
    unit_masks = [None] * n_bins

    for bin_idx in range(n_bins):
        X_bin, unit_mask = fr.filter_active_units(
            fr_tb[:, bin_idx, :],
            min_units=min_units,
        )
        unit_masks[bin_idx] = unit_mask
        n_active_units = int(np.sum(unit_mask))

        row = {
            "bin_idx": int(bin_idx),
            "time": float(times[bin_idx]),
            "k_pa": int(k_pa),
            "n_active_units": n_active_units,
        }

        if X_bin is None:
            row["status"] = "insufficient_active_units"
            rows.append(row)
            print(
                f"Skipping bin {bin_idx}: insufficient active units",
                flush=True,
            )
            continue

        if n_active_units < k_pa:
            row["status"] = "insufficient_units_for_k_pa"
            rows.append(row)
            print(
                f"Skipping bin {bin_idx}: active units={n_active_units} "
                f"is smaller than k_pa={k_pa}",
                flush=True,
            )
            continue

        try:
            out = compute_both_nulls(
                X_bin,
                signed_contrast=signed_contrast,
                k_pa=k_pa,
                min_trials=min_trials,
                num_iter=num_iter,
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            rows.append(row)
            print(
                f"Error processing bin {bin_idx}: {exc}",
                flush=True,
            )
            continue

        full_outputs[bin_idx] = out
        row["status"] = "ok"
        for key, value in out.items():
            if np.isscalar(value):
                row[key] = value
        rows.append(row)

    return pd.DataFrame(rows), full_outputs, unit_masks


def run_one_eid(
    one,
    atlas,
    eid,
    target_prefix,
    t_start,
    t_end,
    bin_size,
    step_size,
    k_pa,
    min_trials=5,
    min_units=5,
    num_iter=500,
):
    print(f"Processing {eid}...")

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
    time_windows = fr.make_time_windows(
        t_start,
        t_end,
        bin_size,
        step_size,
    )
    times = np.mean(time_windows, axis=1)
    fr_tb, unit_ids = fr.compute_time_resolved_firing_rates(
        spikes,
        stim_on,
        region_cluster_ids,
        time_windows,
    )

    time_df, full_outputs, unit_masks = run_metric_by_bin(
        fr_tb,
        times,
        signed_contrast=signed_contrast,
        k_pa=k_pa,
        min_trials=min_trials,
        min_units=min_units,
        num_iter=num_iter,
    )

    time_df.insert(0, "eid", eid)
    time_df.insert(1, "pid", pid)
    time_df.insert(2, "target_prefix", target_prefix)
    time_df.insert(3, "bin_start", time_windows[:, 0])
    time_df.insert(4, "bin_end", time_windows[:, 1])

    details = {
        "eid": eid,
        "pid": pid,
        "target_prefix": target_prefix,
        "time_windows": time_windows,
        "unit_ids": unit_ids,
        "unit_masks": unit_masks,
        "full_outputs": full_outputs,
    }

    print(f"Finished processing {eid}.", flush=True)
    return time_df, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", type=str, required=True)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--bin-size", type=float, default=0.03)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument(
        "--kpa-csv",
        type=str,
        default=None,
        help="CSV containing one row per session with columns eid and k_pa.",
    )
    args = parser.parse_args()

    target_prefix = args.target_prefix.strip()
    if not target_prefix:
        parser.error("--target-prefix cannot be empty")

    time_tag = make_time_tag(
        args.t_start,
        args.t_end,
        args.bin_size,
        args.step_size,
    )
    data_path = (
        REPO_ROOT
        / "results"
        / "region_scan"
        / target_prefix
        / f"{target_prefix}_subjects_by_lab.json"
    )
    output_dir = (
        REPO_ROOT
        / "results"
        / "conditional_fisher"
        / target_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.kpa_csv is None:
        kpa_csv = (
            REPO_ROOT
            / "results"
            / "k_pa"
            / target_prefix
            / f"{target_prefix}_kpa_summary.csv"
        )
    else:
        kpa_csv = Path(args.kpa_csv)
        if not kpa_csv.is_absolute():
            kpa_csv = REPO_ROOT / kpa_csv

    print(f"Reading precomputed k_pa from {kpa_csv}", flush=True)
    kpa_df = pd.read_csv(kpa_csv)
    kpa_by_eid = {
        str(row.eid): int(row.k_pa)
        for row in kpa_df.itertuples(index=False)
    }

    print("Setting up ONE...", flush=True)
    one = ibl_io.one_setup(
        cache_dir="/scratch/midway3/xiaorantu/ONE"
    )
    print(f"ONE cache directory: {one.cache_dir}", flush=True)

    atlas = AllenAtlas()
    eids = ibl_io.build_eids_from_results(data_path)
    eids_to_run = eids[: args.max_sessions]

    all_rows = []
    details = {}

    for eid in eids_to_run:
        if str(eid) not in kpa_by_eid:
            print(
                f"Skipping {eid}: missing precomputed session k_pa",
                flush=True,
            )
            details[eid] = None
            continue

        k_pa = kpa_by_eid[str(eid)]
        if k_pa <= 3:
            print(
                f"Skipping {eid}: k_pa={k_pa} is not greater than 3",
                flush=True,
            )
            details[eid] = None
            continue

        try:
            time_df, detail = run_one_eid(
                one=one,
                atlas=atlas,
                eid=eid,
                target_prefix=target_prefix,
                t_start=args.t_start,
                t_end=args.t_end,
                bin_size=args.bin_size,
                step_size=args.step_size,
                k_pa=k_pa,
                min_trials=args.min_trials,
                min_units=args.min_units,
                num_iter=args.num_iter,
            )
            all_rows.append(time_df)
            details[eid] = detail
        except Exception as exc:
            print(f"Error processing {eid}: {exc}", flush=True)
            details[eid] = None

    summary_csv = (
        output_dir
        / f"{time_tag}_kpa_conditional_fisher_summary.csv"
    )
    if all_rows:
        summary_df = pd.concat(all_rows, ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)

    details_pickle = (
        output_dir
        / f"{time_tag}_kpa_conditional_fisher_details.pkl"
    )
    with open(details_pickle, "wb") as file:
        pickle.dump(details, file)

    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved details to {details_pickle}")


if __name__ == "__main__":
    main()












    
