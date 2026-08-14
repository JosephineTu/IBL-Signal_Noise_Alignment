import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
import argparse
from iblatlas.atlas import AllenAtlas

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
sys.path.insert(0, str(REPO_ROOT))

import src.conditional_fisher_null as cf
import src.firing_rates as fr
import src.ibl_io as ibl_io
import src.trial_selection as ts
from src.time_bin import run_timebinned_metric
from run_timebinned_alignment import make_bin_tag

def run_metric_by_bin(fr_tb, times, *, signed_contrast, min_trials, min_units):
    n_bins = fr_tb.shape[1]
    rows = []
    full_outputs = [None] * n_bins
    unit_masks = [None] * n_bins
    statuses = ["not_run"] * n_bins
    n_active_units = np.zeros(n_bins, dtype=int)

    for bin_idx in range(n_bins):
        X_bin, unit_mask = fr.filter_active_units(fr_tb[:, bin_idx, :], min_units=min_units)
        unit_masks[bin_idx] = unit_mask
        n_active_units[bin_idx] = int(np.sum(unit_mask))

        if X_bin is None:
            statuses[bin_idx] = "insufficient_active_units"
            print(f"Skipping bin {bin_idx}: insufficient active units", flush=True)
            continue

        try:
            bin_df, bin_outputs = run_timebinned_metric(X_bin[:, np.newaxis, :],
                                                        np.asarray([times[bin_idx]]),
                                                        cf.compute_conditional_fisher_null,
                                                        signed_contrast=signed_contrast,
                                                        min_trials=min_trials)
        except Exception as e:
            statuses[bin_idx] = "error"
            print(f"Error processing bin {bin_idx}: {e}", flush=True)
            continue

        if len(bin_df) != 1:
            statuses[bin_idx] = "error"
            print(f"Unexpected output for bin {bin_idx}: expected 1 row, got {len(bin_df)}", flush=True)
            continue
        bin_df = bin_df.copy()
        bin_df["bin_idx"] = bin_idx
        rows.append(bin_df)

        full_outputs[bin_idx] = (bin_outputs[0] if bin_outputs else None)
        statuses[bin_idx] = "ok"

    if rows:
        time_df = (pd.concat(rows, ignore_index=True).set_index("bin_idx").reindex(range(n_bins)).reset_index())
    else:
        time_df = pd.DataFrame({"bin_idx": np.arange(n_bins)})

    time_df["time"] = np.asarray(times)
    time_df["n_active_units"] = n_active_units
    time_df["status"] = statuses

    return time_df, full_outputs, unit_masks


def run_one_eid(one, atlas, eid, target_prefix, t_start, t_end, bin_size, step_size, k=3, min_trials=5, min_units=5):
    print(f"Processing {eid}...")
    trials = ibl_io.load_trials(one, eid)
    pid = ibl_io.pick_best_insertion(one, atlas, eid, target_prefix=target_prefix)
    spikes, clusters = ibl_io.load_spikes_and_clusters(one, atlas, pid=pid)
    region_cluster_ids = ibl_io.get_region_cluster_ids(clusters, target_prefix=target_prefix)
    stim_on = trials["stimOn_times"]
    signed_contrast = ts.get_signed_contrast(trials)
    time_windows = fr.make_time_windows(t_start, t_end, bin_size, step_size)
    times = np.mean(time_windows, axis=1)
    fr_tb, unit_ids = fr.compute_time_resolved_firing_rates(spikes, stim_on, region_cluster_ids, time_windows)
    time_df, full_outputs, unit_masks = run_metric_by_bin(fr_tb, times, signed_contrast=signed_contrast, min_trials=min_trials, min_units=min_units)
    time_df.insert(0, "eid", eid)
    time_df.insert(1, "target_prefix", target_prefix)
    time_df.insert(2, "bin_start", time_windows[:, 0])
    time_df.insert(3, "bin_end", time_windows[:, 1])
    details = {
        "eid": eid,
        "target_prefix": target_prefix,
        "unit_masks": unit_masks,
        "full_outputs": full_outputs,
    }
    print(f"Finished processing {eid}.")
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
    parser.add_argument("--max-sessions", type=int, default=10)
    args = parser.parse_args()

    target_prefix = args.target_prefix.strip()
    if not target_prefix:
        parser.error("--target-prefix cannot be empty")
    data_path = (REPO_ROOT / "results" / "region_scan" / target_prefix / f"{target_prefix}_subjects_by_lab.json")
    output_dir = (REPO_ROOT / "results" / "conditional_fisher" / target_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Setting up ONE...", flush=True)
    one = ibl_io.one_setup(cache_dir="/scratch/midway3/xiaorantu/ONE")
    print(
    f"ONE cache directory: {one.cache_dir}",
    flush=True,
    )
    atlas = AllenAtlas()
    eids = ibl_io.build_eids_from_results(data_path)
    eids_to_run = eids[: args.max_sessions]
    all_rows = []
    details = {}
    bin_tag = make_bin_tag(args.t_start, args.t_end, args.bin_size, args.step_size, k=3)
    for eid in eids_to_run:
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
                k=3,
                min_trials=args.min_trials,
                min_units=args.min_units
            )
            all_rows.append(time_df)
            details[eid] = detail
        except Exception as e:
            print(f"Error processing {eid}: {e}", flush=True)
            details[eid] = None
        summary_csv = output_dir / f"{bin_tag}_conditional_fisher_summary.csv"
        if all_rows:
            summary_df = pd.concat(all_rows, ignore_index=True)
            summary_df.to_csv(summary_csv, index=False)

        details_pickle = output_dir / f"{bin_tag}_conditional_fisher_details.pkl"
        with open(details_pickle, "wb") as f:
            pickle.dump(details, f)

        print(f"\nSaved summary to {summary_csv}")
        print(f"Saved details to {details_pickle}")

if __name__ == "__main__":
    main()












    
