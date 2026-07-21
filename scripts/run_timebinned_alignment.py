# run_timebinned_alignment.py
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import argparse
import pickle
import numpy as np
import pandas as pd
from iblatlas.atlas import AllenAtlas
import src.ibl_io as ibl_io
import src.alignment_metrics as am
import src.trial_selection as ts
import src.firing_rates as fr
from src.time_bin import run_timebinned_metric

def run_one_eid(one, atlas, eid, target_prefix='VISp', 
                t_start=0.0, t_end=0.4, bin_size=0.08, 
                step_size=0.02, k=3, min_trials=5, 
                output_dir=Path('results/timebinned_alignment'),):
    print(f'Processing {eid}...')
    trials = ibl_io.load_trials(one, eid)
    pid = ibl_io.pick_best_insertion(one, atlas, eid, target_prefix=target_prefix)
    spikes, clusters = ibl_io.load_spikes_and_clusters(one, atlas, pid=pid)
    region_cluster_ids = ibl_io.get_region_cluster_ids(clusters, target_prefix=target_prefix)
    stim_on = trials['stimOn_times']
    signed_contrast = ts.get_signed_contrast(trials)
    time_windows = fr.make_time_windows(t_start, t_end, bin_size, step_size)
    times = np.mean(time_windows, axis=1)
    fr_tb, unit_ids = fr.compute_time_resolved_firing_rates(spikes, stim_on, 
                                                            region_cluster_ids, time_windows)
    fr_tb_filtered, unit_masks = fr.filter_active_units(fr_tb, min_units=5)
    time_df, full_outputs = run_timebinned_metric(fr_tb_filtered, times, am.signal_noise_alignment, 
                                                  signed_contrast=signed_contrast, 
                                                  k=k, min_trials=min_trials, eps=1e-12)
    time_df.insert(0, "eid", eid)
    time_df.insert(1, "pid", pid)
    time_df.insert(2, "target_prefix", target_prefix)
    time_df.insert(3, "bin_start", time_windows[:, 0])
    time_df.insert(4, "bin_end", time_windows[:, 1])
    time_df.insert(5, "bin_size", float(bin_size))
    time_df.insert(6, "step_size", float(step_size))

    bin_tag = make_bin_tag(
        t_start=t_start,
        t_end=t_end,
        bin_size=bin_size,
        step_size=step_size,
        k=k,
    )

    details = {
        'eid': eid,
        'pid': pid,
        'target_prefix': target_prefix,
        'bin_tag': bin_tag,
        'time_windows': time_windows,
        'full_outputs': full_outputs,
    }

    print(
        f"  finished {eid} | "
        f"n_trials={fr_tb.shape[0]}, n_bins={fr_tb.shape[1]}, n_units={fr_tb.shape[2]}"
    )

    return time_df, details

def make_bin_tag(t_start, t_end, bin_size, step_size, k):
    def fmt(x):
        return str(x).replace(".", "p").replace("-", "m")

    return (
        f"t{fmt(t_start)}to{fmt(t_end)}"
        f"_bin{fmt(bin_size)}"
        f"_step{fmt(step_size)}"
        f"_k{k}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="/scratch/midway3/xiaorantu/ONE",
    )
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "results" / "VISp_subjects_by_lab.json"),
    )
    parser.add_argument(
        "--output-dir",
        default="results/timebinned_alignment",
    )
    parser.add_argument("--target-prefix", default="VISp")
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=0.4)
    parser.add_argument("--bin-size", type=float, default=0.08)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=5)
    parser.add_argument("--session-index", type=int, default=None)
    args = parser.parse_args()
    one = ibl_io.one_setup(cache_dir=args.cache_dir)
    atlas = AllenAtlas()
    data_path = Path(args.data_path)
    eids = ibl_io.build_eids_from_results(data_path)
    if args.session_index is not None:
        eids_to_run = [eids[args.session_index]]
    else:
        eids_to_run = eids[: args.max_sessions]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    details = {}
    for eid in eids_to_run:
        try:
            time_df, detail = run_one_eid(
                one=one,
                atlas=atlas,
                eid=eid,
                target_prefix=args.target_prefix,
                t_start=args.t_start,
                t_end=args.t_end,
                bin_size=args.bin_size,
                step_size=args.step_size,
                k=args.k,
                min_trials=args.min_trials,
                output_dir=output_dir,
            )
            all_rows.append(time_df)
            details[eid] = detail
        except Exception as exc:
            print(f"FAILED eid {eid}: {repr(exc)}")
            fail_row = pd.DataFrame(
                [
                    {
                        "eid": eid,
                        "status": "failed",
                        "error": repr(exc),
                    }
                ]
            )
            all_rows.append(fail_row)
    summary_df = pd.concat(all_rows, ignore_index=True)
    bin_tag = make_bin_tag(
        t_start=args.t_start,
        t_end=args.t_end,
        bin_size=args.bin_size,
        step_size=args.step_size,
        k=args.k,
    )
    summary_csv = output_dir / f"{bin_tag}_timebinned_alignment_summary.csv"
    details_pkl = output_dir / f"{bin_tag}_timebinned_alignment_details.pkl"
    summary_df.to_csv(summary_csv, index=False)
    with open(details_pkl, "wb") as f:
        pickle.dump(details, f)
    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved details to {details_pkl}")
if __name__ == "__main__":
    main()








