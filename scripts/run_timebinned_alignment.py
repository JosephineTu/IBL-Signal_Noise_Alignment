# run_timebinned_alignment.py
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from iblatlas.atlas import AllenAtlas


# Normally this file lives in <repo>/scripts/. The fallback also makes a copy
# placed directly in the repository root resolve paths correctly.
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
from src.time_bin import run_timebinned_metric


def _single_output(full_outputs):
    """Unwrap the one-bin result while preserving an unexpected container."""
    if isinstance(full_outputs, (list, tuple)) and len(full_outputs) == 1:
        return full_outputs[0]
    return full_outputs


def run_metric_by_bin(
    fr_tb,
    times,
    *,
    signed_contrast,
    k,
    min_trials,
    min_units,
):
    """Run valid bins and retain invalid bins as NaN rows."""
    n_bins = fr_tb.shape[1]
    rows = []
    full_outputs = [None] * n_bins
    unit_masks = []
    statuses = []
    n_active_units = []

    for bin_index in range(n_bins):
        X_bin, unit_mask = fr.filter_active_units(
            fr_tb[:, bin_index, :],
            min_units=min_units,
        )
        unit_masks.append(unit_mask)
        n_active = int(np.sum(unit_mask))
        n_active_units.append(n_active)

        if X_bin is None:
            statuses.append("insufficient_active_units")
            print(
                f"  skipping bin {bin_index}: "
                f"active units {n_active} < {min_units}",
                flush=True,
            )
            continue

        bin_df, bin_outputs = run_timebinned_metric(
            X_bin[:, np.newaxis, :],
            np.asarray([times[bin_index]], dtype=float),
            am.signal_noise_alignment,
            signed_contrast=signed_contrast,
            k=k,
            min_trials=min_trials,
            eps=1e-12,
        )
        if len(bin_df) != 1:
            raise RuntimeError(
                "Expected one output row for one time bin, "
                f"got {len(bin_df)} for bin {bin_index}"
            )

        bin_df = bin_df.copy()
        bin_df["bin_index"] = bin_index
        rows.append(bin_df)
        full_outputs[bin_index] = _single_output(bin_outputs)
        statuses.append("ok")

    if rows:
        time_df = (
            pd.concat(rows, ignore_index=True)
            .set_index("bin_index")
            .reindex(range(n_bins))
            .rename_axis("bin_index")
            .reset_index()
        )
    else:
        # There are no metric columns to infer, but keep one row per bin so the
        # session and its time axis remain represented in the saved summary.
        time_df = pd.DataFrame({"bin_index": np.arange(n_bins)})

    # Assign these after reindexing so skipped bins retain their coordinates
    # while all unavailable metric columns remain NaN.
    time_df["time"] = np.asarray(times, dtype=float)
    time_df["status"] = statuses
    time_df["n_active_units"] = n_active_units

    return time_df, full_outputs, unit_masks


def run_one_eid(
    one,
    atlas,
    eid,
    target_prefix,
    t_start,
    t_end,
    bin_size,
    step_size,
    k=3,
    min_trials=5,
    min_units=5,
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
        k=k,
        min_trials=min_trials,
        min_units=min_units,
    )

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
        "eid": eid,
        "pid": pid,
        "target_prefix": target_prefix,
        "bin_tag": bin_tag,
        "time_windows": time_windows,
        "full_outputs": full_outputs,
        "unit_masks": unit_masks,
        "unit_ids": unit_ids,
    }

    n_skipped = int((time_df["status"] != "ok").sum())
    print(
        f"  finished {eid} | n_trials={fr_tb.shape[0]}, "
        f"n_bins={fr_tb.shape[1]}, n_units={fr_tb.shape[2]}, "
        f"skipped_bins={n_skipped}",
        flush=True,
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
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--t-start", type=float, default=0.04)
    parser.add_argument("--t-end", type=float, default=0.14)
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-trials", type=int, default=5)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--session-index", type=int, default=None)
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
    output_dir = (
        REPO_ROOT
        / "results"
        / "timebinned_alignment"
        / target_prefix
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Region-scan input not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {data_path}")
    print(f"Output directory: {output_dir}")

    one = ibl_io.one_setup(cache_dir=args.cache_dir)
    atlas = AllenAtlas()
    eids = ibl_io.build_eids_from_results(data_path)
    if args.session_index is not None:
        eids_to_run = [eids[args.session_index]]
    else:
        eids_to_run = eids[: args.max_sessions]

    all_rows = []
    details = {}
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
                k=args.k,
                min_trials=args.min_trials,
                min_units=args.min_units,
            )
            all_rows.append(time_df)
            details[eid] = detail
        except Exception as exc:
            print(f"FAILED eid {eid}: {exc!r}", flush=True)
            all_rows.append(
                pd.DataFrame(
                    [{"eid": eid, "status": "failed", "error": repr(exc)}]
                )
            )

    if not all_rows:
        raise RuntimeError("No sessions were selected")

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
    with open(details_pkl, "wb") as file:
        pickle.dump(details, file)

    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved details to {details_pkl}")


if __name__ == "__main__":
    main()
















