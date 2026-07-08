# check_prestim_recording.py
from pathlib import Path
import sys
import argparse
import gc
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from iblatlas.atlas import AllenAtlas
import src.ibl_io as ibl_io


COLUMNS = [
    "eid",
    "pid",
    "status",
    "first_stimOn",
    "recording_start_all_spikes",
    "recording_end_all_spikes",
    "pre_first_stim_duration_all_spikes",
    "n_spikes_before_first_stim_all",
    "n_units_region",
    "recording_start_region_spikes",
    "recording_end_region_spikes",
    "pre_first_stim_duration_region_spikes",
    "n_region_spikes_before_first_stim",
    "error",
]


def finite_min(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else np.nan


def finite_max(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan


def check_one_eid(one, atlas, eid, target_prefix="VISp"):
    row = {
        "eid": eid,
        "pid": "",
        "status": "failed",
        "first_stimOn": np.nan,
        "recording_start_all_spikes": np.nan,
        "recording_end_all_spikes": np.nan,
        "pre_first_stim_duration_all_spikes": np.nan,
        "n_spikes_before_first_stim_all": np.nan,
        "n_units_region": np.nan,
        "recording_start_region_spikes": np.nan,
        "recording_end_region_spikes": np.nan,
        "pre_first_stim_duration_region_spikes": np.nan,
        "n_region_spikes_before_first_stim": np.nan,
        "error": "",
    }

    try:
        trials = ibl_io.load_trials(one, eid)
        stim_on = np.asarray(trials["stimOn_times"], float)
        stim_on = stim_on[np.isfinite(stim_on)]

        if stim_on.size == 0:
            raise RuntimeError("No finite stimOn_times found.")

        first_stim = float(np.min(stim_on))
        row["first_stimOn"] = first_stim

        pid = ibl_io.pick_best_insertion(
            one,
            atlas,
            eid,
            target_prefix=target_prefix,
        )
        row["pid"] = pid

        print(f"  pid: {pid}")
        print(f"  first stimOn: {first_stim:.3f}")

        spikes, clusters = ibl_io.load_spikes_and_clusters(
            one,
            atlas,
            pid=pid,
        )

        spike_times = np.asarray(spikes["times"], float)
        spike_clusters = np.asarray(spikes["clusters"])

        rec_start = finite_min(spike_times)
        rec_end = finite_max(spike_times)

        row["recording_start_all_spikes"] = rec_start
        row["recording_end_all_spikes"] = rec_end
        row["pre_first_stim_duration_all_spikes"] = first_stim - rec_start
        row["n_spikes_before_first_stim_all"] = int(np.sum(spike_times < first_stim))

        region_cluster_ids = ibl_io.get_region_cluster_ids(
            clusters,
            target_prefix=target_prefix,
        )
        region_cluster_ids = np.asarray(region_cluster_ids)

        row["n_units_region"] = int(region_cluster_ids.size)

        region_spike_mask = np.isin(spike_clusters, region_cluster_ids)
        region_spike_times = spike_times[region_spike_mask]

        region_start = finite_min(region_spike_times)
        region_end = finite_max(region_spike_times)

        row["recording_start_region_spikes"] = region_start
        row["recording_end_region_spikes"] = region_end
        row["pre_first_stim_duration_region_spikes"] = first_stim - region_start
        row["n_region_spikes_before_first_stim"] = int(
            np.sum(region_spike_times < first_stim)
        )

        row["status"] = "ok"

        print(
            "  all-spike pre-first-stim duration: "
            f"{row['pre_first_stim_duration_all_spikes']:.2f} s"
        )
        print(
            f"  {target_prefix} pre-first-stim duration: "
            f"{row['pre_first_stim_duration_region_spikes']:.2f} s"
        )
        print(
            f"  {target_prefix} spikes before first stim: "
            f"{row['n_region_spikes_before_first_stim']}"
        )

    except Exception as exc:
        row["error"] = repr(exc)
        print(f"  FAILED: {repr(exc)}")

    finally:
        for name in [
            "trials",
            "spikes",
            "clusters",
            "spike_times",
            "spike_clusters",
            "region_cluster_ids",
            "region_spike_mask",
            "region_spike_times",
            "stim_on",
        ]:
            if name in locals():
                del locals()[name]
        gc.collect()

    return row


def merge_parts(parts_dir, out_path):
    parts = sorted(Path(parts_dir).glob("pre_first_stim_recording_check_*.csv"))

    if len(parts) == 0:
        raise RuntimeError(f"No part files found in {parts_dir}")

    dfs = [pd.read_csv(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)

    df = df[COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nMerged parts:")
    print(f"  n_parts: {len(parts)}")
    print(f"  saved: {out_path}")

    print("\nSummary:")
    print(df[[
        "eid",
        "status",
        "pre_first_stim_duration_all_spikes",
        "pre_first_stim_duration_region_spikes",
        "n_units_region",
        "n_region_spikes_before_first_stim",
    ]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/scratch/midway3/xiaorantu/ONE")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--parts-dir", default=None)
    parser.add_argument("--target-prefix", default="VISp")
    parser.add_argument("--max-sessions", type=int, default=5)
    parser.add_argument("--session-index", type=int, default=None)
    parser.add_argument("--merge-parts", action="store_true")
    args = parser.parse_args()

    data_path = (
        Path(args.data_path)
        if args.data_path is not None
        else REPO_ROOT / "results" / "VISp_subjects_by_lab.json"
    )

    out_path = (
        Path(args.out_path)
        if args.out_path is not None
        else REPO_ROOT / "results" / "pre_first_stim_recording_check.csv"
    )

    parts_dir = (
        Path(args.parts_dir)
        if args.parts_dir is not None
        else REPO_ROOT / "results" / "pre_first_stim_recording_check_parts"
    )

    if args.merge_parts:
        merge_parts(parts_dir, out_path)
        return

    one = ibl_io.one_setup(cache_dir=args.cache_dir)
    atlas = AllenAtlas()

    eids = ibl_io.build_eids_from_results(data_path)
    eids = eids[: args.max_sessions]

    print(f"Loaded {len(eids)} eids from {data_path}")

    if args.session_index is None:
        rows = []
        for i, eid in enumerate(eids, start=1):
            print(f"\n[{i}/{len(eids)}] {eid}")
            rows.append(check_one_eid(one, atlas, eid, target_prefix=args.target_prefix))

        df = pd.DataFrame(rows)[COLUMNS]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print("\nSaved:")
        print(out_path)

        print("\nSummary:")
        print(df[[
            "eid",
            "status",
            "pre_first_stim_duration_all_spikes",
            "pre_first_stim_duration_region_spikes",
            "n_units_region",
            "n_region_spikes_before_first_stim",
        ]])

    else:
        eid = eids[args.session_index]

        print(f"\n[{args.session_index + 1}/{len(eids)}] {eid}")
        row = check_one_eid(one, atlas, eid, target_prefix=args.target_prefix)

        parts_dir.mkdir(parents=True, exist_ok=True)
        part_path = parts_dir / f"pre_first_stim_recording_check_{args.session_index:03d}.csv"

        df = pd.DataFrame([row])[COLUMNS]
        df.to_csv(part_path, index=False)

        print("\nSaved part:")
        print(part_path)


if __name__ == "__main__":
    main()