"""
session_trial_side_stats.py

"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from iblatlas.atlas import AllenAtlas
from one.api import ONE

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

from run_0_100ms_decoders import build_eids, load_session_0_100ms  # noqa: E402

ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"

SUMMARY_FIELDS = [
    "eid",
    "pid",
    "target_prefix",
    "status",
    "error",
    "n_trials",
    "n_left",
    "n_right",
    "n_zero",
    "left_fraction",
    "n_total_units",
]

MERGED_EXTRA_FIELDS = ["epsilon", "se_intercept", "p_value", "slope", "epsilon_status"]


def compute_side_stats(one, atlas, eid, target_prefix):
    loaded = load_session_0_100ms(
        one=one,
        atlas=atlas,
        eid=eid,
        target_prefix=target_prefix,
    )
    signed_contrast = np.asarray(loaded["signed_contrast"])
    n_total_units = loaded["X"].shape[1]

    n_left = int(np.sum(signed_contrast < 0))
    n_right = int(np.sum(signed_contrast > 0))
    n_zero = int(np.sum(signed_contrast == 0))
    n_trials = int(signed_contrast.shape[0])
    left_fraction = (
        n_left / (n_left + n_right) if (n_left + n_right) > 0 else float("nan")
    )

    return {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "status": "ok",
        "error": "",
        "n_trials": n_trials,
        "n_left": n_left,
        "n_right": n_right,
        "n_zero": n_zero,
        "left_fraction": left_fraction,
        "n_total_units": int(n_total_units),
    }


def write_csv(rows, path, fieldnames):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def try_merge_with_epsilon_summary(rows, target_prefix, output_dir):
    epsilon_summary_path = (
        REPO_ROOT
        / "results"
        / "subsampling_epsilon"
        / target_prefix
        / f"{target_prefix}_subsampling_epsilon_summary.csv"
    )
    if not epsilon_summary_path.is_file():
        print(
            f"(no epsilon summary found at {epsilon_summary_path} -- "
            "skipping merge; run run_subsampling_epsilon.py first if you "
            "want the merged CSV)"
        )
        return None

    epsilon_by_eid = {}
    with open(epsilon_summary_path, newline="") as f:
        for row in csv.DictReader(f):
            epsilon_by_eid[row["eid"]] = row

    merged_fields = SUMMARY_FIELDS + MERGED_EXTRA_FIELDS
    merged_rows = []
    for row in rows:
        eps_row = epsilon_by_eid.get(row["eid"])
        merged = dict(row)
        if eps_row is not None:
            merged["epsilon"] = eps_row.get("epsilon", "")
            merged["se_intercept"] = eps_row.get("se_intercept", "")
            merged["p_value"] = eps_row.get("p_value", "")
            merged["slope"] = eps_row.get("slope", "")
            merged["epsilon_status"] = eps_row.get("status", "")
        else:
            merged["epsilon"] = merged["se_intercept"] = ""
            merged["p_value"] = merged["slope"] = ""
            merged["epsilon_status"] = "not_found"
        merged_rows.append(merged)

    merged_path = output_dir / f"{target_prefix}_trial_side_stats_merged.csv"
    write_csv(merged_rows, merged_path, merged_fields)
    print(f"merged={merged_path}")
    return merged_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    input_json = (
        REPO_ROOT
        / "results"
        / "region_scan"
        / args.target_prefix
        / f"{args.target_prefix}_subjects_by_lab.json"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "results" / "session_trial_side_stats" / args.target_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.target_prefix}_trial_side_stats.csv"

    eids = build_eids(input_json)
    print(f"target_prefix={args.target_prefix}")
    print(f"input_json={input_json}")
    print(f"n_sessions={len(eids)}")
    print(f"output_dir={output_dir}")

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=ONE_CACHE_DIR,
    )
    atlas = AllenAtlas()

    rows = []
    for session_index, eid in enumerate(eids, start=1):
        print(f"[{session_index}/{len(eids)}] eid={eid}")
        try:
            row = compute_side_stats(one, atlas, eid, args.target_prefix)
            print(
                f"  ok: n_trials={row['n_trials']}, n_left={row['n_left']}, "
                f"n_right={row['n_right']}, n_zero={row['n_zero']}, "
                f"left_fraction={row['left_fraction']:.3f}, "
                f"n_total_units={row['n_total_units']}"
            )
        except Exception as exc:
            row = {
                "eid": eid,
                "pid": "",
                "target_prefix": args.target_prefix,
                "status": "failed",
                "error": repr(exc),
                "n_trials": "",
                "n_left": "",
                "n_right": "",
                "n_zero": "",
                "left_fraction": "",
                "n_total_units": "",
            }
            print(f"  failed: {row['error']}")

        rows.append(row)
        write_csv(rows, output_path, SUMMARY_FIELDS)

    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: {n_ok}/{len(rows)} sessions succeeded")
    print(f"summary={output_path}")

    try_merge_with_epsilon_summary(rows, args.target_prefix, output_dir)


if __name__ == "__main__":
    main()