# plot_alignment_summary.py
from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path("/home/xiaorantu/signal_noise_alignment_git")
SUMMARY_SUFFIX = "_timebinned_alignment_summary.csv"
DETAILS_SUFFIX = "_timebinned_alignment_details.pkl"


def short_eid(eid, n=8):
    return str(eid)[:n]


def load_details(details_pkl):
    if not details_pkl.is_file():
        raise FileNotFoundError(f"Details PKL not found: {details_pkl}")

    with details_pkl.open("rb") as file:
        details = pickle.load(file)

    if not isinstance(details, Mapping):
        raise TypeError("Expected details PKL to be a dictionary indexed by eid")

    return {str(eid): value for eid, value in details.items()}


def select_bin_output(full_outputs, bin_index):
    """Select one bin from list- or dictionary-backed full outputs."""
    if isinstance(full_outputs, Mapping):
        for key in (bin_index, str(bin_index)):
            if key in full_outputs:
                return full_outputs[key]
        return None

    try:
        return full_outputs[bin_index]
    except (IndexError, KeyError, TypeError):
        return None


def extract_null_mean(details, eid, bin_index):
    session_details = details.get(str(eid))
    if not isinstance(session_details, Mapping):
        raise KeyError(f"No details for EID {eid}")

    full_outputs = session_details.get("full_outputs")
    if full_outputs is None:
        raise KeyError(f"EID {eid} does not contain full_outputs")

    output = select_bin_output(full_outputs, bin_index)
    if output is None:
        raise KeyError(f"No output for EID {eid}, bin {bin_index}")
    if not isinstance(output, Mapping):
        raise TypeError(
            f"Output for EID {eid}, bin {bin_index} is not a dictionary"
        )
    if "null_cosine2" not in output:
        raise KeyError(
            f"No null_cosine2 for EID {eid}, bin {bin_index}; "
            f"available keys={list(output)[:12]}"
        )

    null_samples = np.asarray(output["null_cosine2"], dtype=float).reshape(-1)
    null_samples = null_samples[np.isfinite(null_samples)]
    if null_samples.size == 0:
        raise ValueError(f"No finite null_cosine2 for EID {eid}, bin {bin_index}")

    return float(np.mean(null_samples))


def load_alignment_improvement(summary_csv, details_pkl):
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()

    if "bin_index" in df.columns:
        bin_column = "bin_index"
    elif "bin_idx" in df.columns:
        bin_column = "bin_idx"
    else:
        raise RuntimeError(f"Missing bin_index/bin_idx column in {summary_csv}")

    required_columns = ["eid", bin_column, "time", "cosine2_top1"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {summary_csv}: {missing}")

    df["eid"] = df["eid"].astype(str)
    for column in (bin_column, "time", "cosine2_top1"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    df[bin_column] = df[bin_column].astype(int)

    details = load_details(details_pkl)
    valid_indices = []
    null_means = []

    for row_index, row in df.iterrows():
        try:
            null_mean = extract_null_mean(
                details,
                row["eid"],
                int(row[bin_column]),
            )
        except (KeyError, TypeError, ValueError) as error:
            print(
                f"Skipping eid={row['eid']}, "
                f"bin={int(row[bin_column])}: {error}",
                flush=True,
            )
            continue

        valid_indices.append(row_index)
        null_means.append(null_mean)

    if not valid_indices:
        raise RuntimeError(
            "No summary rows matched valid null_cosine2 distributions; "
            "inspect the preceding per-bin messages for the actual PKL keys"
        )

    df = df.loc[valid_indices].copy()
    df["null_cosine2_mean"] = null_means
    df["alignment_improvement"] = (
        df["cosine2_top1"] - df["null_cosine2_mean"]
    )
    return df


def plot_alignment_improvement(df, output_path, target_prefix, bin_tag):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("tab20")

    for index, eid in enumerate(df["eid"].drop_duplicates()):
        session_df = df[df["eid"] == eid].sort_values("time")
        ax.plot(
            session_df["time"],
            session_df["alignment_improvement"],
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            alpha=0.85,
            color=cmap(index % cmap.N),
            label=short_eid(eid),
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("time from stimOn (s)")
    ax.set_ylabel(
        "alignment improvement\n"
        "(observed cos² − permutation-null mean)"
    )
    ax.set_title(
        "Excess alignment over permutation null\n"
        f"{target_prefix} | {bin_tag}"
    )
    ax.legend(
        title="eid",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    args = parser.parse_args()

    target_prefix = args.target_prefix.strip()
    if not target_prefix:
        parser.error("--target-prefix cannot be empty")

    result_dir = (
        PROJECT_DIR / "results" / "timebinned_alignment" / target_prefix
    )
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    summary_files = sorted(result_dir.glob(f"*{SUMMARY_SUFFIX}"))
    if not summary_files:
        raise FileNotFoundError(
            f"No time-binned summary CSV files found in {result_dir}"
        )

    output_dir = (
        PROJECT_DIR
        / "figures"
        / "alignment_summary"
        / target_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(summary_files)} bin tags in {result_dir}")
    n_saved = 0
    n_skipped = 0

    for summary_csv in summary_files:
        bin_tag = summary_csv.name[: -len(SUMMARY_SUFFIX)]
        details_pkl = result_dir / f"{bin_tag}{DETAILS_SUFFIX}"
        output_path = output_dir / f"{bin_tag}_alignment_improvement.png"

        print(f"\nProcessing bin tag: {bin_tag}")
        print(f"Summary CSV: {summary_csv}")
        print(f"Details PKL: {details_pkl}")

        if not details_pkl.is_file():
            print(f"Skipping {bin_tag}: matching details PKL not found")
            n_skipped += 1
            continue

        try:
            df = load_alignment_improvement(summary_csv, details_pkl)
            print(
                f"Loaded {df['eid'].nunique()} sessions "
                f"and {len(df)} session-bin rows"
            )
            plot_alignment_improvement(
                df,
                output_path,
                target_prefix,
                bin_tag,
            )
        except (FileNotFoundError, RuntimeError, TypeError) as error:
            print(f"Skipping {bin_tag}: {error}", flush=True)
            n_skipped += 1
            continue

        n_saved += 1

    print(
        f"\nFinished {target_prefix}: "
        f"saved {n_saved} figure(s), skipped {n_skipped} bin tag(s)"
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()