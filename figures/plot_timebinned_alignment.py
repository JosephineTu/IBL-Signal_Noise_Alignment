# plot_timebinned_alignment.py
from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_SUFFIX = "_timebinned_alignment_summary.csv"
DETAILS_SUFFIX = "_timebinned_alignment_details.pkl"


def short_eid(eid, n=8):
    return str(eid)[:n]


def infer_details_path(summary_csv):
    """Infer the details PKL path from the summary CSV path."""
    summary_csv = Path(summary_csv)
    filename = summary_csv.name

    if not filename.endswith(SUMMARY_SUFFIX):
        raise ValueError(
            "Cannot infer details PKL because the summary filename does "
            f"not end with {SUMMARY_SUFFIX!r}: {summary_csv}"
        )

    details_filename = (
        filename[: -len(SUMMARY_SUFFIX)] + DETAILS_SUFFIX
    )

    return summary_csv.with_name(details_filename)


def load_details(details_pkl):
    details_pkl = Path(details_pkl)

    if not details_pkl.exists():
        raise FileNotFoundError(
            f"Details PKL does not exist: {details_pkl}"
        )

    with details_pkl.open("rb") as file:
        details = pickle.load(file)

    if not isinstance(details, Mapping):
        raise TypeError(
            "Expected the top level of the details PKL to be a dictionary"
        )

    return details


def iter_full_outputs(full_outputs):
    """Support either dictionary or list storage for time-bin outputs."""
    if isinstance(full_outputs, Mapping):
        yield from full_outputs.items()
    else:
        yield from enumerate(full_outputs)


def build_null_summary_dataframe(details):
    """
    Extract each time bin's null_cosine2 array and summarize it.

    Returns one row per eid/bin_idx.
    """
    rows = []

    for eid, detail in details.items():
        if not isinstance(detail, Mapping):
            continue

        full_outputs = detail.get("full_outputs")
        if full_outputs is None:
            continue

        time_windows = detail.get("time_windows")
        if time_windows is not None:
            time_windows = np.asarray(time_windows, dtype=float)

        for bin_key, output in iter_full_outputs(full_outputs):
            if not isinstance(output, Mapping):
                continue

            if "null_cosine2" not in output:
                continue

            try:
                bin_idx = int(bin_key)
            except (TypeError, ValueError):
                continue

            null_samples = np.asarray(
                output["null_cosine2"],
                dtype=float,
            ).reshape(-1)

            null_samples = null_samples[
                np.isfinite(null_samples)
            ]

            if null_samples.size == 0:
                continue

            row = {
                "eid": str(eid),
                "bin_idx": bin_idx,
                "null_cosine2_mean": float(
                    np.mean(null_samples)
                ),
                "null_cosine2_p05": float(
                    np.percentile(null_samples, 5)
                ),
                "null_cosine2_p50": float(
                    np.percentile(null_samples, 50)
                ),
                "null_cosine2_p95": float(
                    np.percentile(null_samples, 95)
                ),
                "null_cosine2_n": int(null_samples.size),
            }

            # These values are also available in full_outputs.
            if np.isscalar(output.get("null_pval")):
                row["null_pval_pkl"] = float(
                    output["null_pval"]
                )

            if np.isscalar(output.get("cosine2_top1")):
                row["cosine2_top1_pkl"] = float(
                    output["cosine2_top1"]
                )

            # Useful fallback if the CSV time column is missing.
            if (
                time_windows is not None
                and time_windows.ndim == 2
                and time_windows.shape[1] == 2
                and 0 <= bin_idx < len(time_windows)
            ):
                bin_start = float(time_windows[bin_idx, 0])
                bin_end = float(time_windows[bin_idx, 1])

                row["bin_start_pkl"] = bin_start
                row["bin_end_pkl"] = bin_end
                row["time_pkl"] = 0.5 * (
                    bin_start + bin_end
                )

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No null_cosine2 arrays were found in the details PKL"
        )

    null_df = pd.DataFrame(rows)

    duplicated = null_df.duplicated(
        subset=["eid", "bin_idx"],
        keep=False,
    )

    if duplicated.any():
        examples = null_df.loc[
            duplicated,
            ["eid", "bin_idx"],
        ].head()

        raise RuntimeError(
            "Duplicate eid/bin_idx entries found in details PKL:\n"
            f"{examples}"
        )

    return null_df


def load_and_merge_results(summary_csv, details_pkl):
    summary_csv = Path(summary_csv)
    details_pkl = Path(details_pkl)

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV does not exist: {summary_csv}"
        )

    df = pd.read_csv(summary_csv)

    if "status" in df.columns:
        df = df[
            df["status"].fillna("ok") == "ok"
        ].copy()

    required_csv_columns = [
        "eid",
        "bin_idx",
        "time",
        "cosine2_top1",
    ]

    missing = [
        column
        for column in required_csv_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in {summary_csv}: {missing}"
        )

    df["eid"] = df["eid"].astype(str)
    df["bin_idx"] = pd.to_numeric(
        df["bin_idx"],
        errors="coerce",
    )
    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce",
    )
    df["cosine2_top1"] = pd.to_numeric(
        df["cosine2_top1"],
        errors="coerce",
    )

    optional_numeric_columns = [
        "bin_start",
        "bin_end",
        "null_pval",
    ]

    for column in optional_numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "eid",
            "bin_idx",
            "time",
            "cosine2_top1",
        ]
    ).copy()

    df["bin_idx"] = df["bin_idx"].astype(int)

    details = load_details(details_pkl)
    null_df = build_null_summary_dataframe(details)

    merged = df.merge(
        null_df,
        on=["eid", "bin_idx"],
        how="left",
        validate="one_to_one",
    )

    missing_null = merged["null_cosine2_mean"].isna()

    if missing_null.any():
        examples = merged.loc[
            missing_null,
            ["eid", "bin_idx", "time"],
        ].head()

        raise RuntimeError(
            "Some CSV rows could not be matched to null distributions "
            "in the details PKL:\n"
            f"{examples}"
        )

    # Optional consistency check between CSV and PKL observed values.
    if "cosine2_top1_pkl" in merged.columns:
        difference = np.abs(
            merged["cosine2_top1"]
            - merged["cosine2_top1_pkl"]
        )

        inconsistent = (
            difference.notna()
            & (difference > 1e-10)
        )

        if inconsistent.any():
            print(
                "Warning: cosine2_top1 differs between CSV and PKL "
                f"for {int(inconsistent.sum())} rows."
            )

    return merged


def plot_one_eid_alignment(eid, data, out_dir, tag):
    data = data.sort_values("time")

    time = data["time"].to_numpy(dtype=float)
    observed = data["cosine2_top1"].to_numpy(dtype=float)
    null_mean = data[
        "null_cosine2_mean"
    ].to_numpy(dtype=float)
    null_p05 = data[
        "null_cosine2_p05"
    ].to_numpy(dtype=float)
    null_p95 = data[
        "null_cosine2_p95"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    ax.fill_between(
        time,
        null_p05,
        null_p95,
        color="tab:orange",
        alpha=0.22,
        linewidth=0,
        label="neuron-permutation null 5–95%",
    )

    ax.plot(
        time,
        null_mean,
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        label="neuron-permutation null mean",
    )

    ax.plot(
        time,
        observed,
        color="tab:blue",
        linewidth=2,
        marker="o",
        label="observed cos² top-1",
    )

    # Only show stimOn=0 when zero is within the analyzed range.
    if "bin_start" in data.columns and "bin_end" in data.columns:
        bin_start = pd.to_numeric(
            data["bin_start"],
            errors="coerce",
        ).to_numpy(dtype=float)

        bin_end = pd.to_numeric(
            data["bin_end"],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_start = bin_start[np.isfinite(bin_start)]
        finite_end = bin_end[np.isfinite(bin_end)]

        if finite_start.size and finite_end.size:
            x_min = float(np.min(finite_start))
            x_max = float(np.max(finite_end))

            if x_min <= 0 <= x_max:
                ax.axvline(
                    0,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.6,
                    label="stimOn",
                )

            ax.set_xlim(x_min, x_max)
    elif np.min(time) <= 0 <= np.max(time):
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label="stimOn",
        )

    ax.set_xlabel("time from stimOn (s)")
    ax.set_ylabel("cos² alignment")

    ax.set_title(
        "Signal alignment with top noise eigenmode\n"
        f"eid: {short_eid(eid)}"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    output_path = (
        out_dir
        / f"{short_eid(eid)}_{tag}_figure5_alignment_top1.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_one_summary(
    summary_csv,
    out_dir,
    details_pkl=None,
    tag=None,
):
    summary_csv = Path(summary_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if details_pkl is None:
        details_pkl = infer_details_path(summary_csv)
    else:
        details_pkl = Path(details_pkl)

    print(f"  summary CSV: {summary_csv}")
    print(f"  details PKL: {details_pkl}")

    df = load_and_merge_results(
        summary_csv=summary_csv,
        details_pkl=details_pkl,
    )

    if tag is None:
        tag = summary_csv.name.replace(
            SUMMARY_SUFFIX,
            "",
        )

    n_eids = 0

    for eid, data in df.groupby(
        "eid",
        sort=False,
    ):
        plot_one_eid_alignment(
            eid=eid,
            data=data,
            out_dir=out_dir,
            tag=tag,
        )
        n_eids += 1

    print(
        f"  saved {n_eids} eid plots to {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-csv",
        default=(
            "/home/xiaorantu/signal_noise_alignment/"
            "results/timebinned_alignment/"
            "t0p0to0p1_bin0p03_step0p01_k3"
            "_timebinned_alignment_summary.csv"
        ),
        help="Summary CSV for single-summary mode.",
    )

    parser.add_argument(
        "--details-pkl",
        default=None,
        help=(
            "Details PKL for single-summary mode. "
            "If omitted, it is inferred from the summary filename."
        ),
    )

    parser.add_argument(
        "--summary-glob",
        default=None,
        help=(
            "Glob for multiple summary CSVs, for example "
            "'results/timebinned_alignment/"
            "*_timebinned_alignment_summary.csv'. "
            "Each details PKL is inferred from its summary filename."
        ),
    )

    parser.add_argument(
        "--out-root",
        default="figures",
        help="Root directory for multi-summary mode.",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for single-summary mode.",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output filename tag.",
    )

    args = parser.parse_args()

    if args.summary_glob is not None:
        if args.details_pkl is not None:
            raise RuntimeError(
                "--details-pkl cannot be combined with --summary-glob; "
                "details paths are inferred for each summary."
            )

        summary_files = sorted(
            Path(".").glob(args.summary_glob)
        )

        if not summary_files:
            raise RuntimeError(
                f"No files matched --summary-glob "
                f"{args.summary_glob}"
            )

        out_root = Path(args.out_root)

        print(
            f"Found {len(summary_files)} summary CSVs."
        )

        for index, summary_csv in enumerate(
            summary_files,
            start=1,
        ):
            out_dir = (
                out_root / f"figure_5_{index}"
            )

            print(
                f"\n[{index}/{len(summary_files)}] "
                f"{summary_csv}"
            )

            plot_one_summary(
                summary_csv=summary_csv,
                details_pkl=None,
                out_dir=out_dir,
                tag=args.tag,
            )

    else:
        summary_csv = Path(args.summary_csv)

        if args.out_dir is None:
            out_dir = Path("figure_5")
        else:
            out_dir = Path(args.out_dir)

        plot_one_summary(
            summary_csv=summary_csv,
            details_pkl=args.details_pkl,
            out_dir=out_dir,
            tag=args.tag,
        )


if __name__ == "__main__":
    main()