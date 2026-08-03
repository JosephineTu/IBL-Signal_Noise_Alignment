# plot_alignment_summary.py
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
    summary_csv = Path(summary_csv)

    if not summary_csv.name.endswith(SUMMARY_SUFFIX):
        raise ValueError(
            "Cannot infer details PKL from summary filename: "
            f"{summary_csv.name}"
        )

    prefix = summary_csv.name[: -len(SUMMARY_SUFFIX)]

    return summary_csv.with_name(
        prefix + DETAILS_SUFFIX
    )


def infer_bin_tag(summary_csv):
    summary_csv = Path(summary_csv)

    if summary_csv.name.endswith(SUMMARY_SUFFIX):
        return summary_csv.name[: -len(SUMMARY_SUFFIX)]

    return summary_csv.stem


def load_pickle(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Details PKL not found: {path}"
        )

    with path.open("rb") as file:
        details = pickle.load(file)

    if not isinstance(details, Mapping):
        raise TypeError(
            "Expected details PKL to contain a dictionary "
            "indexed by eid"
        )

    return details


def get_bin_output(full_outputs, bin_idx):
    """
    Retrieve one bin output while allowing integer or string keys.
    """
    if isinstance(full_outputs, Mapping):
        possible_keys = [
            bin_idx,
            int(bin_idx),
            str(int(bin_idx)),
        ]

        for key in possible_keys:
            if key in full_outputs:
                return full_outputs[key]

        return None

    try:
        return full_outputs[int(bin_idx)]
    except (IndexError, TypeError):
        return None


def extract_null_mean(details, eid, bin_idx):
    details_by_eid = {
        str(key): value
        for key, value in details.items()
    }

    eid = str(eid)

    if eid not in details_by_eid:
        raise KeyError(
            f"EID {eid} was not found in details PKL"
        )

    session_details = details_by_eid[eid]

    if not isinstance(session_details, Mapping):
        raise TypeError(
            f"Details for EID {eid} are not a dictionary"
        )

    full_outputs = session_details.get("full_outputs")

    if full_outputs is None:
        raise KeyError(
            f"EID {eid} does not contain full_outputs"
        )

    output = get_bin_output(
        full_outputs,
        bin_idx,
    )

    if output is None:
        raise KeyError(
            f"No output for EID {eid}, bin {bin_idx}"
        )

    if "null_cosine2" not in output:
        raise KeyError(
            f"No null_cosine2 for EID {eid}, bin {bin_idx}"
        )

    null_samples = np.asarray(
        output["null_cosine2"],
        dtype=float,
    ).reshape(-1)

    null_samples = null_samples[
        np.isfinite(null_samples)
    ]

    if null_samples.size == 0:
        raise ValueError(
            f"No finite null samples for EID {eid}, "
            f"bin {bin_idx}"
        )

    return float(np.mean(null_samples))


def load_alignment_improvement(
    summary_csv,
    details_pkl,
):
    summary_csv = Path(summary_csv)

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {summary_csv}"
        )

    df = pd.read_csv(summary_csv)

    if "status" in df.columns:
        df = df[
            df["status"].fillna("ok") == "ok"
        ].copy()

    required_columns = [
        "eid",
        "bin_idx",
        "time",
        "cosine2_top1",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in {summary_csv}: {missing}"
        )

    df["eid"] = df["eid"].astype(str)

    for column in [
        "bin_idx",
        "time",
        "cosine2_top1",
    ]:
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

    details = load_pickle(details_pkl)

    null_means = []
    valid_rows = []

    for row_index, row in df.iterrows():
        try:
            null_mean = extract_null_mean(
                details=details,
                eid=row["eid"],
                bin_idx=row["bin_idx"],
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            print(
                "Skipping "
                f"eid={row['eid']}, "
                f"bin_idx={row['bin_idx']}: "
                f"{error}"
            )
            continue

        valid_rows.append(row_index)
        null_means.append(null_mean)

    if not valid_rows:
        raise RuntimeError(
            "No CSV rows could be matched to valid "
            "null_cosine2 distributions"
        )

    df = df.loc[valid_rows].copy()
    df["null_cosine2_mean"] = null_means

    df["alignment_improvement"] = (
        df["cosine2_top1"]
        - df["null_cosine2_mean"]
    )

    return df


def plot_alignment_improvement(
    df,
    output_path,
    bin_tag,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    eids = list(
        df["eid"].drop_duplicates()
    )

    cmap = plt.get_cmap("tab20")

    for index, eid in enumerate(eids):
        session_df = (
            df[df["eid"] == eid]
            .sort_values("time")
        )

        time = session_df[
            "time"
        ].to_numpy(dtype=float)

        improvement = session_df[
            "alignment_improvement"
        ].to_numpy(dtype=float)

        color = cmap(index % cmap.N)

        ax.plot(
            time,
            improvement,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            alpha=0.85,
            color=color,
            label=short_eid(eid),
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_xlabel("time from stimOn (s)")

    ax.set_ylabel(
        "alignment improvement\n"
        "(observed cos² − permutation-null mean)"
    )

    ax.set_title(
        "Signal–noise alignment improvement by session\n"
        f"{bin_tag}"
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

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-csv",
        default=(
            "results/timebinned_alignment/"
            "t0p04to0p14_bin0p05_step0p01_k3"
            "_timebinned_alignment_summary.csv"
        ),
    )

    parser.add_argument(
        "--details-pkl",
        default=None,
        help=(
            "Matching details PKL. If omitted, it is "
            "inferred from the summary CSV filename."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. If omitted, a filename is "
            "generated under figures/alignment_summary."
        ),
    )

    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)

    if args.details_pkl is None:
        details_pkl = infer_details_path(
            summary_csv
        )
    else:
        details_pkl = Path(
            args.details_pkl
        )

    bin_tag = infer_bin_tag(
        summary_csv
    )

    if args.output is None:
        output_path = (
            Path("figures/alignment_summary")
            / f"{bin_tag}_alignment_improvement.png"
        )
    else:
        output_path = Path(args.output)

    print(f"Summary CSV: {summary_csv}")
    print(f"Details PKL: {details_pkl}")

    df = load_alignment_improvement(
        summary_csv=summary_csv,
        details_pkl=details_pkl,
    )

    print(
        f"Loaded {df['eid'].nunique()} sessions "
        f"and {len(df)} session-bin rows"
    )

    plot_alignment_improvement(
        df=df,
        output_path=output_path,
        bin_tag=bin_tag,
    )


if __name__ == "__main__":
    main()