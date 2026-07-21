from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def short_eid(eid, n=8):
    return str(eid)[:n]


def plot_one_eid_alignment(eid, d, out_dir, tag):
    d = d.sort_values("time")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        d["time"],
        d["cosine2_top1"],
        linewidth=2,
        marker="o",
        label="observed cos² top-1",
    )

    ax.plot(
        d["time"],
        d["expected_random_cosine2"],
        linewidth=2,
        linestyle="--",
        marker="o",
        label="random expected 1/N",
    )

    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("time from stimOn (s)")
    ax.set_ylabel("cos² alignment")
    ax.set_title(
        f"Figure 5: signal alignment with top noise eigenmode\n"
        f"eid: {short_eid(eid)}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / f"{short_eid(eid)}_{tag}_figure5_alignment_top1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_one_summary(summary_csv, out_dir, tag=None):
    summary_csv = Path(summary_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()

    needed = [
        "eid",
        "time",
        "cosine2_top1",
        "expected_random_cosine2",
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {summary_csv}: {missing}")

    for c in ["time", "cosine2_top1", "expected_random_cosine2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "eid",
            "time",
            "cosine2_top1",
            "expected_random_cosine2",
        ]
    )

    if tag is None:
        tag = summary_csv.stem.replace("_timebinned_alignment_summary", "")

    n_eids = 0
    for eid, d in df.groupby("eid", sort=False):
        plot_one_eid_alignment(eid, d, out_dir, tag)
        n_eids += 1

    print(f"  saved {n_eids} eid plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-csv",
        default="/home/xiaorantu/signal_noise_alignment/results/timebinned_alignment/t0p04to0p14_bin0p05_step0p01_k3_timebinned_alignment_summary.csv",
        help="Single summary CSV. Kept for backward compatibility.",
    )

    parser.add_argument(
        "--summary-glob",
        default=None,
        help='Glob for multiple summary CSVs, e.g. "results/*timebinned_alignment_summary.csv"',
    )

    parser.add_argument(
        "--out-root",
        default="figures",
        help="Root directory. Multiple summaries will be saved as figure_5_1, figure_5_2, ...",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for single-summary mode.",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag. In multi-summary mode, default tag comes from each CSV stem.",
    )

    args = parser.parse_args()

    if args.summary_glob is not None:
        summary_files = sorted(Path(".").glob(args.summary_glob))

        if len(summary_files) == 0:
            raise RuntimeError(f"No files matched --summary-glob {args.summary_glob}")

        out_root = Path(args.out_root)

        print(f"Found {len(summary_files)} summary CSVs.")

        for i, summary_csv in enumerate(summary_files, start=1):
            out_dir = out_root / f"figure_5_{i}"

            print(f"\n[{i}/{len(summary_files)}] {summary_csv}")
            print(f"  out_dir = {out_dir}")

            plot_one_summary(
                summary_csv=summary_csv,
                out_dir=out_dir,
                tag=args.tag,
            )

    elif args.summary_csv is not None:
        summary_csv = Path(args.summary_csv)

        if args.out_dir is None:
            out_dir = Path("figure_5")
        else:
            out_dir = Path(args.out_dir)

        plot_one_summary(
            summary_csv=summary_csv,
            out_dir=out_dir,
            tag=args.tag,
        )

    else:
        raise RuntimeError("Provide either --summary-csv or --summary-glob.")


if __name__ == "__main__":
    main()