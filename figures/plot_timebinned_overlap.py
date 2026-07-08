# plot_timebinned_overlap.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def short_eid(eid, n=8):
    return str(eid)[:n]


def plot_one_eid_overlap(eid, d, out_dir, tag):
    d = d.sort_values("time")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        d["time"],
        d["overlap_topk"],
        linewidth=2,
        marker="o",
        label="observed top-k overlap",
    )

    ax.plot(
        d["time"],
        d["expected_random_overlap"],
        linewidth=2,
        linestyle="--",
        marker="o",
        label="random expected k/N",
    )

    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("time from stimOn (s)")
    ax.set_ylabel("top-k overlap")
    ax.set_title(
        f"Figure 4: signal overlap with top-k noise subspace\n"
        f"eid: {short_eid(eid)}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        out_dir / f"{short_eid(eid)}_{tag}_figure4_overlap_topk.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-dir", default="figures/figure4")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"].copy()

    needed = [
        "eid",
        "time",
        "overlap_topk",
        "expected_random_overlap",
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {summary_csv}: {missing}")

    for c in ["time", "overlap_topk", "expected_random_overlap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["eid", "time", "overlap_topk", "expected_random_overlap"])

    if args.tag is None:
        tag = summary_csv.stem.replace("_timebinned_alignment_summary", "")
    else:
        tag = args.tag

    for eid, d in df.groupby("eid", sort=False):
        plot_one_eid_overlap(eid, d, out_dir, tag)

    print(f"Saved per-eid Figure 4 overlap plots to {out_dir}")


if __name__ == "__main__":
    main()