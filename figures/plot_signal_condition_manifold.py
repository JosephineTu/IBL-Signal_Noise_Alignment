#plot_signal_condition_manifold.py
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def short_eid(eid, n=6):
    return str(eid)[:n]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def format_cond(c):
    c = safe_float(c)
    if not np.isfinite(c):
        return str(c)
    return f"{c:.3g}"


def get_session_row(summary_df, eid):
    hit = summary_df[summary_df["eid"].astype(str) == str(eid)]
    if len(hit) == 0:
        return None
    return hit.iloc[0]


def session_text(row):
    if row is None:
        return ""

    lines = [
        f"eid: {short_eid(row['eid'])}",
        f"trials: {int(row['n_trials'])}",
        f"units: {int(row['n_units'])}",
        f"conditions: {int(row['n_conditions'])}",
    ]

    if "stim_pc1_var" in row:
        lines.append(f"PC1 var: {row['stim_pc1_var']:.3f}")
    if "stim_pc123_var" in row:
        lines.append(f"PC1-3 var: {row['stim_pc123_var']:.3f}")
    if "sig_overlap_stim_pc3" in row:
        lines.append(f"sig-PC1:3 overlap: {row['sig_overlap_stim_pc3']:.3f}")

    return "\n".join(lines)


def plot_one_eid_signal(eid, detail, row, out_dir):
    sm = detail.get("signal_manifold", {})

    conds = sm.get("condition_order", None)
    scores = sm.get("condition_mean_scores", None)
    evr = sm.get("explained_variance_ratio", None)

    if conds is None or scores is None:
        print(f"skip signal plot for {eid}: missing condition_order or condition_mean_scores")
        return

    conds = np.asarray(conds, float)
    scores = np.asarray(scores, float)

    if scores.ndim != 2 or scores.shape[1] < 1:
        print(f"skip signal plot for {eid}: bad scores shape {scores.shape}")
        return

    x = scores[:, 0]
    y = scores[:, 1] if scores.shape[1] > 1 else np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.scatter(x, y, s=60)

    for i, c in enumerate(conds):
        ax.text(
            x[i],
            y[i],
            format_cond(c),
            fontsize=9,
            ha="left",
            va="bottom",
        )

    cond_to_i = {float(c): i for i, c in enumerate(conds)}
    positive_conds = sorted([float(c) for c in conds if c > 0])

    scale = max(np.ptp(x), np.ptp(y), 1e-6)

    for c in positive_conds:
        c_neg = -c
        c_pos = c

        if c_neg not in cond_to_i or c_pos not in cond_to_i:
            continue

        i0 = cond_to_i[c_neg]
        i1 = cond_to_i[c_pos]

        x0, y0 = x[i0], y[i0]
        dx, dy = x[i1] - x0, y[i1] - y0

        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            length_includes_head=True,
            head_width=0.01 * scale,
            head_length=0.02 * scale,
            linewidth=1.3,
            alpha=0.75,
        )

        ax.text(
            x0 + 0.55 * dx,
            y0 + 0.55 * dy,
            f"{format_cond(c_neg)}→{format_cond(c_pos)}",
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.axhline(0, linewidth=1, alpha=0.5)
    ax.axvline(0, linewidth=1, alpha=0.5)

    if evr is not None and len(evr) > 0:
        xlabel = f"condition mean PC1 ({float(evr[0]):.1%} var)"
    else:
        xlabel = "condition mean PC1"

    if evr is not None and len(evr) > 1:
        ylabel = f"condition mean PC2 ({float(evr[1]):.1%} var)"
    else:
        ylabel = "condition mean PC2"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"{short_eid(eid)} signal condition manifold\n"
        "points = signed contrast conditions; arrows = -c to +c"
    )

    ax.text(
        1.03,
        0.98,
        session_text(row),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{short_eid(eid)}_signal_condition_manifold.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="results/condition_geometry/condition_geometry_summary.csv",
    )
    parser.add_argument(
        "--details-pkl",
        default="results/condition_geometry/condition_geometry_details.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/figure_1",
    )
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)

    with open(args.details_pkl, "rb") as f:
        details = pickle.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for eid, detail in details.items():
        row = get_session_row(summary_df, eid)
        plot_one_eid_signal(eid, detail, row, out_dir)

    print(f"Saved signal condition manifold figures to {out_dir}")


if __name__ == "__main__":
    main()