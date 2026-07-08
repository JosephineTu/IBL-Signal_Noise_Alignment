# plot_noise_condition_similarity.py
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

    if "noise_mean_condition_similarity" in row:
        lines.append(f"mean noise sim: {row['noise_mean_condition_similarity']:.3f}")
    if "noise_min_condition_similarity" in row:
        lines.append(f"min noise sim: {row['noise_min_condition_similarity']:.3f}")
    if "random_noise_similarity_mean" in row:
        lines.append(f"random mean: {row['random_noise_similarity_mean']:.3f}")
    if "random_noise_similarity_p95" in row:
        lines.append(f"random p95: {row['random_noise_similarity_p95']:.3f}")

    return "\n".join(lines)


def make_noise_similarity_matrix(pairdict):
    conds = sorted(
        set(
            [safe_float(k[0]) for k in pairdict.keys()]
            + [safe_float(k[1]) for k in pairdict.keys()]
        )
    )

    mat = np.eye(len(conds), dtype=float)
    cond_to_i = {c: i for i, c in enumerate(conds)}

    for key, val in pairdict.items():
        a = safe_float(key[0])
        b = safe_float(key[1])

        if not np.isfinite(a) or not np.isfinite(b):
            continue

        if a not in cond_to_i or b not in cond_to_i:
            continue

        i = cond_to_i[a]
        j = cond_to_i[b]

        mat[i, j] = float(val)
        mat[j, i] = float(val)

    return conds, mat


def plot_one_eid_noise(eid, detail, row, out_dir):
    noise = detail.get("noise_subspace", {})
    pairdict = noise.get("noise_subspace_similarities", {})

    if not pairdict:
        print(f"skip noise plot for {eid}: no noise_subspace_similarities")
        return

    conds, mat = make_noise_similarity_matrix(pairdict)
    labels = [format_cond(c) for c in conds]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("top-k noise subspace similarity")

    ax.set_xticks(np.arange(len(conds)))
    ax.set_yticks(np.arange(len(conds)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("signed contrast condition: contrastLeft - contrastRight")
    ax.set_ylabel("signed contrast condition: contrastLeft - contrastRight")

    ax.set_title(
        f"{short_eid(eid)} noise subspace similarity across conditions\n"
        "entry = similarity between two condition-specific top-k noise subspaces"
    )

    ax.text(
        1.05,
        0.98,
        session_text(row),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{short_eid(eid)}_noise_condition_similarity.png", dpi=300, bbox_inches="tight")
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
        default="figures/figure_2",
    )
    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)

    with open(args.details_pkl, "rb") as f:
        details = pickle.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for eid, detail in details.items():
        row = get_session_row(summary_df, eid)
        plot_one_eid_noise(eid, detail, row, out_dir)

    print(f"Saved noise condition similarity figures to {out_dir}")


if __name__ == "__main__":
    main()