# plot_condition_noise_random_baseline.py
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


def session_text(row, noise):
    lines = []

    if row is not None:
        lines.extend([
            f"eid: {short_eid(row['eid'])}",
        ])

        for col, label in [
            ("n_trials", "trials"),
            ("n_units", "units"),
            ("n_conditions", "conditions"),
        ]:
            if col in row and pd.notna(row[col]):
                lines.append(f"{label}: {int(row[col])}")

    obs_mean = noise.get("mean_similarity", np.nan)
    obs_min = noise.get("min_similarity", np.nan)
    rand = noise.get("random_similarity", {})

    if np.isfinite(obs_mean):
        lines.append(f"observed mean: {obs_mean:.3f}")
    if np.isfinite(obs_min):
        lines.append(f"observed min: {obs_min:.3f}")

    if rand:
        if "mean" in rand:
            lines.append(f"random mean: {rand['mean']:.3f}")
        if "p95" in rand:
            lines.append(f"random p95: {rand['p95']:.3f}")

        samples = np.asarray(rand.get("samples", []), float)
        if samples.size > 0 and np.isfinite(obs_mean):
            z = (obs_mean - np.nanmean(samples)) / (np.nanstd(samples) + 1e-12)
            p_upper = np.nanmean(samples >= obs_mean)
            lines.append(f"z: {z:.2f}")
            lines.append(f"p upper: {p_upper:.3f}")

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


def select_eids(details, n_examples=None, seed=0, requested_eids=None):
    all_eids = list(details.keys())

    if requested_eids:
        requested = [str(e) for e in requested_eids]
        return [eid for eid in all_eids if str(eid) in requested]

    valid = []
    for eid in all_eids:
        noise = details[eid].get("noise_subspace", {})
        rand = noise.get("random_similarity", {})
        samples = np.asarray(rand.get("samples", []), float)

        if noise.get("noise_subspace_similarities", {}) and samples.size > 0:
            valid.append(eid)

    # Default: plot all valid eids
    if n_examples is None:
        return valid

    # If requested number >= valid count, plot all valid eids
    if len(valid) <= n_examples:
        return valid

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(valid), size=n_examples, replace=False)
    return [valid[i] for i in idx]

def plot_one_eid_noise_random(eid, detail, row, out_dir):
    noise = detail.get("noise_subspace", {})
    pairdict = noise.get("noise_subspace_similarities", {})
    rand = noise.get("random_similarity", {})

    if not pairdict:
        print(f"skip {eid}: no noise_subspace_similarities")
        return

    samples = np.asarray(rand.get("samples", []), float)
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        print(f"skip {eid}: no random_similarity samples")
        return

    obs_mean = safe_float(noise.get("mean_similarity", np.nan))
    obs_min = safe_float(noise.get("min_similarity", np.nan))
    rand_mean = safe_float(rand.get("mean", np.nan))
    rand_p05 = safe_float(rand.get("p05", np.nan))
    rand_p95 = safe_float(rand.get("p95", np.nan))

    conds, mat = make_noise_similarity_matrix(pairdict)
    labels = [format_cond(c) for c in conds]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    ax0, ax1 = axes

    # Left: observed condition-pair similarity matrix
    im = ax0.imshow(mat, vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax0)
    cbar.set_label("top-k noise subspace similarity")

    ax0.set_xticks(np.arange(len(conds)))
    ax0.set_yticks(np.arange(len(conds)))
    ax0.set_xticklabels(labels, rotation=45, ha="right")
    ax0.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax0.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax0.set_xlabel("signed contrast condition")
    ax0.set_ylabel("signed contrast condition")
    ax0.set_title("Observed condition-wise\nnoise subspace similarity")

    # Right: random pseudo-condition baseline
    ax1.hist(samples, bins=25, alpha=0.75)

    if np.isfinite(obs_mean):
        ax1.axvline(obs_mean, linewidth=2, label=f"observed mean = {obs_mean:.3f}")
    if np.isfinite(rand_mean):
        ax1.axvline(rand_mean, linestyle="--", linewidth=2, label=f"random mean = {rand_mean:.3f}")
    if np.isfinite(rand_p05) and np.isfinite(rand_p95):
        ax1.axvspan(rand_p05, rand_p95, alpha=0.2, label="random 5–95%")

    ax1.set_xlabel("mean condition-pair subspace similarity")
    ax1.set_ylabel("shuffle count")
    ax1.set_title("Pseudo-condition random baseline")
    ax1.legend(fontsize=8)

    ax1.text(
        1.02,
        0.98,
        session_text(row, noise),
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle(
        f"{short_eid(eid)} contrast-condition noise geometry vs random pseudo-conditions",
        y=1.03,
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / f"{short_eid(eid)}_condition_noise_random_baseline.png",
        dpi=300,
        bbox_inches="tight",
    )
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
        default="figures/figure_3",
    )
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eids", nargs="*", default=None)

    args = parser.parse_args()

    summary_df = pd.read_csv(args.summary_csv)

    with open(args.details_pkl, "rb") as f:
        details = pickle.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_eids = select_eids(
        details,
        n_examples=args.n_examples,
        seed=args.seed,
        requested_eids=args.eids,
    )

    print("Selected eids:")
    for eid in selected_eids:
        print(f"  {eid}")

    for eid in selected_eids:
        detail = details[eid]
        row = get_session_row(summary_df, eid)
        plot_one_eid_noise_random(eid, detail, row, out_dir)

    print(f"Saved Figure 3 panels to {out_dir}")


if __name__ == "__main__":
    main()