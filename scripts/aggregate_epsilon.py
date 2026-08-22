"""
aggregate_epsilon_by_region.py

Rolls per-session epsilon (the information-limiting/alignment floor from
run_subsampling_epsilon.py) up into a per-brain-region estimate, per the
project notes' "Per-session eps (intercept) inference" section: don't
average epsilon unweighted, use inverse-variance weighting from
se_intercept (random-effects-style meta-analysis), and correct for
multiple comparisons across sessions rather than counting raw p<0.05.

IMPORTANT gotcha this script guards against (found while diagnosing the
bimodal epsilon pattern, via inspect_decoder_predictions.py /
inspect_visual_responsiveness.py): sessions whose decoder has fully
collapsed to predicting the balanced mean (epsilon sitting right at the
trivial "predict-the-mean" ceiling for the 9 IBL signed-contrast levels,
~0.2404) tend to have a VERY SMALL se_intercept -- the flat MSE(N) curve
is highly reproducible across repeats even though what it's reproducibly
estimating is "no signal", not a real information-limiting floor. Naive
inverse-variance weighting would let these misleadingly-precise,
uninformative sessions dominate the region-level average and bias it
toward the trivial ceiling. This script flags such sessions
(is_collapsed = epsilon >= --collapse-frac * trivial_mse, default
frac=0.95) and reports the meta-analysis BOTH including and excluding
them, rather than silently picking one.

Does NOT re-implement the WLS fit itself -- reads epsilon/se_intercept/
p_value straight from run_subsampling_epsilon.py's summary CSV(s). The
only new computation here is the DerSimonian-Laird random-effects
meta-analysis and the multiple-comparison correction, neither of which
exists elsewhere in the repo yet (flagged in the notes as "not yet
implemented").

Region discovery: pass --target-prefix one or more times, or omit it to
auto-discover every results/subsampling_epsilon/<region>/ directory that
has a summary CSV.

Writes:
  results/epsilon_meta_analysis/
      epsilon_meta_analysis_by_region.csv     (one row per region x subset)
      epsilon_meta_analysis_sessions.csv      (per-session, with is_collapsed flag)
      epsilon_meta_analysis_forest.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)

# IBL standard signed-contrast levels (fraction, both signs); the balanced
# ("predict-the-mean", equal weight per condition) trivial MSE ceiling for
# this level set is ~0.24045138888888887 -- computed here rather than
# hardcoded so the threshold is auditable/adjustable if the level set ever
# changes.
IBL_CONTRASTS = [0.0, 0.0625, 0.125, 0.25, 1.0]
IBL_SIGNED_CONTRASTS = sorted(set(IBL_CONTRASTS + [-c for c in IBL_CONTRASTS]))


def trivial_balanced_mse(levels=IBL_SIGNED_CONTRASTS):
    vals = np.array(levels, dtype=float)
    weights = np.ones(len(vals)) / len(vals)
    mean_pred = np.sum(weights * vals)
    return float(np.sum(weights * (vals - mean_pred) ** 2))


TRIVIAL_MSE = trivial_balanced_mse()

DATA_COLOR = "#3F6FBF"      # muted blue -- non-collapsed sessions
COLLAPSED_COLOR = "#B0B0B0"  # neutral gray -- collapsed sessions (down-weighted)
RE_COLOR = "#D9622B"        # warm orange -- region-level random-effects estimate
GRID_COLOR = "#D8D8D8"


def meta_analyze(epsilons, ses):
    """DerSimonian-Laird random-effects inverse-variance meta-analysis."""
    epsilons = np.asarray(epsilons, dtype=float)
    ses = np.asarray(ses, dtype=float)
    k = len(epsilons)
    if k == 0:
        return None

    w_fe = 1.0 / ses ** 2
    theta_fe = float(np.sum(w_fe * epsilons) / np.sum(w_fe))
    se_fe = float(np.sqrt(1.0 / np.sum(w_fe)))

    if k == 1:
        return {
            "k": 1, "theta_fe": theta_fe, "se_fe": se_fe,
            "theta_re": theta_fe, "se_re": se_fe,
            "tau2": 0.0, "Q": 0.0, "df": 0, "I2": 0.0,
        }

    Q = float(np.sum(w_fe * (epsilons - theta_fe) ** 2))
    df = k - 1
    C = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    w_re = 1.0 / (ses ** 2 + tau2)
    theta_re = float(np.sum(w_re * epsilons) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0

    return {
        "k": k, "theta_fe": theta_fe, "se_fe": se_fe,
        "theta_re": theta_re, "se_re": se_re,
        "tau2": tau2, "Q": Q, "df": df, "I2": I2,
    }


def bonferroni_reject(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    return pvals < (alpha / n)


def fdr_bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up FDR control, no statsmodels dependency."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    reject = np.zeros(n, dtype=bool)
    if np.any(below):
        max_i = int(np.max(np.where(below)[0]))
        reject[order[: max_i + 1]] = True
    return reject


def discover_regions(explicit_prefixes):
    root = REPO_ROOT / "results" / "subsampling_epsilon"
    if explicit_prefixes:
        return list(explicit_prefixes)
    if not root.is_dir():
        return []
    regions = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / f"{d.name}_subsampling_epsilon_summary.csv"
        if summary_path.is_file():
            regions.append(d.name)
    return regions


def load_region_sessions(target_prefix):
    summary_path = (
        REPO_ROOT / "results" / "subsampling_epsilon" / target_prefix
        / f"{target_prefix}_subsampling_epsilon_summary.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"no summary CSV found at {summary_path}")
    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row["epsilon"] == "":
                continue
            rows.append({
                "eid": row["eid"],
                "epsilon": float(row["epsilon"]),
                "se_intercept": float(row["se_intercept"]),
                "p_value": float(row["p_value"]),
                "slope": float(row["slope"]),
            })
    return rows


def summarize_region(target_prefix, sessions, collapse_frac):
    collapse_threshold = collapse_frac * TRIVIAL_MSE
    for s in sessions:
        s["is_collapsed"] = s["epsilon"] >= collapse_threshold

    out_rows = []
    for label, subset in [
        ("all", sessions),
        ("excluding_collapsed", [s for s in sessions if not s["is_collapsed"]]),
    ]:
        if not subset:
            out_rows.append({
                "target_prefix": target_prefix, "subset": label, "k": 0,
                "theta_re": "", "se_re": "", "ci95_lo": "", "ci95_hi": "",
                "theta_fe": "", "se_fe": "", "tau2": "", "I2": "", "Q": "", "df": "",
                "n_sig_raw": "", "n_sig_bonferroni": "", "n_sig_fdr_bh": "",
            })
            continue
        eps = [s["epsilon"] for s in subset]
        ses = [s["se_intercept"] for s in subset]
        pvals = [s["p_value"] for s in subset]
        ma = meta_analyze(eps, ses)
        n_sig_raw = int(np.sum(np.asarray(pvals) < 0.05))
        n_sig_bonf = int(np.sum(bonferroni_reject(pvals)))
        n_sig_fdr = int(np.sum(fdr_bh_reject(pvals)))
        ci_lo = ma["theta_re"] - 1.96 * ma["se_re"]
        ci_hi = ma["theta_re"] + 1.96 * ma["se_re"]
        out_rows.append({
            "target_prefix": target_prefix, "subset": label, "k": ma["k"],
            "theta_re": ma["theta_re"], "se_re": ma["se_re"],
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "theta_fe": ma["theta_fe"], "se_fe": ma["se_fe"],
            "tau2": ma["tau2"], "I2": ma["I2"], "Q": ma["Q"], "df": ma["df"],
            "n_sig_raw": n_sig_raw, "n_sig_bonferroni": n_sig_bonf,
            "n_sig_fdr_bh": n_sig_fdr,
        })
    return out_rows


REGION_FIELDS = [
    "target_prefix", "subset", "k", "theta_re", "se_re", "ci95_lo", "ci95_hi",
    "theta_fe", "se_fe", "tau2", "I2", "Q", "df",
    "n_sig_raw", "n_sig_bonferroni", "n_sig_fdr_bh",
]
SESSION_FIELDS = [
    "target_prefix", "eid", "epsilon", "se_intercept", "p_value", "slope",
    "is_collapsed",
]


def write_csv(rows, path, fieldnames):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def plot_forest(region_sessions, region_summaries, output_path):
    regions = list(region_sessions.keys())
    n = len(regions)
    if n == 0:
        raise ValueError("no regions to plot")
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    axes = axes[0]

    for ax, region in zip(axes, regions):
        sessions = sorted(region_sessions[region], key=lambda s: s["epsilon"])
        y = np.arange(len(sessions))
        eps = np.array([s["epsilon"] for s in sessions])
        se = np.array([s["se_intercept"] for s in sessions])
        collapsed = np.array([s["is_collapsed"] for s in sessions])

        colors = np.where(collapsed, COLLAPSED_COLOR, DATA_COLOR)
        ax.errorbar(
            eps, y, xerr=1.96 * se, fmt="o", ms=3, ecolor=colors, elinewidth=1,
            capsize=0, zorder=2,
        )
        ax.scatter(eps, y, s=14, c=colors, zorder=3)

        ax.axvline(TRIVIAL_MSE, color="black", linestyle=":", linewidth=1,
                    alpha=0.5, zorder=1, label="trivial (predict-mean) MSE")

        summ = {r["subset"]: r for r in region_summaries[region]}
        if summ.get("all", {}).get("theta_re") != "":
            ax.axvline(summ["all"]["theta_re"], color=RE_COLOR, linewidth=1.5,
                        zorder=4, label="region RE mean (all sessions)")
        if summ.get("excluding_collapsed", {}).get("theta_re") != "":
            ax.axvline(summ["excluding_collapsed"]["theta_re"], color=RE_COLOR,
                        linewidth=1.5, linestyle="--", zorder=4,
                        label="region RE mean (excl. collapsed)")

        ax.set_yticks([])
        ax.set_xlabel("epsilon (95% CI)")
        ax.set_title(region, fontsize=10)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(fontsize=6, loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", action="append", default=None,
                         help="repeatable; omit to auto-discover all regions "
                              "under results/subsampling_epsilon/")
    parser.add_argument("--collapse-frac", type=float, default=0.95,
                         help="a session is flagged collapsed if "
                              "epsilon >= collapse_frac * trivial_mse "
                              f"(trivial_mse={TRIVIAL_MSE:.6f})")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else REPO_ROOT / "results" / "epsilon_meta_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = discover_regions(args.target_prefix)
    print(f"trivial_mse={TRIVIAL_MSE:.6f}  collapse_threshold={args.collapse_frac * TRIVIAL_MSE:.6f}")
    print(f"regions={regions}")
    if not regions:
        print("no regions found -- nothing to do")
        return

    region_rows = []
    session_rows = []
    region_sessions = {}
    region_summaries = {}

    for target_prefix in regions:
        sessions = load_region_sessions(target_prefix)
        print(f"[{target_prefix}] n_sessions_ok={len(sessions)}")
        if not sessions:
            continue
        summaries = summarize_region(target_prefix, sessions, args.collapse_frac)
        region_sessions[target_prefix] = sessions
        region_summaries[target_prefix] = summaries
        region_rows.extend(summaries)
        for s in sessions:
            row = {k: s[k] for k in ["eid", "epsilon", "se_intercept", "p_value", "slope", "is_collapsed"]}
            row["target_prefix"] = target_prefix
            session_rows.append(row)

        for summ in summaries:
            if summ["k"] == 0:
                print(f"  [{summ['subset']}] k=0")
                continue
            print(
                f"  [{summ['subset']}] k={summ['k']}  "
                f"theta_re={summ['theta_re']:.4f} (95% CI [{summ['ci95_lo']:.4f}, {summ['ci95_hi']:.4f}])  "
                f"tau2={summ['tau2']:.6f}  I2={summ['I2']:.1f}%  "
                f"n_sig(raw/bonf/fdr)={summ['n_sig_raw']}/{summ['n_sig_bonferroni']}/{summ['n_sig_fdr_bh']}"
            )

    region_path = output_dir / "epsilon_meta_analysis_by_region.csv"
    session_path = output_dir / "epsilon_meta_analysis_sessions.csv"
    write_csv(region_rows, region_path, REGION_FIELDS)
    write_csv(session_rows, session_path, SESSION_FIELDS)
    print(f"region_summary={region_path}")
    print(f"session_detail={session_path}")

    if region_sessions:
        plot_path = output_dir / "epsilon_meta_analysis_forest.png"
        plot_forest(region_sessions, region_summaries, plot_path)
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()