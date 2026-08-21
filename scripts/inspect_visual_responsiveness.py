"""
inspect_visual_responsiveness.py

Next diagnostic after inspect_decoder_predictions.py confirmed, on real
data, that the epsilon~=0.24 "collapsed" VISp sessions really do produce
near-constant decoder predictions (test_r2~=0) -- not a curve-fitting
artifact. Left/right trial imbalance, n_trials, n_total_units and
trials/unit all came back null/non-significant as explanations for WHICH
sessions collapse. This checks a more basic possibility before anything
about contrast tuning or alignment: do the recorded units respond to
stimulus onset AT ALL (independent of contrast), or are they simply not
visually responsive in this window for these sessions?

For each session, compares per-unit firing rate in the SAME 0-100ms
post-stimOn window load_session_0_100ms already computes (X_stim --
reused directly from that function's return value, NOT recomputed) against
a pre-stimulus baseline window (default -100..0ms) computed via a second
spike-count pass over the SAME pid / SAME unit_ids returned by
load_session_0_100ms, so the two windows are guaranteed to compare the
exact same set of units rather than something that merely looks similar.

Does NOT re-implement region filtering, insertion selection, or trial
validity logic -- those all happen inside load_session_0_100ms itself
(this script calls it once and reuses pid/unit_ids/trials/X from its
return value). SpikeSortingLoader, get_spike_counts_in_bins and ensure_1d
are imported from run_0_100ms_decoders.py rather than re-derived, since
that is where load_session_0_100ms itself gets them from.

Per unit: paired Wilcoxon signed-rank test (stim rate vs. baseline rate,
paired by trial). Reports, per session:
  - frac_responsive_uncorrected  (fraction of units with p<0.05)
  - frac_responsive_bonferroni   (fraction of units with p < 0.05/n_units)
  - mean_stim_rate, mean_baseline_rate  (population means, Hz)
  - population_paired_p          (paired t-test on the trial-averaged,
                                   population-mean rate: stim vs baseline)

Writes:
  results/inspect_visual_responsiveness/<target_prefix>/
      {target_prefix}_visual_responsiveness.csv
      {target_prefix}_visual_responsiveness_grid.png
      details/{eid}_visual_responsiveness.pkl
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

from run_0_100ms_decoders import (  # noqa: E402
    SpikeSortingLoader,
    build_eids,
    ensure_1d,
    get_spike_counts_in_bins,
    load_session_0_100ms,
)

SEED = 0
BASELINE_START = -0.1  # seconds relative to stimOn -- 100ms pre-stimulus
BASELINE_END = 0.0
ALPHA = 0.05
ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"

DATA_COLOR = "#3F6FBF"       # muted blue -- non-significant units
SIG_COLOR = "#D9622B"        # warm orange -- significant units (p<0.05)
IDENTITY_COLOR = "#8A8A8A"   # neutral gray -- y=x reference (no modulation)
GRID_COLOR = "#D8D8D8"


def compute_visual_responsiveness(
    one, atlas, eid, target_prefix,
    baseline_start=BASELINE_START, baseline_end=BASELINE_END,
):
    loaded = load_session_0_100ms(one=one, atlas=atlas, eid=eid, target_prefix=target_prefix)
    X_stim = loaded["X"]                      # (n_trials, n_units), 0-100ms rate, already computed
    trials = loaded["trials"]
    unit_ids = np.asarray(loaded["unit_ids"])
    pid = loaded["pid"]

    stim_on = ensure_1d(trials["stimOn_times"], "stimOn_times")

    # second spike-count pass, SAME pid -> SAME spikes/clusters as the
    # 0-100ms window used, just a different interval.
    loader = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = loader.load_spike_sorting()

    baseline_intervals = np.column_stack(
        [stim_on + baseline_start, stim_on + baseline_end]
    )
    counts, cluster_ids = get_spike_counts_in_bins(
        spikes["times"], spikes["clusters"], baseline_intervals
    )
    cluster_ids = np.asarray(cluster_ids)

    if counts.shape == (len(cluster_ids), len(baseline_intervals)):
        counts = counts.T
    elif counts.shape == (len(baseline_intervals), len(cluster_ids)):
        pass
    else:
        raise RuntimeError(
            f"Unexpected spike-count shape {counts.shape} for eid={eid}"
        )

    # reindex to EXACTLY match unit_ids' order/subset from the stim-window
    # load -- get_spike_counts_in_bins is not guaranteed to return clusters
    # in the same order (or the same subset, if a cluster had zero spikes
    # in every baseline interval) as the earlier stim-window call.
    id_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
    n_trials = counts.shape[0]
    n_units = len(unit_ids)
    counts_aligned = np.zeros((n_trials, n_units), dtype=float)
    n_missing = 0
    for j, uid in enumerate(unit_ids):
        col = id_to_col.get(uid)
        if col is None:
            n_missing += 1
            continue
        counts_aligned[:, j] = counts[:, col]

    X_baseline = counts_aligned / (baseline_end - baseline_start)

    if X_stim.shape != X_baseline.shape:
        raise RuntimeError(
            f"X_stim shape {X_stim.shape} != X_baseline shape {X_baseline.shape} "
            f"for eid={eid} -- trial count mismatch between the two windows"
        )

    per_unit_p = np.full(n_units, np.nan)
    per_unit_mean_stim = X_stim.mean(axis=0)
    per_unit_mean_baseline = X_baseline.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(n_units):
            diff = X_stim[:, j] - X_baseline[:, j]
            if np.allclose(diff, 0):
                per_unit_p[j] = 1.0
                continue
            try:
                _, p = stats.wilcoxon(X_stim[:, j], X_baseline[:, j])
            except ValueError:
                # wilcoxon can still fail on degenerate inputs (e.g. all
                # ties after our allclose check due to float noise) --
                # fall back to a paired t-test rather than crash the session
                _, p = stats.ttest_rel(X_stim[:, j], X_baseline[:, j])
            per_unit_p[j] = p

    frac_responsive_uncorrected = float(np.mean(per_unit_p < ALPHA))
    bonferroni_alpha = ALPHA / max(n_units, 1)
    frac_responsive_bonferroni = float(np.mean(per_unit_p < bonferroni_alpha))

    pop_stim_per_trial = X_stim.mean(axis=1)
    pop_baseline_per_trial = X_baseline.mean(axis=1)
    if np.allclose(pop_stim_per_trial, pop_baseline_per_trial):
        pop_t, pop_p = 0.0, 1.0
    else:
        pop_t, pop_p = stats.ttest_rel(pop_stim_per_trial, pop_baseline_per_trial)

    return {
        "eid": eid,
        "pid": pid,
        "target_prefix": target_prefix,
        "n_total_units": n_units,
        "n_trials": n_trials,
        "n_missing_baseline_units": n_missing,
        "frac_responsive_uncorrected": frac_responsive_uncorrected,
        "frac_responsive_bonferroni": frac_responsive_bonferroni,
        "mean_stim_rate": float(X_stim.mean()),
        "mean_baseline_rate": float(X_baseline.mean()),
        "population_paired_t": float(pop_t),
        "population_paired_p": float(pop_p),
        "per_unit_p": per_unit_p,
        "per_unit_mean_stim": per_unit_mean_stim,
        "per_unit_mean_baseline": per_unit_mean_baseline,
    }


SUMMARY_FIELDS = [
    "eid", "pid", "target_prefix", "status", "error", "group_label",
    "epsilon_from_summary", "n_total_units", "n_trials",
    "n_missing_baseline_units", "frac_responsive_uncorrected",
    "frac_responsive_bonferroni", "mean_stim_rate", "mean_baseline_rate",
    "population_paired_t", "population_paired_p",
]


def write_csv(rows, path, fieldnames=SUMMARY_FIELDS):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def pick_sessions(target_prefix, n_per_group):
    summary_path = (
        REPO_ROOT / "results" / "subsampling_epsilon" / target_prefix
        / f"{target_prefix}_subsampling_epsilon_summary.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"no subsampling_epsilon summary found at {summary_path} -- "
            "run run_subsampling_epsilon.py first, or pass explicit --eid "
            "values instead of relying on auto-selection"
        )
    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and row["epsilon"] != "":
                rows.append(row)
    rows.sort(key=lambda r: float(r["epsilon"]))
    learning = rows[:n_per_group]
    collapsed = rows[-n_per_group:]
    picks = [(r["eid"], "learning", float(r["epsilon"])) for r in learning]
    picks += [(r["eid"], "collapsed", float(r["epsilon"])) for r in collapsed]
    return picks


def plot_grid(results, output_path, ncols=4):
    n = len(results)
    if n == 0:
        raise ValueError("no session results to plot")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False
    )
    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        stim = res["per_unit_mean_stim"]
        base = res["per_unit_mean_baseline"]
        sig = res["per_unit_p"] < ALPHA

        ax.scatter(
            base[~sig], stim[~sig], s=14, color=DATA_COLOR, alpha=0.7,
            linewidths=0, zorder=2, label="n.s. (p>=0.05)",
        )
        ax.scatter(
            base[sig], stim[sig], s=14, color=SIG_COLOR, alpha=0.85,
            linewidths=0, zorder=3, label="responsive (p<0.05)",
        )
        lo = 0.0
        hi = max(1.0, float(np.nanmax(np.concatenate([stim, base]))) * 1.05)
        ax.plot([lo, hi], [lo, hi], "--", color=IDENTITY_COLOR, linewidth=1,
                 zorder=1, label="y = x (no modulation)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{res['group_label']}: {res['eid'][:8]}\n"
            f"eps={res['epsilon_from_summary']:.3f}  "
            f"frac_resp={res['frac_responsive_uncorrected']:.2f}  "
            f"pop_p={res['population_paired_p']:.1e}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="best", frameon=False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.supxlabel("baseline firing rate (Hz, -100..0ms pre-stim)", fontsize=10)
    fig.supylabel("stimulus-evoked firing rate (Hz, 0-100ms post-stim)", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument(
        "--eid", action="append", default=None,
        help="inspect this specific eid (repeatable); omit to auto-pick "
             "via --n-per-group from an existing epsilon summary CSV",
    )
    parser.add_argument("--n-per-group", type=int, default=4)
    parser.add_argument("--baseline-start", type=float, default=BASELINE_START)
    parser.add_argument("--baseline-end", type=float, default=BASELINE_END)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else REPO_ROOT / "results" / "inspect_visual_responsiveness" / args.target_prefix
    )
    details_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    if args.eid:
        picks = [(eid, "manual", float("nan")) for eid in args.eid]
    else:
        picks = pick_sessions(args.target_prefix, args.n_per_group)

    print(f"target_prefix={args.target_prefix}")
    print(f"baseline_window=[{args.baseline_start}, {args.baseline_end}]")
    print(f"n_sessions_to_inspect={len(picks)}")
    for eid, label, eps in picks:
        print(f"  {label:10s} eid={eid} epsilon={eps}")

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=ONE_CACHE_DIR,
    )
    atlas = AllenAtlas()

    csv_rows = []
    plot_results = []
    for session_index, (eid, label, eps) in enumerate(picks, start=1):
        print(f"[{session_index}/{len(picks)}] eid={eid} ({label})")
        try:
            res = compute_visual_responsiveness(
                one, atlas, eid, args.target_prefix,
                baseline_start=args.baseline_start, baseline_end=args.baseline_end,
            )
            res["group_label"] = label
            res["epsilon_from_summary"] = eps
            details_path = details_dir / f"{eid}_visual_responsiveness.pkl"
            with open(details_path, "wb") as f:
                pickle.dump(res, f)
            plot_results.append(res)
            print(
                f"  ok: frac_responsive(uncorrected)={res['frac_responsive_uncorrected']:.3f}, "
                f"frac_responsive(bonferroni)={res['frac_responsive_bonferroni']:.3f}, "
                f"mean_stim={res['mean_stim_rate']:.2f}Hz, "
                f"mean_baseline={res['mean_baseline_rate']:.2f}Hz, "
                f"population_paired_p={res['population_paired_p']:.2e}"
            )
            row = {k: res.get(k, "") for k in SUMMARY_FIELDS}
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row = {k: "" for k in SUMMARY_FIELDS}
            row.update(
                eid=eid, pid="", target_prefix=args.target_prefix,
                status="failed", error=repr(exc), group_label=label,
                epsilon_from_summary=eps,
            )
            print(f"  failed: {row['error']}")
        csv_rows.append(row)

    summary_path = output_dir / f"{args.target_prefix}_visual_responsiveness.csv"
    write_csv(csv_rows, summary_path)
    print(f"summary={summary_path}")

    if plot_results:
        plot_path = output_dir / f"{args.target_prefix}_visual_responsiveness_grid.png"
        plot_grid(plot_results, plot_path)
        print(f"plot={plot_path}")
    else:
        print("no successful sessions -- skipping plot")


if __name__ == "__main__":
    main()