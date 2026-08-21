"""
inspect_decoder_predictions.py

Direct diagnostic for the "does the decoder actually collapse to a
near-constant prediction" question. Everything so far (left/right trial
imbalance, n_trials, n_total_units, trials/unit) came back null or
non-significant against epsilon/slope -- so instead of hunting for more
summary-level covariates, this looks at the decoder's actual held-out
predictions directly, for a handful of chosen sessions.

For each chosen session, refits ONE Ridge decoder at the FULL recorded
neuron population (not neuron-subsampled -- the most-informative case)
on a single 50/50 train/test split, and records:
  - test_r2                    (already computed inside ridge_regression,
                                 just never kept by the subsampling-curve
                                 pipeline -- near 0 or negative means the
                                 decoder does no better on held-out data
                                 than predicting the balanced mean)
  - pred_std vs true_std       (if predictions barely vary while the true
                                 labels vary a lot, that IS "collapse")
  - pred_true_corr             (correlation between prediction and truth)
  - pred_mean_by_condition     (mean prediction at each of the 9 signed
                                 contrast levels -- flat across levels is
                                 the direct signature of collapse; a
                                 monotonic trend is the signature of a
                                 real decodable signal)

Does NOT re-implement decoder fitting or session loading -- reuses
decoder.make_train_test_sets / decoder.ridge_regression exactly as
run_subsampling_epsilon.py does (same RidgeCV alpha grid), and build_eids
/ load_session_0_100ms / make_contrast_masks from run_0_100ms_decoders.py.
decoder.ridge_regression itself does not return the raw predictions, only
test_r2 / coefficients / the standardized design matrices (via
return_extras=True) -- since the passed-in `model` object is fit in
place, y_test_pred is recovered afterwards with one extra
model.predict(X_test_std) call, which reproduces exactly what
ridge_regression computed internally (deterministic, not a re-fit).

Session selection: pass --eid one or more times to inspect specific
sessions, or omit it and pass --n-per-group (default 4) to auto-pick
that many sessions with the highest epsilon ("collapsed"-looking) and
that many with the lowest epsilon ("learning") from an existing
subsampling_epsilon summary CSV for the same target_prefix.

Writes:
  results/inspect_predictions/<target_prefix>/
      {target_prefix}_prediction_diagnostics.csv
      {target_prefix}_prediction_diagnostics_grid.png
      details/{eid}_prediction_diagnostics.pkl   (full y_test / y_test_pred)
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from sklearn.linear_model import RidgeCV

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

from src import decoder  # noqa: E402
from run_0_100ms_decoders import (  # noqa: E402
    build_eids,
    load_session_0_100ms,
    make_contrast_masks,
)

SEED = 0
RIDGE_ALPHAS = np.logspace(-2, 4, 25)  # same grid as run_subsampling_epsilon.py
ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"

DATA_COLOR = "#3F6FBF"     # muted blue -- individual test-trial predictions
MEAN_COLOR = "#D9622B"     # warm orange -- per-condition mean prediction
IDENTITY_COLOR = "#8A8A8A"  # neutral gray -- y=x reference (perfect decoding)
GRID_COLOR = "#D8D8D8"


def inspect_one_session(one, atlas, eid, target_prefix, seed=SEED):
    loaded = load_session_0_100ms(
        one=one, atlas=atlas, eid=eid, target_prefix=target_prefix
    )
    X = loaded["X"]
    masks = make_contrast_masks(loaded["signed_contrast"])

    X_train, y_train, X_test, y_test = decoder.make_train_test_sets(
        masks, X, seed=seed
    )
    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
    (
        balanced_test_mse,
        test_r2,
        w,
        X_train_std,
        X_test_std,
        scaler,
    ) = decoder.ridge_regression(
        X_train, y_train, X_test, y_test, model, return_extras=True
    )
    # ridge_regression computes y_test_pred internally but only returns
    # test_r2/w/the standardized matrices -- model.predict reproduces the
    # exact same predictions since `model` was fit in place above.
    y_test_pred = model.predict(X_test_std)

    contrasts = np.unique(y_test)
    pred_mean_by_condition = {
        float(c): float(y_test_pred[y_test == c].mean()) for c in contrasts
    }
    pred_std_by_condition = {
        float(c): float(y_test_pred[y_test == c].std()) for c in contrasts
    }

    pred_std = float(np.std(y_test_pred))
    true_std = float(np.std(y_test))
    pred_true_corr = (
        float(np.corrcoef(y_test_pred, y_test)[0, 1]) if pred_std > 0 else 0.0
    )

    return {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "seed": seed,
        "n_total_units": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "balanced_test_mse": float(balanced_test_mse),
        "test_r2": float(test_r2),
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_true_corr": pred_true_corr,
        "pred_mean_by_condition": pred_mean_by_condition,
        "pred_std_by_condition": pred_std_by_condition,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


SUMMARY_FIELDS = [
    "eid",
    "pid",
    "target_prefix",
    "status",
    "error",
    "group_label",
    "epsilon_from_summary",
    "n_total_units",
    "n_train",
    "n_test",
    "balanced_test_mse",
    "test_r2",
    "pred_std",
    "true_std",
    "pred_true_corr",
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
        REPO_ROOT
        / "results"
        / "subsampling_epsilon"
        / target_prefix
        / f"{target_prefix}_subsampling_epsilon_summary.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"no subsampling_epsilon summary found at {summary_path} -- "
            "either run run_subsampling_epsilon.py first, or pass explicit "
            "--eid values instead of relying on auto-selection"
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
        y_test = res["y_test"]
        y_pred = res["y_test_pred"]

        jitter = (np.random.default_rng(0).uniform(-0.02, 0.02, size=len(y_test)))
        ax.scatter(
            y_test + jitter, y_pred, s=8, color=DATA_COLOR, alpha=0.35,
            linewidths=0, zorder=2, label="test trials",
        )

        conditions = sorted(res["pred_mean_by_condition"].keys())
        means = [res["pred_mean_by_condition"][c] for c in conditions]
        ax.plot(
            conditions, means, "o-", color=MEAN_COLOR, linewidth=1.5,
            markersize=4, zorder=3, label="mean pred | true contrast",
        )

        lims = [min(conditions) - 0.1, max(conditions) + 0.1]
        ax.plot(lims, lims, "--", color=IDENTITY_COLOR, linewidth=1, zorder=1,
                 label="y = x (perfect)")

        ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{res['group_label']}: {res['eid'][:8]}\n"
            f"eps={res['epsilon_from_summary']:.3f}  r2={res['test_r2']:.3f}  "
            f"pred_std/true_std={res['pred_std']/max(res['true_std'],1e-9):.2f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="best", frameon=False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.supxlabel("true signed contrast", fontsize=10)
    fig.supylabel("predicted signed contrast", fontsize=10)
    fig.tight_layout()
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "results" / "inspect_predictions" / args.target_prefix
    )
    details_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    if args.eid:
        picks = [(eid, "manual", float("nan")) for eid in args.eid]
    else:
        picks = pick_sessions(args.target_prefix, args.n_per_group)

    print(f"target_prefix={args.target_prefix}")
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
            res = inspect_one_session(
                one, atlas, eid, args.target_prefix, seed=args.seed
            )
            res["group_label"] = label
            res["epsilon_from_summary"] = eps
            details_path = details_dir / f"{eid}_prediction_diagnostics.pkl"
            with open(details_path, "wb") as f:
                pickle.dump(res, f)
            plot_results.append(res)
            print(
                f"  ok: test_r2={res['test_r2']:.4f}, "
                f"pred_std={res['pred_std']:.4f}, true_std={res['true_std']:.4f}, "
                f"pred_true_corr={res['pred_true_corr']:.4f}"
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

    summary_path = output_dir / f"{args.target_prefix}_prediction_diagnostics.csv"
    write_csv(csv_rows, summary_path)
    print(f"summary={summary_path}")

    if plot_results:
        plot_path = output_dir / f"{args.target_prefix}_prediction_diagnostics_grid.png"
        plot_grid(plot_results, plot_path)
        print(f"plot={plot_path}")
    else:
        print("no successful sessions -- skipping plot")


if __name__ == "__main__":
    main()