"""
run_subsampling_epsilon.py

"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from sklearn.linear_model import RidgeCV

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

from src.subsampling_curve import (  # noqa: E402
    fit_information_limiting_intercept,
    get_test_mse_cv,
    log_spaced_neuron_counts,
)
from run_0_100ms_decoders import (  # noqa: E402
    build_eids,
    load_session_0_100ms,
    make_contrast_masks,
)


SEED = 0
# RidgeCV picks alpha per fit from this grid via efficient LOO (cv=None) --
# NOT a single fixed alpha -- see notes on why a fixed alpha across very
# different neuron-subsample sizes N can make balanced_test_mse rise with
# N (overfitting at large N) and produce a spurious negative regression
# slope that has nothing to do with signal-noise alignment.
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
K = 12  # max number of N grid points requested from log_spaced_neuron_counts
N_MIN = 20  # floor on neuron-subsample size (see notes: <20 is too degenerate)
# repeat count per N -- get_test_mse_cv threads this through explicitly
# (unlike the earlier get_test_mse, which silently ignored it).
NUM_SAMPLES = 20
# stratified k-fold CV per repeat instead of a single 50/50 split -- see
# notes: at a fixed 50/50 split, training-set size doesn't grow with the
# neuron-subsample size N, so at large N the decoder can be estimating more
# parameters than the training data comfortably supports, which alone can
# make balanced_test_mse rise with N (negative regression slope) even with
# a well-chosen alpha. n_splits=5 trains on 4/5 of the trials per fold
# while still testing every trial exactly once across folds.
N_SPLITS = 5
ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"


def neuron_wise_trial_shuffle(X, rng):
    """
    Independently permutes the trial (row) order of EACH neuron (column)
    of X. After this, for any given trial position, the population
    vector is an arbitrary combination of different neurons' unrelated
    trials -- both that neuron's tuning to the trial's real contrast AND
    any cross-neuron noise correlation are destroyed, while each
    neuron's own marginal distribution of firing rates across trials is
    preserved exactly (same values, just reassigned to different trial
    positions, independently per neuron).

    Trial POSITIONS are unchanged (X_shuffled has the same shape/order as
    X), so the real signed_contrast / masks can be reused as-is against
    X_shuffled without any relabeling -- each trial position still gets
    its real contrast label, it's just paired with a fabricated
    population vector instead of the real one.
    """
    n_trials, n_neurons = X.shape
    X_shuffled = np.empty_like(X)
    for j in range(n_neurons):
        perm = rng.permutation(n_trials)
        X_shuffled[:, j] = X[perm, j]
    return X_shuffled


def run_one_session_epsilon(
    one, atlas, eid, target_prefix, k=K, n_min=N_MIN, seed=SEED, n_splits=N_SPLITS
):
    loaded = load_session_0_100ms(
        one=one,
        atlas=atlas,
        eid=eid,
        target_prefix=target_prefix,
    )
    X = loaded["X"]
    n_total = X.shape[1]

    masks = make_contrast_masks(loaded["signed_contrast"])
    ints = log_spaced_neuron_counts(n_total, k=k, n_min=n_min)

    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
    mse_results, mse_std_results = get_test_mse_cv(
        X, ints, masks, model, seed, num_samples=NUM_SAMPLES, n_splits=n_splits
    )

    fit = fit_information_limiting_intercept(
        mse_results, mse_std_results, num_samples=NUM_SAMPLES
    )

    details = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "shuffle": "none",
        "seed": seed,
        "k": k,
        "n_min": n_min,
        "num_samples": NUM_SAMPLES,
        "n_splits": n_splits,
        # NOTE: model.alpha_ after the loop only reflects whichever (N,
        # repeat, fold) combination was fit LAST inside get_test_mse_cv --
        # RidgeCV re-selects alpha independently on every single .fit()
        # call, so this one value is not "the" alpha used throughout and
        # is not saved here to avoid it being misread that way. Getting a
        # proper per-N alpha_selected history would need get_test_mse_cv
        # itself to capture and return it per fit -- ask if you want that
        # added.
        "n_total_units": int(n_total),
        "n_grid": ints,
        "mse_results": mse_results,
        "mse_std_results": mse_std_results,
        "epsilon": fit["intercept"],
        "slope": fit["slope"],
        "se_intercept": fit["se_intercept"],
        "p_value": fit["p_value"],
        "n_points": fit["n_points"],
    }

    summary = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "shuffle": "none",
        "status": "ok",
        "error": "",
        "n_total_units": int(n_total),
        "n_grid_points": fit["n_points"],
        "epsilon": fit["intercept"],
        "se_intercept": fit["se_intercept"],
        "p_value": fit["p_value"],
        "slope": fit["slope"],
    }
    return summary, details


def run_one_session_epsilon_shuffled(
    one, atlas, eid, target_prefix, k=K, n_min=N_MIN, seed=SEED, n_splits=N_SPLITS
):
    """
    Same as run_one_session_epsilon, but computed on a neuron-wise
    trial-shuffled pseudo-X (see neuron_wise_trial_shuffle) instead of
    the real X. Everything downstream of X -- masks, log_spaced_neuron_
    counts, get_test_mse_cv, fit_information_limiting_intercept -- is the
    identical, unmodified real pipeline; only the input X differs.

    ONE shuffle realization is drawn per session (seeded from `seed`,
    same convention as everywhere else in this script) and reused across
    all N / all repeats inside get_test_mse_cv -- mirroring how the real
    epsilon is computed from one fixed real X, not re-shuffled per
    repeat. If you want the shuffle-realization variance itself
    quantified (multiple independent shuffles per session, averaged),
    that's a straightforward follow-up (loop this over several
    shuffle seeds) -- not done here since it wasn't asked for.
    """
    loaded = load_session_0_100ms(
        one=one,
        atlas=atlas,
        eid=eid,
        target_prefix=target_prefix,
    )
    X_real = loaded["X"]
    n_total = X_real.shape[1]

    shuffle_rng = np.random.default_rng(seed)
    X = neuron_wise_trial_shuffle(X_real, shuffle_rng)

    masks = make_contrast_masks(loaded["signed_contrast"])
    ints = log_spaced_neuron_counts(n_total, k=k, n_min=n_min)

    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
    mse_results, mse_std_results = get_test_mse_cv(
        X, ints, masks, model, seed, num_samples=NUM_SAMPLES, n_splits=n_splits
    )

    fit = fit_information_limiting_intercept(
        mse_results, mse_std_results, num_samples=NUM_SAMPLES
    )

    details = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "shuffle": "neuron_wise_trial",
        "seed": seed,
        "k": k,
        "n_min": n_min,
        "num_samples": NUM_SAMPLES,
        "n_splits": n_splits,
        "n_total_units": int(n_total),
        "n_grid": ints,
        "mse_results": mse_results,
        "mse_std_results": mse_std_results,
        "epsilon": fit["intercept"],
        "slope": fit["slope"],
        "se_intercept": fit["se_intercept"],
        "p_value": fit["p_value"],
        "n_points": fit["n_points"],
    }

    summary = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "shuffle": "neuron_wise_trial",
        "status": "ok",
        "error": "",
        "n_total_units": int(n_total),
        "n_grid_points": fit["n_points"],
        "epsilon": fit["intercept"],
        "se_intercept": fit["se_intercept"],
        "p_value": fit["p_value"],
        "slope": fit["slope"],
    }
    return summary, details


SUMMARY_FIELDS = [
    "eid",
    "pid",
    "target_prefix",
    "shuffle",
    "status",
    "error",
    "n_total_units",
    "n_grid_points",
    "epsilon",
    "se_intercept",
    "p_value",
    "slope",
]


def write_summary(rows, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-min", type=int, default=N_MIN)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--shuffle", action="store_true",
        help="compute epsilon on a neuron-wise trial-shuffled null "
             "(destroys tuning AND noise correlations) instead of the "
             "real data. Writes to a separate output dir by default "
             "(results/subsampling_epsilon_shuffled/<target_prefix>/) so "
             "it never overwrites real results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_json = (
        REPO_ROOT
        / "results"
        / "region_scan"
        / args.target_prefix
        / f"{args.target_prefix}_subjects_by_lab.json"
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif args.shuffle:
        output_dir = REPO_ROOT / "results" / "subsampling_epsilon_shuffled" / args.target_prefix
    else:
        output_dir = REPO_ROOT / "results" / "subsampling_epsilon" / args.target_prefix
    details_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    suffix = "shuffled" if args.shuffle else "epsilon"
    summary_path = (
        output_dir / f"{args.target_prefix}_subsampling_{suffix}_summary.csv"
    )

    eids = build_eids(input_json)
    print(f"target_prefix={args.target_prefix}")
    print(f"input_json={input_json}")
    print(f"n_sessions={len(eids)}")
    print(f"output_dir={output_dir}")
    print(f"shuffle={args.shuffle}")
    print(
        f"k={args.k} n_min={args.n_min} num_samples={NUM_SAMPLES} "
        f"n_splits={args.n_splits} seed={args.seed}"
    )

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=ONE_CACHE_DIR,
    )
    atlas = AllenAtlas()

    run_fn = run_one_session_epsilon_shuffled if args.shuffle else run_one_session_epsilon

    rows = []
    for session_index, eid in enumerate(eids, start=1):
        print(f"[{session_index}/{len(eids)}] eid={eid}")
        try:
            summary, details = run_fn(
                one=one,
                atlas=atlas,
                eid=eid,
                target_prefix=args.target_prefix,
                k=args.k,
                n_min=args.n_min,
                seed=args.seed,
                n_splits=args.n_splits,
            )
            details_path = details_dir / f"{eid}_subsampling_{suffix}.pkl"
            with open(details_path, "wb") as f:
                pickle.dump(details, f)
            print(
                f"  ok: n_units={summary['n_total_units']}, "
                f"n_grid_points={summary['n_grid_points']}, "
                f"epsilon={summary['epsilon']:.6f} "
                f"(se={summary['se_intercept']:.6f}, p={summary['p_value']:.4f})"
            )
        except Exception as exc:
            summary = {
                "eid": eid,
                "pid": "",
                "target_prefix": args.target_prefix,
                "shuffle": "neuron_wise_trial" if args.shuffle else "none",
                "status": "failed",
                "error": repr(exc),
                "n_total_units": "",
                "n_grid_points": "",
                "epsilon": "",
                "se_intercept": "",
                "p_value": "",
                "slope": "",
            }
            print(f"  failed: {summary['error']}")

        rows.append(summary)
        write_summary(rows, summary_path)

    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: {n_ok}/{len(rows)} sessions succeeded")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()