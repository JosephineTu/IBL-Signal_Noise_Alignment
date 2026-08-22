"""
run_subsampling_epsilon_rotation_null.py

Rotate-eigenspectrum null for the FULL subsampling curve (not just a
single-N MSE like run_alignment_3(_direct.py)): estimates mu_by_condition
and Sigma_noise from one real train fold, then reconstructs a full
pseudo-X (all trials, real trial order/labels) from mu_by_condition +
noise drawn from Sigma_noise's eigendecomposition, either un-rotated
(rot=I, the "identity" reconstruction) or under a random rotation (same
eigenvalue spectrum, scrambled orientation) -- and runs EACH
reconstruction through the real, unmodified subsampling-curve pipeline
(log_spaced_neuron_counts / get_test_mse_cv / fit_information_limiting_
intercept) to get a full MSE(N) curve and its own fitted epsilon, not
just one MSE value.

Comparison anchor is identity_epsilon (rot=I), NOT the raw real epsilon
from run_subsampling_epsilon.py -- same reason as run_alignment_3(_direct
.py)'s identity_mse: comparing raw real data against a synthetic-
reconstruction null would confound the actual orientation effect with
the Ledoit-Wolf-shrinkage/regeneration model's own systematic gap from
reality. percentile_of_identity = where identity_epsilon falls within
the null_epsilons distribution.

mu_by_condition / Sigma_noise estimation uses the WITHIN-CONDITION
(residual) std fix, not marginal (across-all-conditions) std -- see the
run_alignment_3_direct.py bug writeup: standardizing by marginal
variance before estimating mu_by_condition/Sigma_noise mechanically
couples Sigma_noise's principal axes to the signal direction (tuning-
strong neurons get their z-scored noise artificially shrunk), producing
a systematic LOW bias in any identity-vs-rotation comparison. Validated
fix: scale by pooled within-condition residual std instead (isotropic-
null-style validation: percentile mean 50.55/53.23 across 100 synthetic
trials at two different alpha values, vs. ~15 before the fix). The
reconstructed pseudo-X is scaled back to raw-firing-rate-like units
(multiplied back by resid_std) before being handed to get_test_mse_cv,
so it goes through the exact same preprocessing real X would.

COMPUTE COST WARNING: get_test_mse_cv already runs
k * NUM_SAMPLES * n_splits Ridge(CV) fits for ONE curve (12*20*5=1200 by
default). This script runs that once for the identity draw and once per
null draw -- so N_RANDOM=500 (run_alignment_3(_direct.py)'s convention)
would mean ~600,000 RidgeCV fits per SESSION, which is not tractable.
Default N_RANDOM here is deliberately much smaller (20) -- bump via
--n-random only with this cost in mind, and consider lowering --k /
--n-splits together with it if you need more draws.

Full mse_results/mse_std_results curves are saved for the identity draw
always, and for the first --n-save-null-curves null draws (default 5,
just for visual inspection alongside identity) -- NOT for every null
draw, to keep the per-session pkl a reasonable size when n_random is
large. null_epsilons (scalar per draw) IS saved in full regardless.

Writes:
  results/subsampling_epsilon_rotation_null/<target_prefix>/
      {target_prefix}_subsampling_rotation_null_summary.csv
      details/{eid}_rotation_null.pkl
      plots/{eid}_rotation_null_curve.png   (identity + a few null curves)
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

from src.decoder import make_train_test_sets  # noqa: E402
from src.alignment_metrics import compute_noise_covariance  # noqa: E402
from src.space_rotation_decoder import random_rotation_matrix  # noqa: E402
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
from plot_subsampling_epsilon import DATA_COLOR, GRID_COLOR  # noqa: E402


SEED = 0
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
K = 12
N_MIN = 20
NUM_SAMPLES = 20
N_SPLITS = 5
N_RANDOM = 20  # see COMPUTE COST WARNING above -- deliberately << method 2's 500
N_SAVE_NULL_CURVES = 5  # how many null draws' full curves to keep, for plotting only
ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"

IDENTITY_COLOR = "#3F6FBF"  # matches plot_subsampling_epsilon.py's DATA_COLOR family
NULL_COLOR = "#B0B0B0"
REAL_COLOR = "#2E9E6C"


def estimate_mu_and_sigma_corrected(X_train, y_train):
    """
    mu_by_condition / Sigma_noise, estimated using WITHIN-CONDITION
    (residual) std to scale -- NOT marginal std. Returns them in the
    resid-std-scaled space, plus resid_std itself so callers can rescale
    reconstructed data back to raw units. See module docstring.
    """
    contrasts = np.unique(y_train)
    mu_c_raw = {}
    residual_raw_parts = []
    for c in contrasts:
        mask = y_train == c
        mu_c_raw[c] = X_train[mask].mean(axis=0)
        residual_raw_parts.append(X_train[mask] - mu_c_raw[c])
    R_raw = np.concatenate(residual_raw_parts, axis=0)
    resid_std = np.maximum(R_raw.std(axis=0, ddof=1), 1e-8)

    X_train_scaled = X_train / resid_std
    mu_by_condition = {}
    residual_parts = []
    for c in contrasts:
        mask = y_train == c
        mu_c = X_train_scaled[mask].mean(axis=0)
        mu_by_condition[c] = mu_c
        residual_parts.append(X_train_scaled[mask] - mu_c)
    R = np.concatenate(residual_parts, axis=0)
    Sigma_noise = compute_noise_covariance(R, np.ones(R.shape[0], dtype=bool))
    return mu_by_condition, Sigma_noise, resid_std


def reconstruct_pseudo_X_full_session(signed_contrast, mu_by_condition, Sigma_noise, resid_std, rot, rng):
    """
    Full-session pseudo-X (n_trials x p, real trial order), from
    mu_by_condition + noise ~ N(0, rot @ Sigma_noise @ rot.T), rescaled
    back to raw-firing-rate-like units (* resid_std) so it goes through
    get_test_mse_cv's own preprocessing exactly like real X would. Same
    eigendecomposition trick as run_alignment_3.py's
    rotated_noise_samples (verified there to <0.3% relative Frobenius
    error against 2e5 draws), inlined here rather than imported since
    src/ shouldn't import from scripts/.
    """
    p = Sigma_noise.shape[0]
    n_trials = len(signed_contrast)
    eigvals, eigvecs = np.linalg.eigh(Sigma_noise)
    eigvals = np.clip(eigvals, 0, None)
    A = rot @ eigvecs @ np.diag(np.sqrt(eigvals))
    z = rng.normal(size=(p, n_trials))
    noise_scaled = (A @ z).T
    mu_per_trial_scaled = np.array([mu_by_condition[c] for c in signed_contrast])
    X_scaled = mu_per_trial_scaled + noise_scaled
    return X_scaled * resid_std


def _fit_curve_for_X(X, ints, masks, seed, num_samples, n_splits):
    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
    mse_results, mse_std_results = get_test_mse_cv(
        X, ints, masks, model, seed, num_samples=num_samples, n_splits=n_splits
    )
    fit = fit_information_limiting_intercept(mse_results, mse_std_results, num_samples=num_samples)
    return mse_results, mse_std_results, fit


def run_one_session_rotation_null(
    one, atlas, eid, target_prefix, n_random=N_RANDOM, k=K, n_min=N_MIN,
    seed=SEED, n_splits=N_SPLITS, num_samples=NUM_SAMPLES,
    n_save_null_curves=N_SAVE_NULL_CURVES,
):
    loaded = load_session_0_100ms(one=one, atlas=atlas, eid=eid, target_prefix=target_prefix)
    X_real = loaded["X"]
    signed_contrast = loaded["signed_contrast"]
    n_total = X_real.shape[1]
    masks = make_contrast_masks(signed_contrast)
    ints = log_spaced_neuron_counts(n_total, k=k, n_min=n_min)

    X_train, y_train, X_test, y_test = make_train_test_sets(masks, X_real, seed=seed)
    mu_by_condition, Sigma_noise, resid_std = estimate_mu_and_sigma_corrected(X_train, y_train)
    p = Sigma_noise.shape[0]

    rng = np.random.default_rng(seed)

    X_identity = reconstruct_pseudo_X_full_session(
        signed_contrast, mu_by_condition, Sigma_noise, resid_std, np.eye(p), rng
    )
    mse_results_id, mse_std_results_id, fit_id = _fit_curve_for_X(
        X_identity, ints, masks, seed, num_samples, n_splits
    )

    null_epsilons = np.empty(n_random)
    null_curves = []
    for j in range(n_random):
        rot = random_rotation_matrix(p, rng)
        X_null = reconstruct_pseudo_X_full_session(
            signed_contrast, mu_by_condition, Sigma_noise, resid_std, rot, rng
        )
        mse_results_j, mse_std_results_j, fit_j = _fit_curve_for_X(
            X_null, ints, masks, seed, num_samples, n_splits
        )
        null_epsilons[j] = fit_j["intercept"]
        if j < n_save_null_curves:
            null_curves.append({
                "mse_results": mse_results_j, "mse_std_results": mse_std_results_j,
                "epsilon": fit_j["intercept"], "slope": fit_j["slope"],
            })

    percentile_of_identity = float(100.0 * np.mean(null_epsilons < fit_id["intercept"]))

    details = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "seed": seed,
        "k": k,
        "n_min": n_min,
        "num_samples": num_samples,
        "n_splits": n_splits,
        "n_random": n_random,
        "n_total_units": int(n_total),
        "n_grid": ints,
        # identity draw -- the actual comparison anchor; full curve kept
        # for plotting (compatible with plot_subsampling_epsilon.py's
        # plot_one_session: mse_results/mse_std_results/num_samples/
        # epsilon/slope/eid/n_total_units/p_value)
        "mse_results": mse_results_id,
        "mse_std_results": mse_std_results_id,
        "epsilon": fit_id["intercept"],
        "slope": fit_id["slope"],
        "se_intercept": fit_id["se_intercept"],
        "p_value": fit_id["p_value"],
        "n_points": fit_id["n_points"],
        # null distribution -- scalar epsilon per draw, always kept in full
        "null_epsilons": null_epsilons,
        "null_epsilon_mean": float(null_epsilons.mean()),
        "null_epsilon_std": float(null_epsilons.std()),
        "percentile_of_identity": percentile_of_identity,
        # a handful of full null curves, for the comparison plot only
        "null_curves_saved": null_curves,
    }

    summary = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "status": "ok",
        "error": "",
        "n_total_units": int(n_total),
        "n_grid_points": fit_id["n_points"],
        "identity_epsilon": fit_id["intercept"],
        "null_epsilon_mean": float(null_epsilons.mean()),
        "null_epsilon_std": float(null_epsilons.std()),
        "percentile_of_identity": percentile_of_identity,
        "n_random": n_random,
    }
    return summary, details


def plot_rotation_null_session(details, output_path):
    """
    Overlays: the identity-reconstruction curve + its fit (blue, thick),
    a handful of null-rotation curves (gray, thin, unlabeled individually
    -- just to show the spread), and the fitted line for each. No
    per-null-draw legend entries (would be unreadable with several
    curves) -- a single "null rotations (n=...)" label covers them.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, null_c in enumerate(details["null_curves_saved"]):
        Ns = np.array(sorted(null_c["mse_results"].keys()), dtype=float)
        x = 1.0 / Ns
        y = np.array([null_c["mse_results"][n] for n in Ns])
        ax.plot(x, y, "o-", ms=3, lw=0.8, color=NULL_COLOR, alpha=0.6, zorder=1,
                 label="null rotations" if i == 0 else None)

    Ns = np.array(sorted(details["mse_results"].keys()), dtype=float)
    x_id = 1.0 / Ns
    y_id = np.array([details["mse_results"][n] for n in Ns])
    se_id = np.array([details["mse_std_results"][n] for n in Ns]) / np.sqrt(details["num_samples"])
    ax.errorbar(x_id, y_id, yerr=se_id, fmt="o", ms=5, color=IDENTITY_COLOR,
                 ecolor=IDENTITY_COLOR, elinewidth=1, capsize=2, zorder=3,
                 label=f"identity (real orientation): ε={details['epsilon']:.4f}")
    x_line = np.linspace(0, x_id.max() * 1.05, 100)
    ax.plot(x_line, details["epsilon"] + details["slope"] * x_line, "-",
             color=IDENTITY_COLOR, linewidth=1.5, zorder=2)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(left=0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("1 / N (neurons)")
    ax.set_ylabel("balanced_test_mse")
    ax.set_title(
        f"{details['eid'][:8]}  percentile_of_identity={details['percentile_of_identity']:.1f}  "
        f"(null_mean={details['null_epsilon_mean']:.4f}, n_random={details['n_random']})",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


SUMMARY_FIELDS = [
    "eid", "pid", "target_prefix", "status", "error",
    "n_total_units", "n_grid_points", "identity_epsilon",
    "null_epsilon_mean", "null_epsilon_std", "percentile_of_identity", "n_random",
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
    parser.add_argument("--eid", action="append", default=None,
                          help="process only this eid (repeatable); omit to "
                               "process every eid in the region-scan JSON")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-min", type=int, default=N_MIN)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--n-random", type=int, default=N_RANDOM,
                          help="see COMPUTE COST WARNING in the module docstring")
    parser.add_argument("--n-save-null-curves", type=int, default=N_SAVE_NULL_CURVES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else REPO_ROOT / "results" / "subsampling_epsilon_rotation_null" / args.target_prefix
    )
    details_dir = output_dir / "details"
    plots_dir = output_dir / "plots"
    for d in (output_dir, details_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.target_prefix}_subsampling_rotation_null_summary.csv"

    if args.eid:
        eids = args.eid
    else:
        input_json = (
            REPO_ROOT / "results" / "region_scan" / args.target_prefix
            / f"{args.target_prefix}_subjects_by_lab.json"
        )
        eids = build_eids(input_json)

    print(f"target_prefix={args.target_prefix}")
    print(f"n_sessions={len(eids)}")
    print(f"output_dir={output_dir}")
    print(f"n_random={args.n_random} k={args.k} num_samples={args.num_samples} "
          f"n_splits={args.n_splits} seed={args.seed}")

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=ONE_CACHE_DIR,
    )
    atlas = AllenAtlas()

    rows = []
    for session_index, eid in enumerate(eids, start=1):
        print(f"[{session_index}/{len(eids)}] eid={eid}")
        try:
            summary, details = run_one_session_rotation_null(
                one, atlas, eid, args.target_prefix,
                n_random=args.n_random, k=args.k, n_min=args.n_min,
                seed=args.seed, n_splits=args.n_splits, num_samples=args.num_samples,
                n_save_null_curves=args.n_save_null_curves,
            )
            with open(details_dir / f"{eid}_rotation_null.pkl", "wb") as f:
                pickle.dump(details, f)
            plot_rotation_null_session(details, plots_dir / f"{eid}_rotation_null_curve.png")
            print(
                f"  ok: identity_epsilon={summary['identity_epsilon']:.4f}, "
                f"null_mean={summary['null_epsilon_mean']:.4f}, "
                f"null_std={summary['null_epsilon_std']:.4f}, "
                f"percentile_of_identity={summary['percentile_of_identity']:.1f}"
            )
            row = {k: summary.get(k, "") for k in SUMMARY_FIELDS}
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row = {k: "" for k in SUMMARY_FIELDS}
            row.update(eid=eid, target_prefix=args.target_prefix, status="failed",
                        error=repr(exc), n_random=args.n_random)
            print(f"  failed: {row['error']}")
        rows.append(row)
        write_summary(rows, summary_path)

    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: {n_ok}/{len(rows)} sessions succeeded")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()