# run_decoder_r2_difference.py

from __future__ import annotations

import argparse
import csv
from doctest import REPORT_CDIFF
import os
import pickle
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent
    if (SCRIPT_DIR.parent / "src").is_dir()
    else SCRIPT_DIR
)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from iblatlas.atlas import AllenAtlas
from one.api import ONE

from src.alignment_metrics import compute_noise_covariance
from src.directional_r2_profile import directional_r2_profile
from run_0_100ms_decoders import load_session_0_100ms


ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"
N_RANDOM = 500
SEED = 0
ATOL = 1e-6


def find_detail_paths(decoder_dir, suffix="_decoder_0_100ms.pkl"):
    """
    Recursively finds every *_decoder_0_100ms.pkl under decoder_dir --
    works whether the pipeline wrote them directly into
    results/decoder_0_100ms/<target_prefix>/ or into a details/
    subfolder underneath it.
    """
    return sorted(Path(decoder_dir).rglob(f"*{suffix}"))


def load_details(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def rebuild_std_splits(details, one, atlas, atol=ATOL):
    """
    Re-fetches the full session X, slices it with the saved train/test
    trial indices, and standardizes with the saved scaler_mean/scale
    (not refit). Verifies the reconstruction against the saved
    y_train_pred before returning, so a bad recompute never silently
    flows into the covariance / R^2 computation.
    """
    loaded = load_session_0_100ms(
        one=one,
        atlas=atlas,
        eid=details["eid"],
        target_prefix=details["target_prefix"],
    )
    X_full = np.asarray(loaded["X"], dtype=float)
    n_trials = details["n_trials"]
    if X_full.shape[0] != n_trials:
        raise ValueError(
            f"recomputed X has {X_full.shape[0]} trials, details expects "
            f"{n_trials} -- session data may have changed since the "
            f"original pipeline run."
        )

    train_idx = np.asarray(details["train_trial_indices_filtered"])
    test_idx = np.asarray(details["test_trial_indices_filtered"])

    scaler_mean = np.asarray(details["scaler_mean"])
    scaler_scale = np.asarray(details["scaler_scale"])

    X_train_std = (X_full[train_idx] - scaler_mean) / scaler_scale
    X_test_std = (X_full[test_idx] - scaler_mean) / scaler_scale

    w_std = np.asarray(details["w_decoder_neuron_std"])
    intercept = details["decoder_intercept"]
    y_train_pred_check = X_train_std @ w_std + intercept
    saved_pred = np.asarray(details["y_train_pred"])
    if not np.allclose(y_train_pred_check, saved_pred, atol=atol):
        max_diff = float(np.max(np.abs(y_train_pred_check - saved_pred)))
        raise ValueError(
            f"recomputed X does not reproduce saved y_train_pred "
            f"(max abs diff={max_diff:.3g}) -- trial alignment is off "
            f"for this session."
        )

    return X_train_std, X_test_std


def r2_difference_for_session(details, one, atlas, n_random=N_RANDOM, seed=SEED):
    X_train_std, X_test_std = rebuild_std_splits(details, one, atlas)
    y_train = np.asarray(details["y_train"])
    y_test = np.asarray(details["y_test"])

    residual_parts = []
    for c in np.unique(y_train):
        mask = y_train == c
        mu_c = X_train_std[mask].mean(axis=0)
        residual_parts.append(X_train_std[mask] - mu_c)
    R = np.concatenate(residual_parts, axis=0)
    residual_mask = np.ones(R.shape[0], dtype=bool)
    Sigma_noise = compute_noise_covariance(R, residual_mask)

    return directional_r2_profile(
        X_train_std, y_train, X_test_std, y_test,
        ridge_test_r2=details["balanced_test_r2"],
        noise_covariance=Sigma_noise,
        seed=seed,
        n_random=n_random,
    )


SUMMARY_FIELDS = [
    "eid",
    "pid",
    "target_prefix",
    "status",
    "error",
    "n_trials",
    "n_units",
    "w_r2",
    "top1_r2",
    "null_r2_mean",
    "null_r2_std",
    "null_r2_percentile_of_top1",
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
    parser.add_argument(
        "--decoder-dir",
        default=None,
        help=(
            "Folder to search recursively for *_decoder_0_100ms.pkl. "
            "Defaults to <repo_root>/results/decoder_0_100ms/<target_prefix> "
            "(covers both that folder directly and any details/ subfolder "
            "under it). Pass this explicitly if the default doesn't match "
            "where your pipeline actually wrote its output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder to write the summary CSV / per-eid details pkls to. "
            "Defaults to <repo_root>/results/decoder_r2_difference/"
            "<target_prefix>."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = REPO_ROOT

    decoder_dir = (
        Path(args.decoder_dir).resolve()
        if args.decoder_dir
        else repo_root / "results" / "decoder_0_100ms" / args.target_prefix
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_root / "results" / "decoder_r2_difference" / args.target_prefix
    )
    out_details_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_details_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_dir / f"{args.target_prefix}_decoder_r2_difference_summary.csv"
    )

    detail_paths = find_detail_paths(decoder_dir)
    print(f"target_prefix={args.target_prefix}")
    print(f"decoder_dir={decoder_dir}")
    print(f"n_sessions={len(detail_paths)}")
    print(f"output_dir={output_dir}")
    if not detail_paths:
        print(
            f"WARNING: no *_decoder_0_100ms.pkl found under {decoder_dir} "
            f"-- check --decoder-dir if this isn't expected."
        )

    ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        cache_dir=ONE_CACHE_DIR,
    )
    atlas = AllenAtlas()

    rows = []
    for session_index, path in enumerate(detail_paths, start=1):
        try:
            details = load_details(path)
            eid = details["eid"]
        except Exception as exc:
            print(f"[{session_index}/{len(detail_paths)}] {path}: failed to load pkl ({exc!r})")
            rows.append({
                "eid": path.stem,
                "pid": "",
                "target_prefix": args.target_prefix,
                "status": "failed",
                "error": f"failed to load pkl: {exc!r}",
                "n_trials": "",
                "n_units": "",
                "w_r2": "",
                "top1_r2": "",
                "null_r2_mean": "",
                "null_r2_std": "",
                "null_r2_percentile_of_top1": "",
            })
            write_summary(rows, summary_path)
            continue

        print(f"[{session_index}/{len(detail_paths)}] eid={eid} ({path})")
        try:
            result = r2_difference_for_session(details, one, atlas)
            null_r2 = result["null_r2"]
            percentile = float(100.0 * np.mean(null_r2 < result["noise_top1_r2"]))
            summary = {
                "eid": eid,
                "pid": details.get("pid", ""),
                "target_prefix": args.target_prefix,
                "status": "ok",
                "error": "",
                "n_trials": details.get("n_trials", ""),
                "n_units": details.get("n_units", ""),
                "w_r2": result["w_r2"],
                "top1_r2": result["noise_top1_r2"],
                "null_r2_mean": float(null_r2.mean()),
                "null_r2_std": float(null_r2.std()),
                "null_r2_percentile_of_top1": percentile,
            }
            with open(out_details_dir / f"{eid}_r2_difference.pkl", "wb") as f:
                pickle.dump(result, f)
            print(
                f"  ok: w_r2={summary['w_r2']:.4f}, "
                f"top1_r2={summary['top1_r2']:.4f}, "
                f"null_mean={summary['null_r2_mean']:.4f}, "
                f"percentile={summary['null_r2_percentile_of_top1']:.1f}"
            )
        except Exception as exc:
            summary = {
                "eid": eid,
                "pid": details.get("pid", ""),
                "target_prefix": args.target_prefix,
                "status": "failed",
                "error": repr(exc),
                "n_trials": details.get("n_trials", ""),
                "n_units": details.get("n_units", ""),
                "w_r2": "",
                "top1_r2": "",
                "null_r2_mean": "",
                "null_r2_std": "",
                "null_r2_percentile_of_top1": "",
            }
            print(f"  failed: {summary['error']}")

        rows.append(summary)
        write_summary(rows, summary_path)

    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: {n_ok}/{len(rows)} sessions succeeded")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()