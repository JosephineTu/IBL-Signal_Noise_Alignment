from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from brainbox.io.one import SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from iblatlas.atlas import AllenAtlas
from one.api import ONE
from sklearn.linear_model import Ridge

# NOTE: this assumes decoder.py lives in the same directory as this script
# (e.g. both under scripts/). If decoder.py is under src/ instead, change
# this to `from src.decoder import ...` or add src/ to sys.path.
from src.decoder import make_train_test_sets, ridge_regression


SEED = 0
MIN_TRIALS = 10
MIN_UNITS = 5
ALPHA = 1.0
WINDOW_START = 0.0
WINDOW_END = 0.1
ONE_CACHE_DIR = "/scratch/midway3/xiaorantu/ONE"


def ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(f"{name} has unexpected shape {x.shape}")


def build_eids(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    eids = []
    for lab_results in results.values():
        for subject_results in lab_results.values():
            eids.extend(subject_results["region_eids"])

    return list(dict.fromkeys(eids))


def pick_best_insertion(one, atlas, eid, target_prefix):
    insertions = one.alyx.rest("insertions", "list", session=eid)
    best_pid = None
    best_n_units = -1

    for insertion in insertions:
        pid = insertion["id"]
        try:
            loader = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
            spikes, clusters, channels = loader.load_spike_sorting()
            clusters = loader.merge_clusters(spikes, clusters, channels)
            acronyms = np.asarray(clusters["acronym"])
            n_units = int(
                np.sum([a.startswith(target_prefix) for a in acronyms])
            )
        except Exception:
            continue

        if n_units > best_n_units:
            best_pid = pid
            best_n_units = n_units

    if best_pid is None:
        raise RuntimeError(f"No valid insertion found for eid={eid}")

    return best_pid


def load_session_0_100ms(one, atlas, eid, target_prefix):
    trials = one.load_object(eid, "trials", collection="alf")

    stim_on = ensure_1d(trials["stimOn_times"], "stimOn_times")
    stim_off = ensure_1d(trials["stimOff_times"], "stimOff_times")
    valid_trials = np.isfinite(stim_on) & np.isfinite(stim_off)
    original_trial_indices = np.flatnonzero(valid_trials)
    n_trials_original = len(stim_on)

    for key in list(trials.keys()):
        value = np.asarray(trials[key])
        if value.ndim >= 1 and value.shape[0] == n_trials_original:
            trials[key] = value[valid_trials]
        else:
            trials[key] = value

    stim_on = stim_on[valid_trials]

    pid = pick_best_insertion(one, atlas, eid, target_prefix)
    loader = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters = loader.merge_clusters(spikes, clusters, channels)

    acronyms = np.asarray(clusters["acronym"])
    region_mask = np.asarray(
        [a.startswith(target_prefix) for a in acronyms],
        dtype=bool,
    )
    region_cluster_ids = np.asarray(clusters["cluster_id"])[region_mask]

    intervals = np.column_stack(
        [stim_on + WINDOW_START, stim_on + WINDOW_END]
    )
    counts, cluster_ids = get_spike_counts_in_bins(
        spikes["times"],
        spikes["clusters"],
        intervals,
    )
    cluster_ids = np.asarray(cluster_ids)

    if counts.shape == (len(cluster_ids), len(intervals)):
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[keep, :].T
        unit_ids = cluster_ids[keep]
    elif counts.shape == (len(intervals), len(cluster_ids)):
        keep = np.isin(cluster_ids, region_cluster_ids)
        counts = counts[:, keep]
        unit_ids = cluster_ids[keep]
    else:
        raise RuntimeError(
            f"Unexpected spike-count shape {counts.shape} for eid={eid}"
        )

    X = counts.astype(float) / (WINDOW_END - WINDOW_START)

    contrast_left = ensure_1d(
        trials["contrastLeft"], "contrastLeft"
    ).astype(float)
    contrast_right = ensure_1d(
        trials["contrastRight"], "contrastRight"
    ).astype(float)
    contrast_left = np.nan_to_num(contrast_left, nan=0.0)
    contrast_right = np.nan_to_num(contrast_right, nan=0.0)
    signed_contrast = contrast_left - contrast_right

    return {
        "trials": trials,
        "X": X,
        "signed_contrast": signed_contrast,
        "pid": pid,
        "unit_ids": unit_ids,
        "original_trial_indices": original_trial_indices,
    }


def make_contrast_masks(signed_contrast):
    return {
        float(contrast): signed_contrast == contrast
        for contrast in np.unique(signed_contrast)
    }


def fit_ridge_decoder(X_train, y_train, X_test, y_test):
    model = Ridge(alpha=ALPHA, solver="svd", fit_intercept=True)
    (
        balanced_test_mse,
        balanced_test_r2,
        w_std,
        X_train_std,
        X_test_std,
        scaler,
    ) = ridge_regression(
        X_train, y_train, X_test, y_test, model, return_extras=True
    )

    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)

    w_std_norm = float(np.linalg.norm(w_std))
    if w_std_norm == 0.0:
        raise RuntimeError("Decoder weight norm is zero")

    w_raw = w_std / scaler.scale_
    w_raw_norm = float(np.linalg.norm(w_raw))
    if w_raw_norm == 0.0:
        raise RuntimeError("Raw-space decoder weight norm is zero")

    return {
        "model": model,
        "scaler": scaler,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "balanced_test_mse": float(balanced_test_mse),
        "balanced_test_r2": float(balanced_test_r2),
        "w_decoder_neuron_std": w_std,
        "u_decoder_neuron_std": w_std / w_std_norm,
        "w_decoder_neuron_raw": w_raw,
        "u_decoder_neuron_raw": w_raw / w_raw_norm,
    }


def contrast_count_dict(masks):
    return {
        str(contrast): int(np.sum(mask))
        for contrast, mask in masks.items()
    }


def run_one_session(one, atlas, eid, target_prefix):
    loaded = load_session_0_100ms(
        one=one,
        atlas=atlas,
        eid=eid,
        target_prefix=target_prefix,
    )
    X = loaded["X"]
    if X.shape[1] < MIN_UNITS:
        raise RuntimeError(
            f"Only {X.shape[1]} {target_prefix} units; requires {MIN_UNITS}"
        )

    masks = make_contrast_masks(loaded["signed_contrast"])
    (
        X_train,
        y_train,
        X_test,
        y_test,
        train_indices,
        test_indices,
    ) = make_train_test_sets(
        masks=masks,
        X=X,
        seed=SEED,
        min_trials=MIN_TRIALS,
        return_indices=True,
    )

    fit = fit_ridge_decoder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    original_indices = loaded["original_trial_indices"]
    details = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "seed": SEED,
        "min_trials": MIN_TRIALS,
        "alpha": ALPHA,
        "n_trials": int(X.shape[0]),
        "n_units": int(X.shape[1]),
        "unit_ids": loaded["unit_ids"],
        "contrast_counts": contrast_count_dict(masks),
        "train_trial_indices_filtered": train_indices,
        "test_trial_indices_filtered": test_indices,
        "train_trial_indices_original": original_indices[train_indices],
        "test_trial_indices_original": original_indices[test_indices],
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": fit["y_train_pred"],
        "y_test_pred": fit["y_test_pred"],
        "balanced_test_mse": fit["balanced_test_mse"],
        "balanced_test_r2": fit["balanced_test_r2"],
        "scaler_mean": fit["scaler"].mean_,
        "scaler_scale": fit["scaler"].scale_,
        "decoder_intercept": float(fit["model"].intercept_),
        "w_decoder_neuron_std": fit["w_decoder_neuron_std"],
        "u_decoder_neuron_std": fit["u_decoder_neuron_std"],
        "w_decoder_neuron_raw": fit["w_decoder_neuron_raw"],
        "u_decoder_neuron_raw": fit["u_decoder_neuron_raw"],
    }

    summary = {
        "eid": eid,
        "pid": loaded["pid"],
        "target_prefix": target_prefix,
        "status": "ok",
        "error": "",
        "n_trials": int(X.shape[0]),
        "n_units": int(X.shape[1]),
        "n_contrasts": len(masks),
        "contrast_counts": json.dumps(contrast_count_dict(masks)),
        "seed": SEED,
        "min_trials": MIN_TRIALS,
        "alpha": ALPHA,
        "balanced_test_mse": fit["balanced_test_mse"],
        "balanced_test_r2": fit["balanced_test_r2"],
    }
    return summary, details


SUMMARY_FIELDS = [
    "eid",
    "pid",
    "target_prefix",
    "status",
    "error",
    "n_trials",
    "n_units",
    "n_contrasts",
    "contrast_counts",
    "seed",
    "min_trials",
    "alpha",
    "balanced_test_mse",
    "balanced_test_r2",
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
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_json = (
        repo_root
        / "results"
        / "region_scan"
        / args.target_prefix
        / f"{args.target_prefix}_subjects_by_lab.json"
    )
    output_dir = (
        repo_root
        / "results"
        / "decoder_0_100ms"
        / args.target_prefix
    )
    details_dir = output_dir / "details"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.target_prefix}_decoder_0_100ms_summary.csv"

    eids = build_eids(input_json)
    print(f"target_prefix={args.target_prefix}")
    print(f"input_json={input_json}")
    print(f"n_sessions={len(eids)}")
    print(f"output_dir={output_dir}")

    ONE.setup(
        base_url="https://openalyx.internationalbrainlab.org",
        silent=True,
    )
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
            summary, details = run_one_session(
                one=one,
                atlas=atlas,
                eid=eid,
                target_prefix=args.target_prefix,
            )
            details_path = details_dir / f"{eid}_decoder_0_100ms.pkl"
            with open(details_path, "wb") as f:
                pickle.dump(details, f)
            print(
                f"  ok: n_trials={summary['n_trials']}, "
                f"n_units={summary['n_units']}, "
                f"R2={summary['balanced_test_r2']:.6f}"
            )
        except Exception as exc:
            summary = {
                "eid": eid,
                "pid": "",
                "target_prefix": args.target_prefix,
                "status": "failed",
                "error": repr(exc),
                "n_trials": "",
                "n_units": "",
                "n_contrasts": "",
                "contrast_counts": "",
                "seed": SEED,
                "min_trials": MIN_TRIALS,
                "alpha": ALPHA,
                "balanced_test_mse": "",
                "balanced_test_r2": "",
            }
            print(f"  failed: {summary['error']}")

        rows.append(summary)
        write_summary(rows, summary_path)

    n_ok = sum(row["status"] == "ok" for row in rows)
    print(f"done: {n_ok}/{len(rows)} sessions succeeded")
    print(f"summary={summary_path}")

if __name__ == "__main__":
    main()