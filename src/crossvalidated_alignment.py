"""Repeated split-half signal reliability and cross-fitted alignment.

Inputs are firing rates for one time bin (trials x units) and boolean masks for
the positive and negative stimulus trials.  The permutation null repeats the
entire split, signal-axis, covariance, and alignment pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.covariance import LedoitWolf


def _prepare(fr, pos_mask, neg_mask, k):
    fr = np.asarray(fr, float)
    if fr.ndim != 2 or fr.shape[1] < 2:
        raise ValueError(f"fr must be (trials, units), got {fr.shape}")
    if not np.all(np.isfinite(fr)):
        raise ValueError("fr contains NaN or infinite values")
    pos_mask = np.asarray(pos_mask, bool)
    neg_mask = np.asarray(neg_mask, bool)
    if pos_mask.shape != (len(fr),) or neg_mask.shape != (len(fr),):
        raise ValueError("pos_mask and neg_mask must match the trial dimension")
    if np.any(pos_mask & neg_mask):
        raise ValueError("pos_mask and neg_mask overlap")
    if pos_mask.sum() < 4 or neg_mask.sum() < 4:
        raise ValueError("each condition needs at least four trials")
    if k < 1:
        raise ValueError("k must be >= 1")
    selected = pos_mask | neg_mask
    y = np.where(pos_mask[selected], 1, -1)
    return fr[selected], y


def _split(y, rng):
    a, b = [], []
    for label in (-1, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        cut = len(idx) // 2
        a.extend(idx[:cut])
        b.extend(idx[cut:])
    return np.asarray(a, int), np.asarray(b, int)


def _delta(fr, y, idx):
    return fr[idx][y[idx] == 1].mean(0) - fr[idx][y[idx] == -1].mean(0)


def _noise_subspace(fr, y, idx, k, std_eps):
    x = fr[idx]
    labels = y[idx]
    residuals = np.empty_like(x)
    for label in (-1, 1):
        mask = labels == label
        residuals[mask] = x[mask] - x[mask].mean(0)
    keep = residuals.std(0) > std_eps
    if keep.sum() < max(2, k):
        raise ValueError("too few varying units for noise covariance")
    covariance = LedoitWolf(assume_centered=False).fit(residuals[:, keep]).covariance_
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return vectors[:, order[: min(k, vectors.shape[1])]], keep


def _overlap(d, subspace, eps):
    denominator = float(d @ d)
    if denominator <= eps:
        return np.nan
    projection = subspace.T @ d
    return float((projection @ projection) / denominator)


def _one_split(fr, y, k, rng, eps, std_eps):
    idx_a, idx_b = _split(y, rng)
    d_a = _delta(fr, y, idx_a)
    d_b = _delta(fr, y, idx_b)
    norm_a = np.linalg.norm(d_a)
    norm_b = np.linalg.norm(d_b)

    noise_a, keep_a = _noise_subspace(fr, y, idx_a, k, std_eps)
    noise_b, keep_b = _noise_subspace(fr, y, idx_b, k, std_eps)

    a_to_b = _overlap(d_a[keep_b], noise_b, eps)
    b_to_a = _overlap(d_b[keep_a], noise_a, eps)
    return {
        "signal_reliability": float((d_a @ d_b) / (norm_a * norm_b + eps)),
        "cv_signal_magnitude": float(d_a @ d_b),
        "crossfit_overlap": float(np.nanmean([a_to_b, b_to_a])),
        "overlap_a_to_b": a_to_b,
        "overlap_b_to_a": b_to_a,
    }


def _repeat(fr, y, k, n_splits, rng, eps, std_eps):
    rows = [_one_split(fr, y, k, rng, eps, std_eps) for _ in range(n_splits)]
    return {
        key: np.asarray([row[key] for row in rows], float)
        for key in rows[0]
    }


def _split_summary(values, prefix):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{name}": np.nan for name in ("mean", "median", "q05", "q95")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


def evaluate_crossvalidated_alignment(
    fr,
    pos_mask,
    neg_mask,
    *,
    k: int = 3,
    n_splits: int = 20,
    n_permutations: int = 99,
    null_splits: Optional[int] = None,
    seed: int = 0,
    eps: float = 1e-12,
    std_eps: float = 1e-10,
    return_split_samples: bool = False,
) -> Dict[str, Any]:
    """Evaluate one time bin.

    ``signal_reliability`` is mean cos(d_A, d_B).
    ``cv_signal_magnitude`` is mean d_A.T @ d_B.
    ``crossfit_overlap`` averages signal-A/noise-B and signal-B/noise-A.
    """
    fr, y = _prepare(fr, pos_mask, neg_mask, k)
    if n_splits < 1 or n_permutations < 1:
        raise ValueError("n_splits and n_permutations must be >= 1")
    null_splits = n_splits if null_splits is None else int(null_splits)

    observed = _repeat(
        fr, y, k, int(n_splits), np.random.default_rng(seed), eps, std_eps
    )
    obs_rel = float(np.nanmean(observed["signal_reliability"]))
    obs_mag = float(np.nanmean(observed["cv_signal_magnitude"]))
    obs_align = float(np.nanmean(observed["crossfit_overlap"]))

    rng = np.random.default_rng(seed + 1)
    null_rel = np.full(n_permutations, np.nan)
    null_mag = np.full(n_permutations, np.nan)
    null_align = np.full(n_permutations, np.nan)
    for permutation in range(n_permutations):
        permuted = _repeat(fr, rng.permutation(y), k, null_splits, rng, eps, std_eps)
        null_rel[permutation] = np.nanmean(permuted["signal_reliability"])
        null_mag[permutation] = np.nanmean(permuted["cv_signal_magnitude"])
        null_align[permutation] = np.nanmean(permuted["crossfit_overlap"])

    result = {
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == -1).sum()),
        "n_units": int(fr.shape[1]),
        "noise_subspace_k": int(k),
        "n_splits": int(n_splits),
        "n_permutations": int(n_permutations),
        "signal_reliability": obs_rel,
        "cv_signal_magnitude": obs_mag,
        "crossfit_overlap": obs_align,
        "signal_reliability_null_mean": float(np.nanmean(null_rel)),
        "cv_signal_magnitude_null_mean": float(np.nanmean(null_mag)),
        "crossfit_overlap_null_mean": float(np.nanmean(null_align)),
        "crossfit_alignment_improvement": float(obs_align - np.nanmean(null_align)),
        "signal_reliability_p_upper": float((1 + np.sum(null_rel >= obs_rel)) / (n_permutations + 1)),
        "cv_signal_magnitude_p_upper": float((1 + np.sum(null_mag >= obs_mag)) / (n_permutations + 1)),
        "crossfit_overlap_p_upper": float((1 + np.sum(null_align >= obs_align)) / (n_permutations + 1)),
        "signal_reliability_null": null_rel,
        "cv_signal_magnitude_null": null_mag,
        "crossfit_overlap_null": null_align,
    }
    result.update(_split_summary(observed["signal_reliability"], "signal_reliability_split"))
    result.update(_split_summary(observed["cv_signal_magnitude"], "cv_signal_magnitude_split"))
    result.update(_split_summary(observed["crossfit_overlap"], "crossfit_overlap_split"))
    if return_split_samples:
        result["split_samples"] = observed
    return result


def evaluate_timebinned_crossvalidated_alignment(
    fr_tb, pos_mask, neg_mask, **kwargs
) -> Dict[str, Any]:
    """Evaluate every bin of a (trials, bins, units) firing-rate tensor."""
    fr_tb = np.asarray(fr_tb, float)
    if fr_tb.ndim != 3:
        raise ValueError(f"fr_tb must be (trials, bins, units), got {fr_tb.shape}")
    base_seed = int(kwargs.pop("seed", 0))
    bins = []
    for bin_index in range(fr_tb.shape[1]):
        try:
            out = evaluate_crossvalidated_alignment(
                fr_tb[:, bin_index], pos_mask, neg_mask,
                seed=base_seed + 100_003 * bin_index, **kwargs
            )
            out.update(bin_index=bin_index, status="ok")
        except (ValueError, np.linalg.LinAlgError) as exc:
            out = {"bin_index": bin_index, "status": "failed", "reason": repr(exc)}
        bins.append(out)

    keys = (
        "signal_reliability", "cv_signal_magnitude", "crossfit_overlap",
        "crossfit_overlap_null_mean", "crossfit_alignment_improvement",
        "signal_reliability_p_upper", "cv_signal_magnitude_p_upper",
        "crossfit_overlap_p_upper",
    )
    result = {"bin_results": bins}
    for key in keys:
        result[f"{key}_ts"] = np.asarray([row.get(key, np.nan) for row in bins], float)
    return result