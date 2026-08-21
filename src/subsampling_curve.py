# subsampling_curve.py
import numpy as np
from .decoder import make_train_test_sets, ridge_regression
from scipy import stats

from sklearn.model_selection import StratifiedKFold

def build_labeled_trials(X, masks, min_trials=10):
    X_parts, y_parts = [], []
    for contrast, mask in masks.items():
        trial_idx = np.flatnonzero(mask)
        if len(trial_idx) < min_trials:
            raise ValueError(
                f"contrast {contrast} has {len(trial_idx)} trials < {min_trials}"
            )
        X_parts.append(X[trial_idx])
        y_parts.append(np.full(len(trial_idx), float(contrast), dtype=float))
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def get_test_mse_cv(X, ints, masks, model, seed, num_samples=20, n_splits=5, min_trials=10):
    mse_results = {}
    mse_std_results = {}
    for i in ints:
        i = int(i)
        samples = sample_neurons(X, i, seed, num_samples=num_samples)
        loss_list = []
        for s, sample in enumerate(samples):
            X_all, y_all = build_labeled_trials(sample, masks, min_trials=min_trials)
            _, y_strat = np.unique(y_all, return_inverse=True)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + s)
            fold_mses = []
            for train_idx, test_idx in skf.split(X_all, y_strat):
                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_test, y_test = X_all[test_idx], y_all[test_idx]
                fold_mse, _, _ = ridge_regression(
                    X_train, y_train, X_test, y_test, model
                )
                fold_mses.append(fold_mse)
            loss_list.append(np.mean(fold_mses))
        mse_results[i] = np.mean(loss_list)
        mse_std_results[i] = np.std(loss_list)
    return mse_results, mse_std_results

def sample_neurons(X, N, seed, num_samples=20):
    rng = np.random.default_rng(seed=seed)
    samples = []
    n_neurons = X.shape[1]
    for s in range(num_samples):
        X_sampled = np.empty((X.shape[0], N), dtype=float)
        indices = rng.choice(n_neurons, N, replace=False)
        X_sampled = X[:, indices]
        samples.append(X_sampled)
    return samples

def log_spaced_neuron_counts(n_total, k, n_min=5):
    if n_total < n_min:
        raise ValueError(f"n_total={n_total} is below the floor n_min={n_min}")
    max_distinct = n_total - n_min + 1
    n_points = max(1, min(k, max_distinct))
    if n_points == 1:
        return [n_total]
    raw = np.geomspace(n_min, n_total, num=n_points)
    ints = sorted(set(int(round(x)) for x in raw))
    if len(ints) < n_points:
        used = set(ints)
        candidates = sorted(
            (n for n in range(n_min, n_total + 1) if n not in used),
            reverse=True,
        )
        for n in candidates:
            if len(ints) >= n_points:
                break
            ints.append(n)
        ints = sorted(ints)
    return ints

def get_test_mse(X, ints, samples, masks, model, seed, num_samples=20):
    mse_results = {}
    mse_std_results = {}
    for i in ints:
        i = int(i)
        samples = sample_neurons(X, i, seed, num_samples=num_samples)
        loss_list = []
        s=0
        for sample in samples:
            X_train, y_train, X_test, y_test = make_train_test_sets(masks, sample, seed=seed+s,)
            balanced_test_mse, _, _ = ridge_regression(X_train, y_train, X_test, y_test, model)
            loss_list.append(balanced_test_mse)
            s += 1
        mean_mse = np.mean(loss_list)
        std_mse = np.std(loss_list)
        mse_results[i] = mean_mse
        mse_std_results[i] = std_mse
    return mse_results, mse_std_results

def fit_information_limiting_intercept(mse_results, mse_std_results=None, num_samples=None):
    Ns = np.asarray(sorted(mse_results.keys()), dtype=float)
    if len(Ns) < 3:
        raise ValueError("At least 3 points are required to fit the information limiting curve.")
    x = 1.0 / Ns
    y = np.asarray([mse_results[n] for n in Ns])

    if mse_std_results is not None:
        if num_samples is None:
            raise ValueError("num_samples must be provided when mse_std_results is given.")
        if isinstance(num_samples, dict):
            n_rep = np.asarray([num_samples[n] for n in Ns])
        else:
            n_rep = np.full(len(Ns), float(num_samples))
        std = np.array([mse_std_results[n] for n in Ns])
        se_of_mean = std / np.sqrt(n_rep)
        weights = 1.0 / (se_of_mean ** 2)
    else:
        weights = np.ones(len(Ns))

    X_design = np.column_stack([np.ones_like(x), x])
    W = np.diag(weights)
    XtWX = X_design.T @ W @ X_design
    XtWy = X_design.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    intercept, slope = beta
    resid = y - X_design @ beta
    dof = len(y) - 2
    if dof < 1:
        raise ValueError("Not enough degrees of freedom to compute standard errors.")
    sigma_sq = np.sum(weights * resid ** 2) / dof
    cov_beta = sigma_sq * np.linalg.inv(XtWX)
    se_intercept = np.sqrt(cov_beta[0, 0])

    t_stat = intercept / se_intercept
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "se_intercept": float(se_intercept),
        "p_value": float(p_value),
        "n_points": int(len(Ns)),
    }


    




        



    















    






