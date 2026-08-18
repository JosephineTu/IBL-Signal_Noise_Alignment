# decoder.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def make_train_test_sets (masks, X, seed, min_trials=10):
    X_train_parts = []
    y_train_parts = []
    X_test_parts = []
    y_test_parts = []
    rng = np.random.default_rng(seed=seed)

    for contrast, mask in masks.items():
        trial_idx = np.flatnonzero(mask)
        n_trials = len(trial_idx)
        if n_trials < min_trials:
            raise ValueError(
                f'contrast {contrast} has {n_trials} trials < {min_trials}'
            )

        trial_idx = rng.permutation(trial_idx)
        split_idx = n_trials // 2
        train_idx = trial_idx[:split_idx]
        test_idx = trial_idx[split_idx:]

        X_train_parts.append(X[train_idx])
        y_train_parts.append(np.full(len(train_idx), float(contrast), dtype=float))
        X_test_parts.append(X[test_idx])
        y_test_parts.append(np.full(len(test_idx), float(contrast), dtype=float))

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    train_order = rng.permutation(len(X_train))
    test_order = rng.permutation(len(X_test))

    X_train = X_train[train_order]
    X_test = X_test[test_order]
    y_train = y_train[train_order]
    y_test = y_test[test_order]

    return X_train, y_train, X_test, y_test

def make_balanced_weights(y):
    weights = np.empty(len(y), dtype=float)
    contrasts = np.unique(y)
    n_contrasts = len(contrasts)
    for c in contrasts:
        mask = (y == c)
        weights[mask] = len(y) / (n_contrasts * mask.sum())
    return weights

def ridge_regression(X_train, y_train, X_test, y_test, model):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    train_weights = make_balanced_weights(y_train)
    test_weights = make_balanced_weights(y_test)
    model.fit(X_train_std, y_train, sample_weight=train_weights)
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)

    contrast_losses = []

    for c in np.unique(y_test):
        mask = (y_test == c)
        contrast_mse = np.mean((y_test[mask] - y_test_pred[mask]) ** 2)
        contrast_losses.append(contrast_mse)
    balanced_test_mse = np.mean(contrast_losses)
    test_r2 = r2_score(y_test, y_test_pred, sample_weight=test_weights)
    print(f"balanced test MSE: {balanced_test_mse}")

    w = model.coef_.copy()

    return balanced_test_mse, test_r2, w








