# directional_r2_profile.py
from random import sample

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.covariance import LedoitWolf
from .decoder import make_balanced_weights
from .alignment_metrics import top_noise_subspace

def direction_r2(z_train, y_train, z_test, y_test, train_weights, test_weights):
    model = LinearRegression()
    model.fit(z_train.shape(-1,1), y_train, sample_weight=train_weights)
    y_pred = model.predict(z_test.reshape(-1,1))
    return r2_score(y_test, y_pred, sample_weight=test_weights)

def directional_r2_profile(X_train_std, y_train, X_test_std, y_test, ridge_test_r2, noise_covariance, seed, n_random=500):
    rng = np.random.default_rng(seed)
    train_weights = make_balanced_weights(y_train)
    test_weights = make_balanced_weights(y_test)
    w_r2 = ridge_test_r2

    U_1, lambda_1 = top_noise_subspace(noise_covariance, k=1)
    top1_r2 = direction_r2(X_train_std @ U_1, y_train, X_test_std @ U_1, y_test, train_weights, test_weights)

    n_features = X_train_std.shape[1]
    null_r2 = np.empty(n_random)
    for j in range(n_random):
        # Haar random vector for null noise eigenvector
        u_rand = rng.normal(size=n_features)
        u_rand /= np.linalg.norm(u_rand + 1e-12)
        null_r2[j] = direction_r2(X_train_std@u_rand, y_train, 
                                  X_test_std@u_rand, y_test, train_weights, test_weights)

    return{
        "top1_eigvec": U_1,
        "w_r2": w_r2,
        "noise_top1_r2": top1_r2,
        "null_r2": null_r2,
    }
