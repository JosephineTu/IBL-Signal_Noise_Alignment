import numpy as np
from .trial_selection import get_high_masks, get_pos_neg_masks
from .alignment_metrics import compute_signal_axis, noise_residuals_by_condition, compute_noise_covariance, top_noise_subspace, compute_cosine_similarity

# Haar null
def k_constrained_haar_null(
    k_pa,
    u_sig,
    C,
    num_iter=500,
    eps=1e-12,
):
    rng = np.random.default_rng(seed=42)
    u_sig = np.asarray(u_sig, float)
    u_sig = u_sig / (np.linalg.norm(u_sig) + eps)
    U_k, evals_k = top_noise_subspace(C, k=k_pa,)

    coeffs = U_k.T @ u_sig

    # Normalize after projection into top-K
    projection_norm = np.linalg.norm(coeffs)

    coeffs_k = coeffs / (
        projection_norm + eps
    )
    q_k = coeffs_k ** 2
    q_block = q_k[1:]
    lambda_block = evals_k[1:]

    T_obs = float(
        np.sum(q_block / lambda_block))

    Q_block = float(np.sum(q_block))

    null_Ts = np.empty(
        num_iter,
        dtype=float,
    )

    for iteration in range(num_iter):
        g = rng.standard_normal(size=k_pa - 1)
        r = g / (np.linalg.norm(g) + eps)

        # r**2 sums to 1; multiply by Q_block
        # to preserve total non-top signal mass.
        q_null_block = (
            Q_block * r ** 2
        )

        null_Ts[iteration] = np.sum(
            q_null_block / lambda_block
        )

    p_lower = (
        1 + np.sum(null_Ts <= T_obs)
    ) / (num_iter + 1)

    p_upper = (
        1 + np.sum(null_Ts >= T_obs)
    ) / (num_iter + 1)

    return {
        "T_obs": T_obs,
        "null_Ts": null_Ts,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
    }


# pairing-permutation null
def pairing_permutation_null(
    k_pa,
    u_sig,
    C,
    num_iter=500,
    eps=1e-12,
):
    rng = np.random.default_rng(seed=42)

    u_sig = np.asarray(u_sig, float)
    u_sig = u_sig / (
        np.linalg.norm(u_sig) + eps
    )

    U_k, evals_k = top_noise_subspace(
        C,
        k=k_pa,
    )

    coeffs = U_k.T @ u_sig

    projection_norm = np.linalg.norm(coeffs)

    coeffs_k = coeffs / (
        projection_norm + eps
    )

    q_k = coeffs_k ** 2

    q_block = q_k[1:]
    lambda_block = evals_k[1:]

    T_obs = float(np.sum(q_block / lambda_block))
    null_Ts = np.empty(
        num_iter,
        dtype=float,
    )
    for iteration in range(num_iter):
        q_perm = rng.permutation(q_block)
        null_Ts[iteration] = np.sum(q_perm / lambda_block)

    p_lower = (
        1 + np.sum(null_Ts <= T_obs)
    ) / (num_iter + 1)

    p_upper = (
        1 + np.sum(null_Ts >= T_obs)
    ) / (num_iter + 1)

    return {
        "T_obs": T_obs,
        "null_Ts": null_Ts,
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
    }











