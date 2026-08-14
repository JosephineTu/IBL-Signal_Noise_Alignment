import numpy as np
from .trial_selection import get_high_masks, get_pos_neg_masks
from .alignment_metrics import compute_signal_axis, noise_residuals_by_condition, compute_noise_covariance, top_noise_subspace

def sample_conditional_signal_axis(u_sig, noise_a, rng, eps=1e-12):
    u_sig = np.asarray(u_sig)
    noise_a = np.asarray(noise_a)
    a = noise_a / np.linalg.norm(noise_a)
    alpha = float(a @ u_sig)
    q = alpha ** 2
    g = rng.standard_normal(size=u_sig.shape)
    z = g - (g @ a) * a
    z_norm = z / (np.linalg.norm(z) + eps)
    u_null = alpha * a + np.sqrt(1 - q) * z_norm
    return u_null, q


def conditional_fisher_null(noise_a, signal_axis, C, num_iter=500):
    signal_axis = np.asarray(signal_axis)
    C = np.asarray(C)
    w_obs = np.linalg.solve(C, signal_axis)
    J_obs = signal_axis.T @ w_obs
    u_sig = signal_axis / np.linalg.norm(signal_axis)
    d = np.linalg.norm(signal_axis)
    rng = np.random.default_rng(seed=42)
    null_Js = []
    for i in range(num_iter):
        u_null, _ = sample_conditional_signal_axis(u_sig, noise_a, rng=rng, eps=1e-12)
        w_null = np.linalg.solve(C, u_null)
        J_null = (d ** 2) * (u_null.T @ w_null)
        null_Js.append(J_null)
    null_Js = np.array(null_Js)
    expected_null_J = np.mean(null_Js)
    compensated_logratio = float(np.log(J_obs / expected_null_J))
    return {"J_obs": J_obs, "expected_null_J": expected_null_J, "compensated_logratio": compensated_logratio, "null_Js": null_Js}

def compute_conditional_fisher_null(X, signed_contrast, min_trials=5):
    X = np.asarray(X)
    high_mask = get_high_masks(signed_contrast, min_trials=min_trials)
    pos_mask, neg_mask = get_pos_neg_masks(signed_contrast, high_mask=high_mask, min_trials=min_trials)
    u_sig, signal_axis = compute_signal_axis(X, pos_mask, neg_mask)
    condition_masks = {"pos": pos_mask, "neg": neg_mask}
    R, condition_means, residual_mask = noise_residuals_by_condition(X, condition_masks)
    noise_mask = residual_mask & (pos_mask | neg_mask)
    C = compute_noise_covariance(R, noise_mask)
    U_1, _ = top_noise_subspace(C, k=1)
    noise_a = U_1[:, 0]
    results = conditional_fisher_null(noise_a, signal_axis, C, num_iter=500)
    return results










