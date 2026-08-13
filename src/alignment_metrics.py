import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from itertools import combinations
from .trial_selection import make_contrast_pair_condition_masks, get_high_masks, get_pos_neg_masks

def enforce_sym(C):
    C = np.asarray(C, dtype=float)
    return 0.5 * (C + C.T)

def compute_signal_axis(X, pos_mask, neg_mask, min_trials=5, eps=1e-12):
    if np.sum(pos_mask) < min_trials or np.sum(neg_mask) < min_trials:
        raise RuntimeError("Not enough trials to compute signal axis")
    mu_pos = np.mean(X[pos_mask], axis=0)
    mu_neg = np.mean(X[neg_mask], axis=0)
    # how far apart are the two means
    signal_axis = mu_pos - mu_neg
    u_sig = signal_axis / (np.linalg.norm(signal_axis) + eps)
    return u_sig, signal_axis

def noise_residuals_by_condition(X, condition_masks):
    X = np.asarray(X, float)
    R = np.full_like(X, np.nan, dtype=float)
    condition_means = {}
    residual_mask = np.zeros(X.shape[0], dtype=bool)

    for cond, mask in condition_masks.items():
        mask = np.asarray(mask, bool)
        condition_means[cond] = np.mean(X[mask], axis=0)
        residual_mask[mask] = True
        R[mask] = X[mask] - condition_means[cond]
    return R, condition_means, residual_mask

def compute_noise_covariance(R, residual_mask):
    R = np.asarray(R, float)
    R = R[residual_mask]
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(R)
    C = lw.covariance_
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = enforce_sym(C)
    return C

def top_noise_subspace(C, k=3):
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    vals = np.clip(eigenvalues[order], a_min=0, a_max=None)
    vecs = eigenvectors[:, order]
    k_sel = min(k, vecs.shape[1])
    U_k = vecs[:, :k_sel].copy()
    evals_k = vals[:k_sel].astype(float)
    for i in range(U_k.shape[1]):
        j = np.argmax(np.abs(U_k[:, i]))
        if U_k[j,i] < 0:
            U_k[:, i] *= -1
    return U_k, evals_k

def subspace_similarity(U1, U2, eps=1e-12):
    U1 = np.asarray(U1, float)
    U2 = np.asarray(U2, float)
    k = min(U1.shape[1], U2.shape[1])
    if k < 1: 
        return np.nan
    U1 = U1[:, :k]
    U2 = U2[:, :k]
    return float(np.linalg.norm(U1.T @ U2, ord='fro') ** 2 / (k + eps))

def condition_noise_subspaces(X, condition_masks, k=3, min_trials=10, eps=1e-12):
    X = np.asarray(X, float)
    R, condition_means, residual_mask = noise_residuals_by_condition(X, condition_masks)
    noise_subspace_similarities = {}
    for cond_a, cond_b in combinations(condition_masks.keys(), 2):
        mask_a = condition_masks[cond_a] & residual_mask
        mask_b = condition_masks[cond_b] & residual_mask
        if np.sum(mask_a) < min_trials or np.sum(mask_b) < min_trials:
            continue
        C_a = compute_noise_covariance(R, mask_a)
        U_a, _ = top_noise_subspace(C_a, k=k)
        C_b = compute_noise_covariance(R, mask_b)
        U_b, _ = top_noise_subspace(C_b, k=k)
        similarity = subspace_similarity(U_a, U_b, eps=eps)
        noise_subspace_similarities[(cond_a, cond_b)] = similarity
    vals = np.asarray(list(noise_subspace_similarities.values()), float)
    mean_similarity = float(np.nanmean(vals)) if vals.size else np.nan
    min_similarity = float(np.nanmin(vals)) if vals.size else np.nan
    trial_counts = {cond: int(np.sum(mask)) for cond, mask in condition_masks.items()}
    return noise_subspace_similarities, mean_similarity, min_similarity, trial_counts

def random_subspace_similarity(X, condition_masks, k=3, n_iter=100, min_trials=10,eps=1e-12, seed=0):
    rng = np.random.default_rng(seed)
    R, _, residual_mask = noise_residuals_by_condition(X, condition_masks)
    valid_idx = np.flatnonzero(residual_mask)
    samples = []
    cond_items = list(condition_masks.items())
    for _ in range(n_iter):
        sim_iter = []
        for cond_a, mask_a_real in cond_items:
            for cond_b, mask_b_real in cond_items:
                if cond_a >= cond_b:
                    continue
                idx_a = np.flatnonzero(mask_a_real & residual_mask)
                idx_b = np.flatnonzero(mask_b_real & residual_mask)
                pair_idx = np.concatenate([idx_a, idx_b])
                draw = rng.permutation(pair_idx)
                mask_a = np.zeros(X.shape[0], dtype=bool)
                mask_b = np.zeros(X.shape[0], dtype=bool)
                mask_a[draw[:len(idx_a)]] = True
                mask_b[draw[len(idx_a):]] = True
                C_a = compute_noise_covariance(R, mask_a)
                C_b = compute_noise_covariance(R, mask_b)
                U_a, _ = top_noise_subspace(C_a, k=k)
                U_b, _ = top_noise_subspace(C_b, k=k)
                sim_iter.append(subspace_similarity(U_a, U_b, eps=eps))
        if sim_iter:
            samples.append(np.nanmean(sim_iter))
    samples = np.asarray(samples, float)
    return {
        "samples": samples,
        "mean": float(np.nanmean(samples)),
        "std": float(np.nanstd(samples)),
        "p05": float(np.nanpercentile(samples, 5)),
        "p95": float(np.nanpercentile(samples, 95)),
    }

def compute_cosine_similarity(u1, u2):
    u1 = np.asarray(u1, float)
    u2 = np.asarray(u2, float)
    u1 = u1 / (np.linalg.norm(u1) + 1e-12)
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)
    return np.dot(u1, u2)

def compute_condition_mean_pca(X, condition_masks, n_components=3):
    X = np.asarray(X, float)
    _, condition_means, _ = noise_residuals_by_condition(X, condition_masks)
    valid_conds = [cond for cond in condition_masks.keys() if cond in condition_means.keys()]
    if len(valid_conds) < 2:
        return None, None
    condition_mean_matrix = np.array([condition_means[cond] for cond in valid_conds])
    n_components = min(n_components, len(valid_conds), condition_mean_matrix.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(condition_mean_matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    return components, explained_variance_ratio

def compute_contrast_pair_axes(X, trials, eps=1e-12):
    X = np.asarray(X, float)
    _, condition_masks = make_contrast_pair_condition_masks(trials)
    _, condition_means, _ = noise_residuals_by_condition(X, condition_masks)
    axes = {}
    for cond_a, cond_b in combinations(condition_masks.keys(), 2):
        mean_a = condition_means[cond_a]
        mean_b = condition_means[cond_b]
        delta = mean_a - mean_b
        norm = np.linalg.norm(delta)
        if norm < eps:
            continue
        u_axis = delta / (norm)
        axes[(cond_a, cond_b)] = u_axis
    return axes

def summarize_contrast_pair_axes(axes, u_global, min_contrast=0.5, eps=1e-12):
    matched_axes = {}
    for pair_key, u_axis in axes.items():
        cond_a, cond_b = pair_key
        left_a, right_a = cond_a
        left_b, right_b = cond_b
        a_is_left = (
            np.isclose(right_a, 0)
            and np.isclose(left_b, 0)
            and np.isclose(left_a, right_b)
        )

        b_is_left = (
            np.isclose(right_b, 0)
            and np.isclose(left_a, 0)
            and np.isclose(left_b, right_a)
        )
        if a_is_left:
            contrast = float(left_a)
        elif b_is_left:
            contrast = float(left_b)
        else:
            continue
        if contrast < min_contrast:
            continue
        matched_axes[contrast] = np.asarray(u_axis, float)
    out_1 = {}
    for contrast_a, contrast_b in combinations(sorted(matched_axes), 2):
        u_a = matched_axes[contrast_a]
        u_b = matched_axes[contrast_b]
        sim = compute_cosine_similarity(u_a, u_b)
        out_1[(contrast_a, contrast_b)] = {'abs_cosine': float(np.abs(sim))}
    u_global = np.asarray(u_global, float)
    out_2 = {}
    for contrast, u in matched_axes.items():
        sim = compute_cosine_similarity(u, u_global)
        out_2[contrast] = {'abs_cosine': float(np.abs(sim))}
    return out_1, out_2

def compute_noise_top1_variance(
    noise_covariance,
    n_components=1,
    eps=1e-12,):
    C = np.asarray(noise_covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    top_eigenvector = eigenvectors[:, -1]
    eigenvalues = np.clip(eigenvalues, 0, None)
    eigenvalues = eigenvalues[::-1]
    total_variance = np.sum(eigenvalues)
    if total_variance < eps:
        return np.full(n_components, np.nan)
    explained_variance_ratio = (
        eigenvalues[:n_components] / total_variance
    )
    return explained_variance_ratio, top_eigenvector

def compute_w_2d(u_sig, C, a, eps=1e-12, rank_tol=1e-8):
    u_sig = np.asarray(u_sig, float)
    C = np.asarray(C, float)
    a = np.asarray(a, float)
    C = enforce_sym(C)
    a = a / (np.linalg.norm(a) + eps)
    B = np.column_stack([u_sig, a])
    U, singular_values, _ = np.linalg.svd(B, full_matrices=False)
    rank = np.sum(singular_values > rank_tol)
    if rank < 2:
        return np.nan
    Q = U[:, :rank]
    C_2d = Q.T @ C @ Q
    d_2d = Q.T @ u_sig
    coeff = np.linalg.solve(C_2d, d_2d)
    w_2d = Q @ coeff
    w_2d = w_2d / (np.linalg.norm(w_2d) + eps)
    return w_2d

def compute_w_whitened(u_sig, C, eps=1e-12):
    w_whitened = np.linalg.solve(C + eps * np.eye(C.shape[0]), u_sig)
    w_whitened = w_whitened / (np.linalg.norm(w_whitened) + eps)
    return w_whitened

def signal_noise_alignment(X, signed_contrast, k=3, min_trials=5, eps=1e-12):
    X = np.asarray(X, float)
    high_mask = get_high_masks(signed_contrast, min_trials=min_trials)
    pos_mask, neg_mask = get_pos_neg_masks(signed_contrast, high_mask=high_mask, min_trials=min_trials)
    u_sig,_ = compute_signal_axis(X, pos_mask, neg_mask, min_trials=min_trials, eps=eps)
    condition_masks = {
        'pos': pos_mask,
        'neg': neg_mask,
    }
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    R, condition_means, residual_mask = noise_residuals_by_condition(X, condition_masks)
    noise_mask = pos_mask | neg_mask
    C = compute_noise_covariance(R, noise_mask)
    U_k, evals_k = top_noise_subspace(C, k=k)
    a = U_k[:, 0]
    overlap_topk = float(np.sum((U_k.T @ u_sig) ** 2))
    cosine2_top1 = float((a @ u_sig) ** 2)
    k_eff = U_k.shape[1]
    n_eff = U_k.shape[0]
    expected_random_overlap = k_eff / n_eff
    expected_random_cosine2 = 1 / n_eff
    pval, null_cosine_2 = null_signal_noise_alignment(a, u_sig, cosine2_top1)
    w_2d = compute_w_2d(u_sig, C, a)
    w_whitened = compute_w_whitened(u_sig, C)
    return{
        'n_pos': n_pos,
        'n_neg': n_neg,
        'overlap_topk': overlap_topk,
        'cosine2_top1': cosine2_top1,
        'expected_random_overlap': expected_random_overlap,
        'expected_random_cosine2': expected_random_cosine2,
        "k_eff": k_eff,
        "n_eff": n_eff,
        'noise_a': a,
        'u_sig': u_sig,
        'null_pval': pval,
        'null_cosine2': null_cosine_2,
        'w_2d': w_2d,
        'w_whitened': w_whitened,
    }

def null_signal_noise_alignment(a, u_sig, cosine2_top1, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    n_eff = a.shape[0]
    null_cosine2 = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        draw = rng.permutation(n_eff)
        a_null = a[draw]
        null_cosine2[i] = float((a_null @ u_sig) ** 2)
    p_val = (np.sum(null_cosine2 >= cosine2_top1) + 1) / (n_perm + 1)
    return p_val, null_cosine2

def get_eigenspectrum(C, eps=1e-12):
    C = np.array(C, float)
    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = np.clip(eigenvalues, 0, None)
    eigenvalues = eigenvalues[::-1]
    total_variance = np.sum(eigenvalues)
    if total_variance < eps:
        return (np.full_like(eigenvalues, np.nan))
    explained_variance_ratio = eigenvalues / total_variance
    pc12_ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > eps else np.nan
    return explained_variance_ratio, pc12_ratio, eigenvalues

def test_null_eigenspectrum(X, high_mask, pos_mask, neg_mask, n_iter=1000, eps=1e-12, seed=0):
    rng = np.random.default_rng(seed)
    condition_masks = {
        'pos': np.asarray(high_mask & pos_mask, dtype=bool),
        'neg': np.asarray(high_mask & neg_mask, dtype=bool),
    }
    R, _, residual_mask = noise_residuals_by_condition(X, condition_masks)
    residual_mask = np.asarray(residual_mask, dtype=bool)
    idx_pos = np.flatnonzero(condition_masks['pos'] & residual_mask)
    idx_neg = np.flatnonzero(condition_masks['neg'] & residual_mask)
    
    C_true = compute_noise_covariance(R, residual_mask)
    _, pc12_true, eigenspectrum = get_eigenspectrum(C_true, eps=eps)
    null_ratios = np.full(n_iter, np.nan)
    for iter in range(n_iter):
        R_null = R.copy()
        for idx in (idx_pos, idx_neg):
            for unit in range(R.shape[1]):
                permuted_idx = rng.permutation(idx)
                R_null[idx, unit] = R[permuted_idx, unit]
        C_null = compute_noise_covariance(R_null, residual_mask)
        _, null_ratios[iter],_ = get_eigenspectrum(C_null, eps=eps)
    valid = np.isfinite(null_ratios)
    p_val = (1 + np.count_nonzero(null_ratios[valid] >= pc12_true)) / (1 + np.sum(valid))
    return{
        'pc12_true': float(pc12_true),
        'null_ratios': null_ratios,
        'p_value': float(p_val),
        'null_mean': np.mean(null_ratios[valid]),
        'eigenspectrum': eigenspectrum,
    }

def null_a_similarity(a_pos, a_neg, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    a_pos = np.asarray(a_pos, float)
    a_neg = np.asarray(a_neg, float)
    a_pos = a_pos / np.linalg.norm(a_pos)
    a_neg = a_neg / np.linalg.norm(a_neg)
    observed = float((a_pos @ a_neg) ** 2)
    n_units = a_pos.shape[0]
    null_samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        draw = rng.permutation(n_units)
        a_neg_null = a_neg[draw]
        null_samples[i] = float((a_neg_null @ a_pos) ** 2)
    p_value = (1 + np.count_nonzero(null_samples >= observed)) / (1 + n_iter)
    return{
        'observed': observed,
        'samples': null_samples,
        'p_value': p_value,
    }






    





