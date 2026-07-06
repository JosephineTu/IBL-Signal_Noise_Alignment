import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from itertools import combinations
from trial_selection import make_contrast_pair_condition_masks, get_pos_neg_masks

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
    return u_sig

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
    lw = LedoitWolf(store_precision=False, assume_centered=True)
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
        mask_a = condition_masks[cond_a]
        mask_b = condition_masks[cond_b]
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

def random_subspace_similarity(X, condition_masks, k=3, n_iter=100, eps=1e-12):
    similarities=[]
    R, _, residual_mask = noise_residuals_by_condition(X, condition_masks)
    valid_idx = np.flatnonzero(residual_mask)
    for i in range(n_iter):
        idx = np.random.choice(valid_idx, size=X.shape[0]//3, replace=False)
        mask_1 = np.zeros(X.shape[0], dtype=bool)
        mask_2 = np.zeros(X.shape[0], dtype=bool)
        half = len(idx) // 2
        mask_1[idx[:half]] = True
        mask_2[idx[half:]] = True
        C_1 = compute_noise_covariance(R, mask_1)
        C_2 = compute_noise_covariance(R, mask_2)
        U_1, _ = top_noise_subspace(C_1, k=k)
        U_2, _ = top_noise_subspace(C_2, k=k)
        similarity = subspace_similarity(U_1, U_2, eps=eps)
        similarities.append(similarity)
    return np.mean(similarities)

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

def summarize_contrast_pair_axes(axes, u_global, eps=1e-12):
    keys = list(axes.keys())
    out_1 = {}
    for key_a, key_b in combinations(keys, 2):
        u_a = axes[key_a]
        u_b = axes[key_b]
        cosine_angle = np.dot(u_a, u_b) / (np.linalg.norm(u_a) * np.linalg.norm(u_b) + eps)
        pair = (key_a, key_b)
        out_1[pair] = {
            'cosine_angle': float(cosine_angle),
            'abs_consine_angle': float(np.abs(cosine_angle)),
        }

    u_global = np.asarray(u_global, float)
    out_2 = {}
    for pair_key, u in axes.items():
        u = np.asarray(u, float)
        cosine_global = np.dot(u, u_global) / (np.linalg.norm(u) * np.linalg.norm(u_global) + eps)
        out_2[pair_key] = {
            'cosine_global': float(cosine_global),
            'abs_cosine_global': float(np.abs(cosine_global)),
        }
    return out_1, out_2
