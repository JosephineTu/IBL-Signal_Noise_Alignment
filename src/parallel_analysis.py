# parallel_analysis.py
import numpy as np
from alignment_metrics import compute_noise_covariance, get_eigenspectrum

def find_K_pa(R,pos_mask, neg_mask, num_iter=500):

    rng = np.random.default_rng(seed=42)
    n_neurons = R.shape[1]
    noise_mask = pos_mask | neg_mask
    C_orig = compute_noise_covariance(R, noise_mask)
    _, _, observed_eigenvalues = get_eigenspectrum(C_orig)
    observed_eigenvalues = np.asarray(observed_eigenvalues, dtype=float)
    null_eigenvalues = np.empty(
        (num_iter, n_neurons),
        dtype=float,
    )
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)
    for iteration in range(num_iter):
        R_null = R.copy()
        for neuron_idx in range(n_neurons):
            pos_draw = rng.permutation(pos_idx)
            neg_draw = rng.permutation(neg_idx)
            R_null[pos_idx, neuron_idx] = (R[pos_draw, neuron_idx])
            R_null[neg_idx, neuron_idx] = (R[neg_draw, neuron_idx])
        C_null = compute_noise_covariance(R_null, noise_mask)
        _, _, eigenvalues = get_eigenspectrum(C_null)
        null_eigenvalues[iteration] = eigenvalues

    null_p95 = np.percentile(
        null_eigenvalues,
        95,
        axis=0,
    )
    k_pa = 0
    while (k_pa < n_neurons and observed_eigenvalues[k_pa]
        > null_p95[k_pa]):
        k_pa += 1
    return k_pa
        



            
            



