import numpy as np

from .alignment_metrics import top_noise_subspace


def _corrected_mode_energies(
    k_pa,
    signal_axis,
    C,
    n_pos,
    n_neg,
    eps=1e-12,
):
    """Estimate raw and finite-trial-corrected signal energy by mode."""
    signal_axis = np.asarray(signal_axis, float)
    C = np.asarray(C, float)

    U_k, evals_k = top_noise_subspace(C, k=k_pa)
    coeffs = U_k.T @ signal_axis

    raw_energy = coeffs**2
    mean_noise_scale = 1.0 / n_pos + 1.0 / n_neg
    energy_bias = mean_noise_scale * evals_k
    corrected_energy = raw_energy - energy_bias

    return {
        "U_k": U_k,
        "evals_k": evals_k,
        "coeffs": coeffs,
        "raw_energy": raw_energy,
        "energy_bias": energy_bias,
        "corrected_energy": corrected_energy,
        "mean_noise_scale": float(mean_noise_scale),
    }


def k_constrained_haar_null(
    k_pa,
    signal_axis,
    C,
    n_pos,
    n_neg,
    num_iter=500,
    eps=1e-12,
):
    """Conditional Haar null after correcting finite-trial signal energy.

    As in the original null, mode 1 is excluded and the total non-top
    signal energy is held fixed. Each null draw randomizes its direction
    inside modes 2..K, adds the expected finite-trial mean-estimation
    noise, and then applies the same mode-wise bias correction as the
    observed statistic.
    """
    rng = np.random.default_rng(seed=42)

    spectral = _corrected_mode_energies(
        k_pa=k_pa,
        signal_axis=signal_axis,
        C=C,
        n_pos=n_pos,
        n_neg=n_neg,
        eps=eps,
    )

    lambda_block = spectral["evals_k"][1:]
    raw_energy_block = spectral["raw_energy"][1:]
    bias_block = spectral["energy_bias"][1:]
    corrected_energy_block = spectral["corrected_energy"][1:]

    T_obs_raw = float(
        np.sum(raw_energy_block / (lambda_block + eps))
    )
    T_obs_corrected = float(
        np.sum(corrected_energy_block / (lambda_block + eps))
    )

    total_corrected_block_energy = float(
        np.sum(corrected_energy_block)
    )

    if total_corrected_block_energy <= 0:
        return {
            "status": "nonpositive_corrected_block_energy",
            "T_obs_raw": T_obs_raw,
            "T_obs_corrected": T_obs_corrected,
            "null_Ts_corrected": np.full(num_iter, np.nan),
            "p_lower_corrected": np.nan,
            "p_upper_corrected": np.nan,
            "raw_energy_block": raw_energy_block,
            "energy_bias_block": bias_block,
            "corrected_energy_block": corrected_energy_block,
            "total_corrected_block_energy": total_corrected_block_energy,
            "mean_noise_scale": spectral["mean_noise_scale"],
        }

    null_Ts_corrected = np.empty(num_iter, dtype=float)

    for iteration in range(num_iter):
        g = rng.standard_normal(size=k_pa - 1)
        r = g / (np.linalg.norm(g) + eps)

        true_null_coeffs = (
            np.sqrt(total_corrected_block_energy) * r
        )

        mean_estimation_noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(bias_block),
        )
        estimated_null_coeffs = (
            true_null_coeffs + mean_estimation_noise
        )

        corrected_null_energy = (
            estimated_null_coeffs**2 - bias_block
        )
        null_Ts_corrected[iteration] = np.sum(
            corrected_null_energy / (lambda_block + eps)
        )

    p_lower_corrected = (
        1 + np.sum(null_Ts_corrected <= T_obs_corrected)
    ) / (num_iter + 1)
    p_upper_corrected = (
        1 + np.sum(null_Ts_corrected >= T_obs_corrected)
    ) / (num_iter + 1)

    return {
        "status": "ok",
        "T_obs_raw": T_obs_raw,
        "T_obs_corrected": T_obs_corrected,
        "null_Ts_corrected": null_Ts_corrected,
        "null_mean_corrected": float(
            np.mean(null_Ts_corrected)
        ),
        "p_lower_corrected": float(p_lower_corrected),
        "p_upper_corrected": float(p_upper_corrected),
        "raw_energy_block": raw_energy_block,
        "energy_bias_block": bias_block,
        "corrected_energy_block": corrected_energy_block,
        "total_corrected_block_energy": total_corrected_block_energy,
        "mean_noise_scale": spectral["mean_noise_scale"],
    }


def pairing_permutation_null(
    k_pa,
    signal_axis,
    C,
    n_pos,
    n_neg,
    num_iter=500,
    eps=1e-12,
):
    """Pair corrected signal energies with the reliable eigenvalues."""
    rng = np.random.default_rng(seed=42)

    spectral = _corrected_mode_energies(
        k_pa=k_pa,
        signal_axis=signal_axis,
        C=C,
        n_pos=n_pos,
        n_neg=n_neg,
        eps=eps,
    )

    lambda_block = spectral["evals_k"][1:]
    raw_energy_block = spectral["raw_energy"][1:]
    bias_block = spectral["energy_bias"][1:]
    corrected_energy_block = spectral["corrected_energy"][1:]

    T_obs_raw = float(
        np.sum(raw_energy_block / (lambda_block + eps))
    )
    T_obs_corrected = float(
        np.sum(corrected_energy_block / (lambda_block + eps))
    )

    null_Ts_corrected = np.empty(num_iter, dtype=float)

    for iteration in range(num_iter):
        corrected_energy_perm = rng.permutation(
            corrected_energy_block
        )
        null_Ts_corrected[iteration] = np.sum(
            corrected_energy_perm / (lambda_block + eps)
        )

    p_lower_corrected = (
        1 + np.sum(null_Ts_corrected <= T_obs_corrected)
    ) / (num_iter + 1)
    p_upper_corrected = (
        1 + np.sum(null_Ts_corrected >= T_obs_corrected)
    ) / (num_iter + 1)

    return {
        "status": "ok",
        "T_obs_raw": T_obs_raw,
        "T_obs_corrected": T_obs_corrected,
        "null_Ts_corrected": null_Ts_corrected,
        "null_mean_corrected": float(
            np.mean(null_Ts_corrected)
        ),
        "p_lower_corrected": float(p_lower_corrected),
        "p_upper_corrected": float(p_upper_corrected),
        "raw_energy_block": raw_energy_block,
        "energy_bias_block": bias_block,
        "corrected_energy_block": corrected_energy_block,
        "mean_noise_scale": spectral["mean_noise_scale"],
    }











