import numpy as np
import pandas as pd


def run_timebinned_metric(
    fr_tb,
    times,
    metric_fn,
    *,
    min_units=5,
    active_eps=1e-10,
    **metric_kwargs,
):
    fr_tb = np.asarray(fr_tb, float)  # (n_trials, n_bins, n_units)
    times = np.asarray(times, float)  # (n_bins,)

    if fr_tb.ndim != 3:
        raise ValueError(f"fr_tb must be 3D, got shape {fr_tb.shape}")

    n_trials, n_bins, n_units = fr_tb.shape
    if times.shape != (n_bins,):
        raise ValueError(
            f"times must have shape ({n_bins},), got {times.shape}"
        )

    rows = []
    full_outputs = {}

    for b in range(n_bins):
        X_bin = fr_tb[:, b, :]
        active_mask = np.nanstd(X_bin, axis=0) > active_eps
        n_active_units = int(np.sum(active_mask))

        row = {
            "bin_idx": int(b),
            "time": float(times[b]),
            "n_trials": int(n_trials),
            "n_units": int(n_units),
            "n_active_units": n_active_units,
        }

        if n_active_units < min_units:
            print(
                f"Skipping bin {b}: active units "
                f"{n_active_units} < {min_units}",
                flush=True,
            )
            row["status"] = "insufficient_active_units"
            full_outputs[b] = None
            rows.append(row)
            continue

        # Do not slice X_bin by active_mask here. Keeping the same neuron
        # coordinates across bins is required for comparing axis rotations.
        out = metric_fn(X_bin, **metric_kwargs)
        full_outputs[b] = out
        row["status"] = "ok"

        for key, val in out.items():
            if np.isscalar(val):
                row[key] = float(val)

        rows.append(row)

    time_df = pd.DataFrame(rows)
    return time_df, full_outputs

