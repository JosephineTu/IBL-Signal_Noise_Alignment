#time_bin.py
import numpy as np
import pandas as pd

def run_timebinned_metric(fr_tb, times, metric_fn, **metric_kwargs):
    fr_tb = np.asarray(fr_tb, float) #(n_trials, n_bins, n_units)
    times = np.asarray(times, float) #(n_bins, )
    n_trials, n_bins, n_units = fr_tb.shape
    rows = []
    full_outputs = {}
    for b in range(n_bins):
        X_bin = fr_tb[:, b, :]
        out = metric_fn(X_bin, **metric_kwargs)
        full_outputs[b] = out
        row = {
            'bin_idx': int(b),
            'time': float(times[b]),
            'n_trials': int(n_trials),
            'n_units': int(n_units),
        }
        for key, val in out.items():
            if np.isscalar(val):
                row[key] = float(val)
        rows.append(row)
    time_df = pd.DataFrame(rows)
    return time_df, full_outputs

