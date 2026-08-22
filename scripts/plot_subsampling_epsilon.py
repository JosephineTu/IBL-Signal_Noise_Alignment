"""
plot_subsampling_epsilon.py

Diagnostic plot for the neuron-subsampling curve pipeline
(run_subsampling_epsilon.py): reads every per-eid details pkl under
results/subsampling_epsilon/<target_prefix>/details/, and for each
session plots mean balanced_test_mse vs 1/N (the exact x used inside
fit_information_limiting_intercept), with the fitted
epsilon + slope*(1/N) line overlaid -- so a negative/positive slope or
a non-monotonic curve is visible directly, not just as a signed number
buried in the summary CSV.

No dependency on iblatlas/one/brainbox/src -- reads plain dicts out of
the pickles, so this can run anywhere matplotlib/numpy are available
(not just inside the `ibl` conda env on the cluster).

Usage:
  python scripts/plot_subsampling_epsilon.py --target-prefix VISp
      -> one grid figure, all sessions as small multiples
  python scripts/plot_subsampling_epsilon.py --target-prefix VISp --eid <eid>
      -> one large single-session figure
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "src").is_dir() else SCRIPT_DIR
)

DATA_COLOR = "#3F6FBF"   # muted blue -- observed per-N means
FIT_COLOR = "#D9622B"    # warm orange -- fitted line, reads clearly against blue
GRID_COLOR = "#D8D8D8"


def load_details(details_dir):
    paths = sorted(Path(details_dir).glob("*_subsampling_epsilon.pkl"))
    out = []
    for path in paths:
        with open(path, "rb") as f:
            out.append(pickle.load(f))
    return out


def plot_one_session(ax, details, label_axes=False):
    mse_results = details["mse_results"]
    mse_std_results = details["mse_std_results"]
    num_samples = details["num_samples"]

    Ns = np.array(sorted(mse_results.keys()), dtype=float)
    x = 1.0 / Ns
    y = np.array([mse_results[n] for n in Ns])
    std = np.array([mse_std_results[n] for n in Ns])
    se = std / np.sqrt(num_samples)

    ax.errorbar(
        x, y, yerr=se, fmt="o", ms=4, color=DATA_COLOR, ecolor=DATA_COLOR,
        elinewidth=1, capsize=2, alpha=0.9, zorder=3,
        label="mean MSE(N) ± SE",
    )

    x_line = np.linspace(0, x.max() * 1.05, 100)
    y_line = details["epsilon"] + details["slope"] * x_line
    ax.plot(
        x_line, y_line, "-", color=FIT_COLOR, linewidth=1.5, zorder=2,
        label=f"fit: ε={details['epsilon']:.4f}, slope={details['slope']:.4f}",
    )
    ax.axhline(
        details["epsilon"], color=FIT_COLOR, linewidth=0.75, linestyle=":",
        zorder=1, alpha=0.6,
    )

    ax.set_xlim(left=0)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{details['eid'][:8]}  (n_units={details['n_total_units']}, "
        f"p={details['p_value']:.3f})",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", frameon=False)
    if label_axes:
        ax.set_xlabel("1 / N (neurons)")
        ax.set_ylabel("balanced_test_mse")


def plot_grid(all_details, output_path, ncols=4):
    n = len(all_details)
    if n == 0:
        raise ValueError("no session details found -- nothing to plot")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False
    )
    for idx, details in enumerate(all_details):
        plot_one_session(axes[idx // ncols][idx % ncols], details)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.supxlabel("1 / N (neurons)", fontsize=10)
    fig.supylabel("balanced_test_mse", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--details-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--eid", default=None,
        help="plot only this one session, as a single large figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    details_dir = (
        Path(args.details_dir).resolve()
        if args.details_dir
        else REPO_ROOT / "results" / "subsampling_epsilon" / args.target_prefix / "details"
    )
    if not details_dir.is_dir():
        raise FileNotFoundError(f"details dir not found: {details_dir}")

    all_details = load_details(details_dir)

    if args.eid is not None:
        all_details = [d for d in all_details if d["eid"] == args.eid]
        if not all_details:
            raise ValueError(f"eid={args.eid} not found under {details_dir}")
        output_path = (
            Path(args.output) if args.output
            else details_dir.parent / f"{args.target_prefix}_{args.eid}_subsampling_curve.png"
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_one_session(ax, all_details[0], label_axes=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        output_path = (
            Path(args.output) if args.output
            else details_dir.parent / f"{args.target_prefix}_subsampling_curves_grid.png"
        )
        plot_grid(all_details, output_path)

    n_negative_slope = sum(1 for d in all_details if d["slope"] < 0)
    print(f"n_sessions_plotted={len(all_details)}")
    print(f"n_sessions_with_negative_slope={n_negative_slope}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()