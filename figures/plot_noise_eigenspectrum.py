#!/usr/bin/env python3
"""Plot one noise eigenspectrum per session from a geometry summary CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "condition_geometry"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "figures" / "eigenspectrum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read condition_geometry_summary.csv and plot the eigenspectrum "
            "for every eid."
        )
    )
    parser.add_argument(
        "target_prefix",
        nargs="?",
        help="Region/folder name under results/condition_geometry, e.g. VISl.",
    )
    parser.add_argument(
        "--target-prefix",
        "--target_prefix",
        dest="target_prefix_option",
        help="Named-argument form of target_prefix.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Condition-geometry results root (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Figure output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output figure format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster output (default: 300).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic eigenvalue axis (non-positive values are omitted).",
    )
    args = parser.parse_args()
    if args.target_prefix and args.target_prefix_option:
        parser.error("pass target_prefix either positionally or by option, not both")
    args.target_prefix = args.target_prefix or args.target_prefix_option
    if not args.target_prefix:
        parser.error("target_prefix is required (for example: VISl)")
    del args.target_prefix_option
    return args


def parse_eigenspectrum(value: object) -> np.ndarray:
    """Parse a 1D ndarray serialized into a CSV cell."""
    if isinstance(value, np.ndarray):
        spectrum = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        spectrum = np.asarray(value, dtype=float).reshape(-1)
    else:
        if pd.isna(value):
            raise ValueError("missing value")

        text = str(value).strip()
        if text.startswith("array(") and text.endswith(")"):
            text = text[6:-1].strip()

        # np.fromstring safely handles whitespace/scientific notation.  Commas,
        # brackets, and newlines are normalized first so no eval is required.
        text = re.sub(r"[\[\]\(\),]", " ", text)
        spectrum = np.fromstring(text, sep=" ", dtype=float)

    if spectrum.size == 0:
        raise ValueError("empty or unparseable array")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("contains NaN or infinite values")
    return spectrum


def safe_filename(value: object) -> str:
    """Keep an eid from accidentally creating nested paths."""
    name = str(value).strip()
    if not name:
        raise ValueError("empty eid")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def plot_eigenspectrum(
    spectrum: np.ndarray,
    eid: object,
    target_prefix: str,
    output_path: Path,
    *,
    dpi: int,
    log_y: bool,
) -> None:
    ranks = np.arange(1, spectrum.size + 1)
    values = spectrum

    if log_y:
        keep = values > 0
        if not np.any(keep):
            raise ValueError("has no positive eigenvalues for --log-y")
        ranks = ranks[keep]
        values = values[keep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, values, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Noise eigenspectrum | {target_prefix} | {eid}")
    ax.grid(alpha=0.25)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = (
        args.results_root
        / args.target_prefix
        / "condition_geometry_summary.csv"
    )
    output_dir = args.output_root / args.target_prefix

    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    required_columns = {"eid", "eigenspectrum"}
    missing = required_columns.difference(summary.columns)
    if missing:
        raise KeyError(
            f"Missing required column(s) {sorted(missing)} in {summary_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    written = 0

    for row_number, row in summary.iterrows():
        eid = row["eid"]
        try:
            spectrum = parse_eigenspectrum(row["eigenspectrum"])
            filename = f"{safe_filename(eid)}.{args.format}"
            plot_eigenspectrum(
                spectrum,
                eid,
                args.target_prefix,
                output_dir / filename,
                dpi=args.dpi,
                log_y=args.log_y,
            )
            written += 1
        except (TypeError, ValueError) as exc:
            failures.append(f"row {row_number + 2}, eid={eid!r}: {exc}")

    print(f"Wrote {written} figure(s) to {output_dir}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(
            f"Failed to plot {len(failures)} row(s):\n{details}"
        )


if __name__ == "__main__":
    main()