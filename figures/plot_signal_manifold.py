from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def short_eid(eid, n=8):
    return str(eid)[:n]


def paired_contrasts(condition_order, scores):
    condition_order = np.asarray(condition_order, dtype=float)
    scores = np.asarray(scores, dtype=float)

    positive = sorted(
        float(value)
        for value in condition_order
        if value > 0
    )
    negative_magnitudes = {
        float(abs(value))
        for value in condition_order
        if value < 0
    }
    magnitudes = np.asarray(
        [
            value
            for value in positive
            if any(
                np.isclose(value, negative_value)
                for negative_value in negative_magnitudes
            )
        ],
        dtype=float,
    )

    negative_scores = []
    positive_scores = []
    for magnitude in magnitudes:
        neg_idx = np.flatnonzero(
            np.isclose(condition_order, -magnitude)
        )
        pos_idx = np.flatnonzero(
            np.isclose(condition_order, magnitude)
        )
        if len(neg_idx) != 1 or len(pos_idx) != 1:
            raise RuntimeError(
                f"Expected one +/- pair for contrast {magnitude}"
            )
        negative_scores.append(scores[neg_idx[0]])
        positive_scores.append(scores[pos_idx[0]])

    return (
        magnitudes,
        np.asarray(negative_scores, dtype=float),
        np.asarray(positive_scores, dtype=float),
    )


def shared_side_coordinates(
    negative_scores,
    positive_scores,
    required_fraction=0.75,
    n_directions=200000,
    eps=1e-12,
):
    n_core = min(3, negative_scores.shape[1])
    side_vectors = (
        negative_scores[:, :n_core]
        - positive_scores[:, :n_core]
    )
    side_directions = side_vectors / (
        np.linalg.norm(side_vectors, axis=1, keepdims=True) + eps
    )

    if n_core == 2:
        angles = np.linspace(
            0,
            2 * np.pi,
            n_directions,
            endpoint=False,
        )
        candidates = np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
    else:
        index = np.arange(n_directions, dtype=float)
        z = 1.0 - 2.0 * (index + 0.5) / n_directions
        radius = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        azimuth = golden_angle * index
        candidates = np.column_stack(
            [
                radius * np.cos(azimuth),
                radius * np.sin(azimuth),
                z,
            ]
        )

    signed_projections = side_directions @ candidates.T
    n_contrasts = side_directions.shape[0]
    n_required = int(np.ceil(required_fraction * n_contrasts))
    consensus_scores = np.partition(
        signed_projections,
        -n_required,
        axis=0,
    )[-n_required]
    mean_scores = np.mean(signed_projections, axis=0)
    best_index = np.lexsort(
        (mean_scores, consensus_scores)
    )[-1]
    shared_direction = candidates[best_index]
    signed_cosines = signed_projections[:, best_index]
    consensus_cosine = float(consensus_scores[best_index])
    n_aligned = int(np.sum(signed_cosines > 0))

    residuals = side_directions - (
        signed_cosines[:, None]
        * shared_direction[None, :]
    )
    _, residual_values, residual_right_vectors = np.linalg.svd(
        residuals,
        full_matrices=False,
    )

    if len(residual_values) == 0 or residual_values[0] <= eps:
        residual_direction = np.zeros_like(shared_direction)
        residual_direction[np.argmin(np.abs(shared_direction))] = 1.0
        residual_direction -= (
            residual_direction @ shared_direction
        ) * shared_direction
        residual_direction /= (
            np.linalg.norm(residual_direction) + eps
        )
    else:
        residual_direction = residual_right_vectors[0]

    basis = np.column_stack(
        [shared_direction, residual_direction]
    )
    direction_coordinates = side_directions @ basis

    positive_coordinates = -0.5 * direction_coordinates
    negative_coordinates = 0.5 * direction_coordinates

    return (
        negative_coordinates,
        positive_coordinates,
        signed_cosines,
        consensus_cosine,
        n_aligned,
    )


def plot_one_session(eid, signal_manifold, out_dir, target_prefix):
    condition_order = signal_manifold["condition_order"]
    scores = np.asarray(
        signal_manifold["condition_mean_scores"],
        dtype=float,
    )
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise RuntimeError(
            f"Need at least two condition-mean PCs for eid={eid}"
        )

    magnitudes, negative_scores, positive_scores = paired_contrasts(
        condition_order,
        scores,
    )
    if len(magnitudes) < 2:
        raise RuntimeError(
            f"Need at least two matched contrast pairs for eid={eid}"
        )

    (
        negative_coordinates,
        positive_coordinates,
        signed_cosines,
        consensus_cosine,
        n_aligned,
    ) = shared_side_coordinates(negative_scores, positive_scores)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    color_norm = plt.Normalize(
        vmin=float(np.min(magnitudes)),
        vmax=float(np.max(magnitudes)),
    )
    color_map = plt.get_cmap("viridis")

    for index, magnitude in enumerate(magnitudes):
        chord_color = color_map(color_norm(magnitude))
        negative_point = negative_coordinates[index]
        positive_point = positive_coordinates[index]
        line_style = "-" if signed_cosines[index] > 0 else "--"

        ax.plot(
            [negative_point[0], positive_point[0]],
            [negative_point[1], positive_point[1]],
            color=chord_color,
            linewidth=3.0,
            alpha=0.8,
            linestyle=line_style,
            zorder=1,
        )
        ax.scatter(
            negative_point[0],
            negative_point[1],
            color="tab:blue",
            edgecolor="white",
            marker="o",
            s=78,
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            positive_point[0],
            positive_point[1],
            color="tab:red",
            edgecolor="white",
            marker="o",
            s=78,
            linewidth=1.0,
            zorder=3,
        )

    side_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="tab:red",
            markeredgecolor="white",
            markersize=8,
            label="positive",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="tab:blue",
            markeredgecolor="white",
            markersize=8,
            label="negative",
        ),
    ]
    side_legend = ax.legend(
        handles=side_handles,
        title="stimulus side",
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(side_legend)

    contrast_handles = [
        Line2D(
            [0],
            [0],
            color=color_map(color_norm(magnitude)),
            linewidth=3,
            label=f"{magnitude:g}",
        )
        for magnitude in magnitudes
    ]
    ax.legend(
        handles=contrast_handles,
        title="|contrast|",
        frameon=False,
        loc="upper right",
    )

    ax.axhline(0, color="0.55", linewidth=1.0, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.9, zorder=0)
    ax.set_xlabel("shared +→− direction")
    ax.set_ylabel("deviation from shared direction")
    ax.set_title(
        "Stimulus-side axes across contrasts\n"
        f"{target_prefix} | eid: {short_eid(eid)}"
    )
    ax.text(
        0.02,
        0.02,
        f"aligned {n_aligned}/{len(magnitudes)}\n"
        f"consensus cos {consensus_cosine:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    ax.set_aspect("equal", adjustable="box")

    all_coordinates = np.vstack(
        [negative_coordinates, positive_coordinates]
    )
    x_limit = max(0.58, np.max(np.abs(all_coordinates[:, 0])) + 0.10)
    y_limit = max(0.18, np.max(np.abs(all_coordinates[:, 1])) + 0.10)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    fig.tight_layout()

    output_path = (
        out_dir
        / f"{short_eid(eid)}_{target_prefix}_signal_manifold.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--details-pkl", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    target_prefix = args.target_prefix.strip()

    if args.details_pkl is None:
        details_pkl = (
            repo_root
            / "results"
            / "condition_geometry"
            / target_prefix
            / "condition_geometry_details.pkl"
        )
    else:
        details_pkl = Path(args.details_pkl)

    if args.out_dir is None:
        out_dir = (
            repo_root
            / "figures"
            / "condition_geometry"
            / target_prefix
            / "signal_manifold"
        )
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(details_pkl, "rb") as file:
        details = pickle.load(file)

    saved = 0
    for eid, session_details in details.items():
        signal_manifold = session_details.get("signal_manifold")
        if signal_manifold is None:
            print(f"Skipping {eid}: missing signal_manifold")
            continue

        try:
            output_path = plot_one_session(
                eid=eid,
                signal_manifold=signal_manifold,
                out_dir=out_dir,
                target_prefix=target_prefix,
            )
        except Exception as exc:
            print(f"Skipping {eid}: {exc}")
            continue

        print(f"Saved {output_path}")
        saved += 1

    print(f"Saved {saved} signal-manifold figures to {out_dir}")


if __name__ == "__main__":
    main()