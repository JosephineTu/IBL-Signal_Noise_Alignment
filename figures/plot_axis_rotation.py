from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


EPS = 1e-12


def normalize_axis(vector, eps=EPS):
    """Normalize one neuron-space direction after validating it."""
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError(
            f"Axis must be one-dimensional; got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Axis contains NaN or infinite values")

    norm = np.linalg.norm(vector)
    if norm <= eps:
        raise ValueError("Axis has zero or near-zero norm")

    return vector / norm


def projective_axis_distance_matrix(vectors):
    """Compute sign-invariant distances sqrt(1 - cos^2) between axes."""
    vectors = np.asarray(
        [normalize_axis(vector) for vector in vectors],
        dtype=float,
    )

    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    distance_squared = np.clip(1.0 - cosine**2, 0.0, None)
    return np.sqrt(distance_squared)


def embed_axis_trajectories(
    u_axes,
    noise_axes,
    w_2d_axes,
    w_full_axes,
    random_state=0,
):
    """Jointly embed all four trajectories in one common MDS space."""
    groups = {
        "u_coordinates": [normalize_axis(v) for v in u_axes],
        "noise_coordinates": [normalize_axis(v) for v in noise_axes],
        "w_2d_coordinates": [normalize_axis(v) for v in w_2d_axes],
        "w_full_coordinates": [normalize_axis(v) for v in w_full_axes],
    }

    lengths = {name: len(vectors) for name, vectors in groups.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "All axis groups must have the same number of time bins; "
            f"got {lengths}"
        )

    n_bins = len(u_axes)
    if n_bins < 2:
        raise ValueError("At least two valid time bins are required")

    all_axes = []
    for vectors in groups.values():
        all_axes.extend(vectors)

    distance_matrix = projective_axis_distance_matrix(all_axes)

    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=20,
        max_iter=1000,
    )
    coordinates = mds.fit_transform(distance_matrix)

    result = {
        "distance_matrix": distance_matrix,
        "stress": float(mds.stress_),
    }

    start = 0
    for name in groups:
        stop = start + n_bins
        result[name] = coordinates[start:stop]
        start = stop

    return result


def plot_one_trajectory(
    ax,
    xyz,
    times,
    color,
    marker,
    label,
    marker_size,
):
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        color=color,
        linewidth=2.4,
        label=label,
    )

    return ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=times,
        cmap="viridis",
        marker=marker,
        s=marker_size,
        edgecolor="black",
        linewidth=0.4,
    )


def connect_matched_timepoints(ax, first_xyz, second_xyz):
    """Connect two axis families at matching time bins."""
    for index in range(len(first_xyz)):
        ax.plot(
            [first_xyz[index, 0], second_xyz[index, 0]],
            [first_xyz[index, 1], second_xyz[index, 1]],
            [first_xyz[index, 2], second_xyz[index, 2]],
            color="gray",
            linewidth=0.75,
            alpha=0.28,
        )


def plot_axis_trajectories(
    times,
    u_axes,
    noise_axes,
    w_2d_axes,
    w_full_axes,
    eid,
    output_path,
    random_state=0,
):
    times = np.asarray(times, dtype=float)

    embedding = embed_axis_trajectories(
        u_axes=u_axes,
        noise_axes=noise_axes,
        w_2d_axes=w_2d_axes,
        w_full_axes=w_full_axes,
        random_state=random_state,
    )

    u_xyz = embedding["u_coordinates"]
    a_xyz = embedding["noise_coordinates"]
    w_2d_xyz = embedding["w_2d_coordinates"]
    w_full_xyz = embedding["w_full_coordinates"]

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    time_scatter = plot_one_trajectory(
        ax=ax,
        xyz=u_xyz,
        times=times,
        color="tab:blue",
        marker="o",
        label=r"$u_{\mathrm{stim}}(t)$",
        marker_size=65,
    )

    plot_one_trajectory(
        ax=ax,
        xyz=a_xyz,
        times=times,
        color="tab:red",
        marker="^",
        label=r"$a(t)$",
        marker_size=75,
    )

    plot_one_trajectory(
        ax=ax,
        xyz=w_2d_xyz,
        times=times,
        color="tab:green",
        marker="s",
        label=r"$w_{\mathrm{2D}}(t)$",
        marker_size=62,
    )

    plot_one_trajectory(
        ax=ax,
        xyz=w_full_xyz,
        times=times,
        color="tab:purple",
        marker="D",
        label=r"$w_{\mathrm{full}}(t)$",
        marker_size=62,
    )

    # Existing signal-noise comparison and the new decoder comparison.
    connect_matched_timepoints(ax, u_xyz, a_xyz)
    connect_matched_timepoints(ax, w_2d_xyz, w_full_xyz)

    ax.text(
        u_xyz[0, 0],
        u_xyz[0, 1],
        u_xyz[0, 2],
        f"  start ({times[0]:.3f}s)",
    )
    ax.text(
        u_xyz[-1, 0],
        u_xyz[-1, 1],
        u_xyz[-1, 2],
        f"  end ({times[-1]:.3f}s)",
    )

    colorbar = fig.colorbar(
        time_scatter,
        ax=ax,
        pad=0.1,
        shrink=0.75,
    )
    colorbar.set_label("time from stimOn (s)")

    ax.set_xlabel("axis embedding 1")
    ax.set_ylabel("axis embedding 2")
    ax.set_zlabel("axis embedding 3")
    ax.set_title(
        "Stimulus, noise, and decoder-axis trajectories\n"
        f"eid: {str(eid)[:8]} | MDS stress: {embedding['stress']:.3g}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_first_present_key(mapping, candidates):
    """Support both the requested keys and older descriptive key names."""
    for key in candidates:
        if key in mapping:
            return key
    return None


def axes_have_compatible_dimensions(axis_groups):
    shapes = {
        group_name: {vector.shape for vector in vectors}
        for group_name, vectors in axis_groups.items()
    }

    if any(len(group_shapes) != 1 for group_shapes in shapes.values()):
        return False, shapes

    unique_shapes = {
        next(iter(group_shapes))
        for group_shapes in shapes.values()
    }
    return len(unique_shapes) == 1, shapes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--details-pkl",
        default=(
            "results/timebinned_alignment/"
            "t0p0to0p1_bin0p03_step0p01_k3"
            "_timebinned_alignment_details.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="figures/axis_rotation_with_decoders",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    details_pkl = Path(args.details_pkl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading details from {details_pkl}")
    with details_pkl.open("rb") as file:
        details = pickle.load(file)

    eids = list(details.keys())
    print(f"Found {len(eids)} sessions")

    for eid in eids:
        session_details = details[eid]

        if "time_windows" not in session_details:
            print(f"Skipping {eid}: missing time_windows")
            continue
        if "full_outputs" not in session_details:
            print(f"Skipping {eid}: missing full_outputs")
            continue

        time_windows = np.asarray(
            session_details["time_windows"],
            dtype=float,
        )
        full_outputs = session_details["full_outputs"]
        bin_keys = sorted(full_outputs.keys(), key=lambda key: int(key))

        times = []
        u_axes = []
        noise_axes = []
        w_2d_axes = []
        w_full_axes = []

        for bin_key in bin_keys:
            output = full_outputs[bin_key]
            bin_idx = int(bin_key)

            if not 0 <= bin_idx < len(time_windows):
                print(
                    f"Skipping eid={eid}, bin={bin_key}: "
                    "bin index is outside time_windows"
                )
                continue

            w_2d_key = get_first_present_key(
                output,
                ("w_2D", "w_2d", "decoder_w_2d_eff"),
            )
            w_full_key = get_first_present_key(
                output,
                ("w_full", "decoder_w_full"),
            )

            missing = []
            if "u_sig" not in output:
                missing.append("u_sig")
            if "noise_a" not in output:
                missing.append("noise_a")
            if w_2d_key is None:
                missing.append("w_2D")
            if w_full_key is None:
                missing.append("w_full")

            if missing:
                print(
                    f"Skipping eid={eid}, bin={bin_key}: "
                    f"missing {', '.join(missing)}"
                )
                continue

            candidate_axes = {
                "u_sig": np.asarray(output["u_sig"], dtype=float),
                "noise_a": np.asarray(output["noise_a"], dtype=float),
                "w_2D": np.asarray(output[w_2d_key], dtype=float),
                "w_full": np.asarray(output[w_full_key], dtype=float),
            }

            try:
                candidate_axes = {
                    name: normalize_axis(vector)
                    for name, vector in candidate_axes.items()
                }
            except ValueError as error:
                print(
                    f"Skipping eid={eid}, bin={bin_key}: {error}"
                )
                continue

            times.append(float(np.mean(time_windows[bin_idx])))
            u_axes.append(candidate_axes["u_sig"])
            noise_axes.append(candidate_axes["noise_a"])
            w_2d_axes.append(candidate_axes["w_2D"])
            w_full_axes.append(candidate_axes["w_full"])

        if len(times) < 2:
            print(f"Skipping {eid}: fewer than two valid bins")
            continue

        axis_groups = {
            "u_sig": u_axes,
            "noise_a": noise_axes,
            "w_2D": w_2d_axes,
            "w_full": w_full_axes,
        }
        compatible, shapes = axes_have_compatible_dimensions(axis_groups)
        if not compatible:
            print(
                f"Skipping {eid}: incompatible axis dimensions: {shapes}"
            )
            continue

        times = np.asarray(times, dtype=float)
        output_path = (
            output_dir
            / f"{str(eid)[:8]}_axis_rotation_with_decoders.png"
        )

        print(
            f"Plotting {str(eid)[:8]}: {len(times)} bins, "
            f"{u_axes[0].size} units"
        )

        plot_axis_trajectories(
            times=times,
            u_axes=u_axes,
            noise_axes=noise_axes,
            w_2d_axes=w_2d_axes,
            w_full_axes=w_full_axes,
            eid=eid,
            output_path=output_path,
            random_state=args.random_state,
        )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()



