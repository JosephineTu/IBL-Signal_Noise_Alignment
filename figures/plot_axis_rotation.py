import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import argparse
import pickle 
from pathlib import Path

def normalize_axis(vector, eps=1e-12):
    vector = np.asarray(vector, float)
    return vector / (np.linalg.norm(vector) + eps)

def projective_axis_distance_matrix(vectors):
    """
    Sign-invariant distance between one-dimensional axes:

        distance(i, j) = sqrt(1 - cos²(i, j))
    """
    vectors = np.asarray(
        [normalize_axis(vector) for vector in vectors],
        dtype=float,
    )

    cosine = np.clip(
        vectors @ vectors.T,
        -1.0,
        1.0,
    )

    distance_squared = np.clip(
        1.0 - cosine**2,
        0.0,
        None,
    )

    return np.sqrt(distance_squared)


def embed_axis_trajectories(
    u_axes,
    noise_axes,
    random_state=0,
):
    u_axes = [
        normalize_axis(vector)
        for vector in u_axes
    ]

    noise_axes = [
        normalize_axis(vector)
        for vector in noise_axes
    ]

    if len(u_axes) != len(noise_axes):
        raise ValueError(
            "u_axes and noise_axes must have the same "
            "number of time bins"
        )

    all_axes = u_axes + noise_axes

    distance_matrix = (
        projective_axis_distance_matrix(all_axes)
    )

    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=20,
        max_iter=1000,
    )

    coordinates = mds.fit_transform(
        distance_matrix
    )

    n_bins = len(u_axes)

    u_coordinates = coordinates[:n_bins]
    noise_coordinates = coordinates[n_bins:]

    return {
        "u_coordinates": u_coordinates,
        "noise_coordinates": noise_coordinates,
        "distance_matrix": distance_matrix,
        "stress": float(mds.stress_),
    }

def plot_axis_trajectories(
    times,
    u_axes,
    noise_axes,
    eid,
    output_path,
):
    times = np.asarray(times, float)

    embedding = embed_axis_trajectories(
        u_axes=u_axes,
        noise_axes=noise_axes,
    )

    u_xyz = embedding["u_coordinates"]
    a_xyz = embedding["noise_coordinates"]

    fig = plt.figure(figsize=(9, 7))

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Encoding-axis trajectory
    ax.plot(
        u_xyz[:, 0],
        u_xyz[:, 1],
        u_xyz[:, 2],
        color="tab:blue",
        linewidth=2.5,
        label=r"$u_{\mathrm{stim}}(t)$",
    )

    u_scatter = ax.scatter(
        u_xyz[:, 0],
        u_xyz[:, 1],
        u_xyz[:, 2],
        c=times,
        cmap="viridis",
        marker="o",
        s=65,
        edgecolor="black",
        linewidth=0.4,
    )

    # Noise-PC1 trajectory
    ax.plot(
        a_xyz[:, 0],
        a_xyz[:, 1],
        a_xyz[:, 2],
        color="tab:red",
        linewidth=2.5,
        label=r"$a(t)$",
    )

    ax.scatter(
        a_xyz[:, 0],
        a_xyz[:, 1],
        a_xyz[:, 2],
        c=times,
        cmap="viridis",
        marker="^",
        s=75,
        edgecolor="black",
        linewidth=0.4,
    )

    # Connect u(t) and a(t) at the same time point.
    for index in range(len(times)):
        ax.plot(
            [
                u_xyz[index, 0],
                a_xyz[index, 0],
            ],
            [
                u_xyz[index, 1],
                a_xyz[index, 1],
            ],
            [
                u_xyz[index, 2],
                a_xyz[index, 2],
            ],
            color="gray",
            linewidth=0.8,
            alpha=0.35,
        )

    # Mark beginning and end.
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
        u_scatter,
        ax=ax,
        pad=0.1,
        shrink=0.75,
    )

    colorbar.set_label(
        "time from stimOn (s)"
    )

    ax.set_xlabel("axis embedding 1")
    ax.set_ylabel("axis embedding 2")
    ax.set_zlabel("axis embedding 3")

    ax.set_title(
        "Stimulus and noise-axis trajectories\n"
        f"eid: {str(eid)[:8]}"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

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
        default="figures/axis_rotation",
    )

    args = parser.parse_args()

    details_pkl = Path(args.details_pkl)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading details from {details_pkl}")

    with details_pkl.open("rb") as file:
        details = pickle.load(file)

    # The top-level keys are eids.
    eids = list(details.keys())

    print(f"Found {len(eids)} sessions")

    for eid in eids:
        session_details = details[eid]

        if "time_windows" not in session_details:
            print(
                f"Skipping {eid}: missing time_windows"
            )
            continue

        if "full_outputs" not in session_details:
            print(
                f"Skipping {eid}: missing full_outputs"
            )
            continue

        time_windows = np.asarray(
            session_details["time_windows"],
            dtype=float,
        )

        full_outputs = session_details[
            "full_outputs"
        ]

        # full_outputs is normally indexed by integer bin_idx.
        # int(key) also allows string keys such as "0".
        bin_keys = sorted(
            full_outputs.keys(),
            key=lambda key: int(key),
        )

        valid_bin_keys = []

        for bin_key in bin_keys:
            output = full_outputs[bin_key]

            if "u_sig" not in output:
                print(
                    f"Skipping eid={eid}, bin={bin_key}: "
                    "missing u_sig"
                )
                continue

            if "noise_a" not in output:
                print(
                    f"Skipping eid={eid}, bin={bin_key}: "
                    "missing noise_a"
                )
                continue

            bin_idx = int(bin_key)

            if not 0 <= bin_idx < len(time_windows):
                print(
                    f"Skipping eid={eid}, bin={bin_key}: "
                    "bin index is outside time_windows"
                )
                continue

            valid_bin_keys.append(bin_key)

        if len(valid_bin_keys) < 2:
            print(
                f"Skipping {eid}: fewer than two valid bins"
            )
            continue

        u_axes = [
            np.asarray(
                full_outputs[bin_key]["u_sig"],
                dtype=float,
            )
            for bin_key in valid_bin_keys
        ]

        noise_axes = [
            np.asarray(
                full_outputs[bin_key]["noise_a"],
                dtype=float,
            )
            for bin_key in valid_bin_keys
        ]

        times = np.asarray(
            [
                np.mean(
                    time_windows[int(bin_key)]
                )
                for bin_key in valid_bin_keys
            ],
            dtype=float,
        )

        # Ensure every time bin uses the same neuronal coordinates.
        u_dimensions = {
            vector.shape
            for vector in u_axes
        }

        noise_dimensions = {
            vector.shape
            for vector in noise_axes
        }

        if len(u_dimensions) != 1:
            print(
                f"Skipping {eid}: u_sig dimensions "
                f"change across bins: {u_dimensions}"
            )
            continue

        if len(noise_dimensions) != 1:
            print(
                f"Skipping {eid}: noise_a dimensions "
                f"change across bins: {noise_dimensions}"
            )
            continue

        if u_dimensions != noise_dimensions:
            print(
                f"Skipping {eid}: u_sig and noise_a "
                "have incompatible dimensions"
            )
            continue

        output_path = (
            output_dir
            / f"{str(eid)[:8]}_axis_rotation.png"
        )

        print(
            f"Plotting {str(eid)[:8]}: "
            f"{len(valid_bin_keys)} bins, "
            f"{u_axes[0].size} units"
        )

        plot_axis_trajectories(
            times=times,
            u_axes=u_axes,
            noise_axes=noise_axes,
            eid=eid,
            output_path=output_path,
        )

    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    main()



