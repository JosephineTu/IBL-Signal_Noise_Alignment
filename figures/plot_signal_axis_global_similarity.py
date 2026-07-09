# plot_signal_axis_global_similarity.py
from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def short_eid(eid, n=6):
    return str(eid)[:n]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def normalize_loaded_key(x):
    """
    Make keys robust if they were saved as tuples or strings.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def format_num(x):
    x = safe_float(x)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.3g}"


def format_condition(c):
    """
    Format one condition.

    Supports:
      signed contrast: -1.0, 0.25, ...
      contrast pair: (contrastLeft, contrastRight)
    """
    c = normalize_loaded_key(c)

    if isinstance(c, (tuple, list, np.ndarray)):
        vals = list(c)
        if len(vals) == 2:
            return f"({format_num(vals[0])},{format_num(vals[1])})"
        return "(" + ",".join(format_num(v) for v in vals) + ")"

    return format_num(c)


def format_axis_key(axis_key):
    """
    axis_key is usually:
        (cond_a, cond_b)

    where each cond can be signed contrast or (contrastLeft, contrastRight).
    """
    axis_key = normalize_loaded_key(axis_key)

    if isinstance(axis_key, (tuple, list)) and len(axis_key) == 2:
        return f"{format_condition(axis_key[0])} vs {format_condition(axis_key[1])}"

    return str(axis_key)


def condition_signed_value(cond):
    """
    Return signed contrast value for a condition.

    Supports:
      signed contrast key: -1.0, 0.25, ...
      contrast pair key: (contrastLeft, contrastRight)
    """
    cond = normalize_loaded_key(cond)

    if isinstance(cond, (tuple, list, np.ndarray)) and len(cond) == 2:
        cl = safe_float(cond[0])
        cr = safe_float(cond[1])
        if np.isfinite(cl) and np.isfinite(cr):
            return cl - cr

    val = safe_float(cond)
    if np.isfinite(val):
        return val

    return np.nan


def is_high_condition(cond, threshold=0.5):
    signed = condition_signed_value(cond)
    return np.isfinite(signed) and (abs(signed) >= threshold)


def axis_high_category(axis_key, threshold=0.5):
    """
    axis_key = (cond_a, cond_b)

    Returns:
      both_high
      one_high
      neither_high
      unknown
    """
    axis_key = normalize_loaded_key(axis_key)

    if not (isinstance(axis_key, (tuple, list)) and len(axis_key) == 2):
        return "unknown"

    cond_a, cond_b = axis_key

    high_a = is_high_condition(cond_a, threshold=threshold)
    high_b = is_high_condition(cond_b, threshold=threshold)

    if high_a and high_b:
        return "both_high"
    if high_a or high_b:
        return "one_high"
    return "neither_high"


def axis_sort_key(axis_key):
    """
    Sort pair axes by:
      1. category-like signed midpoint
      2. signed span
      3. string fallback
    """
    axis_key = normalize_loaded_key(axis_key)

    if isinstance(axis_key, (tuple, list)) and len(axis_key) == 2:
        a = condition_signed_value(axis_key[0])
        b = condition_signed_value(axis_key[1])

        if np.isfinite(a) and np.isfinite(b):
            midpoint = 0.5 * (a + b)
            span = abs(a - b)
            return (midpoint, span, str(axis_key))

    return (np.inf, np.inf, str(axis_key))


def get_global_cosine(valdict, use_abs=True):
    if not isinstance(valdict, dict):
        return np.nan

    if use_abs:
        if "abs_cosine_global" in valdict:
            return safe_float(valdict["abs_cosine_global"])
        if "cosine_global" in valdict:
            return abs(safe_float(valdict["cosine_global"]))

    if "cosine_global" in valdict:
        return safe_float(valdict["cosine_global"])

    if "abs_cosine_global" in valdict:
        return safe_float(valdict["abs_cosine_global"])

    return np.nan


def load_session_metadata(summary_df, eid):
    if summary_df is None:
        return {}

    hit = summary_df[summary_df["eid"].astype(str) == str(eid)]
    if len(hit) == 0:
        return {}

    row = hit.iloc[0]

    out = {}
    for col in ["n_trials", "n_units", "n_conditions"]:
        if col in row and pd.notna(row[col]):
            out[col] = int(row[col])

    for col in ["stim_pc1_var", "stim_pc2_var", "stim_pc3_var", "stim_pc123_var"]:
        if col in row and pd.notna(row[col]):
            out[col] = float(row[col])

    return out


def global_summary_to_dataframe(detail, high_threshold=0.5, use_abs=True):
    """
    Convert detail["global_signal_axis_summary"] to a clean dataframe.

    Each row is one contrast-pair signal axis compared to the global
    high-contrast signal axis.
    """
    global_results = detail.get("global_signal_axis_summary", {})

    rows = []

    for axis_key, valdict in global_results.items():
        axis_key = normalize_loaded_key(axis_key)

        cosine = get_global_cosine(valdict, use_abs=use_abs)
        if not np.isfinite(cosine):
            continue

        if isinstance(axis_key, (tuple, list)) and len(axis_key) == 2:
            cond_a, cond_b = axis_key
            signed_a = condition_signed_value(cond_a)
            signed_b = condition_signed_value(cond_b)
            span = abs(signed_a - signed_b) if np.isfinite(signed_a) and np.isfinite(signed_b) else np.nan
            midpoint = 0.5 * (signed_a + signed_b) if np.isfinite(signed_a) and np.isfinite(signed_b) else np.nan
        else:
            cond_a, cond_b = None, None
            signed_a, signed_b = np.nan, np.nan
            span, midpoint = np.nan, np.nan

        category = axis_high_category(axis_key, threshold=high_threshold)

        rows.append(
            {
                "axis_key": axis_key,
                "axis_label": format_axis_key(axis_key),
                "cond_a": cond_a,
                "cond_b": cond_b,
                "signed_a": signed_a,
                "signed_b": signed_b,
                "span": span,
                "midpoint": midpoint,
                "category": category,
                "cosine_global": cosine,
                "sort_key": axis_sort_key(axis_key),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values(
        by=["category", "midpoint", "span", "axis_label"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return df


def category_order_name(category):
    order = {
        "both_high": 0,
        "one_high": 1,
        "neither_high": 2,
        "unknown": 3,
    }
    return order.get(category, 99)


def sort_for_plot(df, mode="category"):
    if len(df) == 0:
        return df

    df = df.copy()

    if mode == "value":
        return df.sort_values("cosine_global", ascending=False).reset_index(drop=True)

    df["category_order"] = df["category"].map(category_order_name)
    return df.sort_values(
        by=["category_order", "midpoint", "span", "axis_label"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def category_marker(category):
    if category == "both_high":
        return "o"
    if category == "one_high":
        return "s"
    if category == "neither_high":
        return "^"
    return "x"


def plot_one_eid_global_similarity(
    eid,
    detail,
    out_dir,
    summary_df=None,
    high_threshold=0.5,
    use_abs=True,
    sort_mode="category",
):
    df = global_summary_to_dataframe(
        detail,
        high_threshold=high_threshold,
        use_abs=use_abs,
    )

    if len(df) == 0:
        print(f"skip {eid}: no global_signal_axis_summary")
        return None

    df = sort_for_plot(df, mode=sort_mode)

    meta = load_session_metadata(summary_df, eid)

    n = len(df)
    fig_width = max(9.0, min(22.0, 0.38 * n))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(n)

    categories = ["both_high", "one_high", "neither_high", "unknown"]

    for cat in categories:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue

        idx = sub.index.to_numpy()
        ax.scatter(
            idx,
            sub["cosine_global"].to_numpy(),
            marker=category_marker(cat),
            s=45,
            label=cat,
            alpha=0.9,
        )

    ax.set_ylim(-0.02 if use_abs else -1.02, 1.02)

    if use_abs:
        ax.set_ylabel("|cosine(pairwise signal axis, global high-contrast signal axis)|")
    else:
        ax.set_ylabel("cosine(pairwise signal axis, global high-contrast signal axis)")

    ax.set_xlabel("contrast-pair signal axis")

    ax.set_xticks(x)
    ax.set_xticklabels(df["axis_label"].tolist(), rotation=70, ha="right", fontsize=7)

    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.8, linestyle=":", linewidth=1, alpha=0.7)

    ax.legend(title="axis category", fontsize=8, title_fontsize=8, loc="lower left")

    title_metric = "|cosine|" if use_abs else "cosine"
    ax.set_title(
        f"{short_eid(eid)} pairwise signal axes vs global high-contrast signal axis\n"
        f"metric = {title_metric}; high condition threshold = |signed contrast| >= {high_threshold}"
    )

    # Add compact metadata box
    meta_lines = [f"eid: {short_eid(eid)}"]
    if "n_trials" in meta:
        meta_lines.append(f"trials: {meta['n_trials']}")
    if "n_units" in meta:
        meta_lines.append(f"units: {meta['n_units']}")
    if "n_conditions" in meta:
        meta_lines.append(f"conditions: {meta['n_conditions']}")
    if "stim_pc123_var" in meta:
        meta_lines.append(f"stim PC1-3 var: {meta['stim_pc123_var']:.3f}")

    category_means = df.groupby("category")["cosine_global"].mean()
    for cat in ["both_high", "one_high", "neither_high"]:
        if cat in category_means:
            meta_lines.append(f"{cat} mean: {category_means[cat]:.3f}")

    ax.text(
        1.01,
        0.98,
        "\n".join(meta_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()

    suffix = "abs_cosine" if use_abs else "cosine"
    out_path = out_dir / f"{short_eid(eid)}_signal_axis_vs_global_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_path}")

    # Return per-axis rows for optional group summary
    df_out = df.copy()
    df_out.insert(0, "eid", eid)
    return df_out


def plot_group_category_summary(all_rows, out_dir, use_abs=True):
    if not all_rows:
        return

    df = pd.concat(all_rows, ignore_index=True)
    df = df[df["category"].isin(["both_high", "one_high", "neither_high"])].copy()

    if len(df) == 0:
        return

    categories = ["both_high", "one_high", "neither_high"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    data = [
        df.loc[df["category"] == cat, "cosine_global"].dropna().to_numpy()
        for cat in categories
    ]

    ax.boxplot(data, labels=categories, showfliers=False)

    # jittered points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=12,
            alpha=0.35,
        )

    ax.set_ylim(-0.02 if use_abs else -1.02, 1.02)

    if use_abs:
        ax.set_ylabel("|cosine(pairwise axis, global axis)|")
        suffix = "abs_cosine"
    else:
        ax.set_ylabel("cosine(pairwise axis, global axis)")
        suffix = "cosine"

    ax.set_xlabel("pairwise signal-axis category")
    ax.set_title("Pairwise signal axes vs global high-contrast axis across sessions")

    fig.tight_layout()

    out_path = out_dir / f"group_signal_axis_vs_global_by_category_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / f"group_signal_axis_vs_global_by_category_{suffix}.csv"
    df.to_csv(csv_path, index=False)

    print(f"saved {out_path}")
    print(f"saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--details-pkl",
        default="results/condition_geometry/condition_geometry_details.pkl",
    )

    parser.add_argument(
        "--summary-csv",
        default="results/condition_geometry/condition_geometry_summary.csv",
    )

    parser.add_argument(
        "--out-dir",
        default="figures/figure_1_2/signal_axis_vs_global",
    )

    parser.add_argument(
        "--metric",
        choices=["abs", "signed"],
        default="abs",
        help="Use abs cosine or signed cosine. Default: abs.",
    )

    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.5,
        help="High contrast threshold on |contrastLeft - contrastRight|.",
    )

    parser.add_argument(
        "--sort-mode",
        choices=["category", "value"],
        default="category",
        help="Sort x-axis by category or by descending cosine value.",
    )

    parser.add_argument(
        "--eids",
        nargs="*",
        default=None,
        help="Optional list of eids to plot. Default: all eids.",
    )

    parser.add_argument(
        "--no-group-summary",
        action="store_true",
        help="Do not save across-session category summary plot.",
    )

    args = parser.parse_args()

    details_path = Path(args.details_pkl)
    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(details_path, "rb") as f:
        details = pickle.load(f)

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = None

    if args.eids:
        requested = set(str(e) for e in args.eids)
        items = [
            (eid, detail)
            for eid, detail in details.items()
            if str(eid) in requested
        ]
    else:
        items = list(details.items())

    print(f"Plotting {len(items)} eids from {details_path}")

    use_abs = args.metric == "abs"
    all_rows = []

    for eid, detail in items:
        df_out = plot_one_eid_global_similarity(
            eid=eid,
            detail=detail,
            out_dir=out_dir,
            summary_df=summary_df,
            high_threshold=args.high_threshold,
            use_abs=use_abs,
            sort_mode=args.sort_mode,
        )

        if df_out is not None:
            all_rows.append(df_out)

    if not args.no_group_summary:
        plot_group_category_summary(
            all_rows=all_rows,
            out_dir=out_dir,
            use_abs=use_abs,
        )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()