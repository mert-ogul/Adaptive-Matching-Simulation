"""
Generate publication-ready figures from grid experiment results (Random vs Backpressure).

Prerequisite: Run `python scripts/run_experiments.py` to generate both result sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Figure size for two-column article: single-column ~3.5--4", full width ~7"
COLUMN_WIDTH_INCH = 3.5
FULL_WIDTH_INCH = 7.0


def load_results(
    random_path: Path,
    backpressure_path: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load numeric CSVs. Return (random_df, backpressure_df); either may be None if missing."""
    random_df = None
    backpressure_df = None
    if random_path.exists():
        random_df = pd.read_csv(random_path)
    if backpressure_path.exists():
        backpressure_df = pd.read_csv(backpressure_path)
    return random_df, backpressure_df


def plot_sojourn_vs_lambda(
    combined: pd.DataFrame,
    outpath: Path,
    width_inch: float = FULL_WIDTH_INCH,
) -> None:
    """Line plot: mean sojourn time vs λ for each δ, Random vs Backpressure with 95% CI bands."""
    deltas = sorted(combined["delta"].unique())
    n_delta = len(deltas)
    fig, axes = plt.subplots(1, n_delta, figsize=(width_inch, 2.2), sharey=True)
    if n_delta == 1:
        axes = [axes]

    for ax, delta in zip(axes, deltas, strict=True):
        sub = combined[combined["delta"] == delta]
        for algo, label, color, marker in [
            ("random", "Random", "C0", "o"),
            ("backpressure", "Backpressure($\\varepsilon$)", "C1", "s"),
        ]:
            alg_sub = sub[sub["algorithm"] == algo].sort_values("lam")
            if alg_sub.empty:
                continue
            lams = alg_sub["lam"].values
            mean = alg_sub["mean_sojourn_time"].values
            ci_low = alg_sub["ci_low"].values
            ci_high = alg_sub["ci_high"].values
            ax.plot(
                lams,
                mean,
                color=color,
                marker=marker,
                label=label,
                linewidth=1.5,
                markersize=4,
            )
            ax.fill_between(lams, ci_low, ci_high, color=color, alpha=0.2)
        ax.set_xlabel("$\\lambda$")
        ax.set_title(f"$\\delta = {delta}$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Mean sojourn time" if ax == axes[0] else "")

    fig.suptitle("Mean sojourn time vs arrival rate $\\lambda$ (95% CI)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(
    random_df: pd.DataFrame | None,
    backpressure_df: pd.DataFrame | None,
    outpath_combined: Path,
    width_inch: float = FULL_WIDTH_INCH,
) -> None:
    """Two-panel heatmap: mean sojourn (lam × delta) for Random and Backpressure, shared scale."""
    dfs = [(random_df, "Random"), (backpressure_df, "Backpressure($\\varepsilon$)")]
    dfs = [(df, title) for df, title in dfs if df is not None]
    if not dfs:
        return

    # Global min/max for shared color scale
    all_means: list[float] = []
    for df, _ in dfs:
        all_means.extend(df["mean_sojourn_time"].tolist())
    vmin = min(all_means)
    vmax = max(all_means)

    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=(width_inch, 2.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (df, title) in zip(axes, dfs, strict=True):
        pivot = df.pivot(index="lam", columns="delta", values="mean_sojourn_time")
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            origin="lower",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("$\\delta$")
        ax.set_ylabel("$\\lambda$")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, label="Mean sojourn time", shrink=0.6)
    fig.suptitle("Mean sojourn time over $(\\lambda, \\delta)$ grid", fontsize=10, y=1.02)
    fig.set_layout_engine("constrained")
    fig.savefig(outpath_combined, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(
    combined: pd.DataFrame,
    outpath: Path,
    width_inch: float = COLUMN_WIDTH_INCH,
    pairs: list[tuple[float, float]] | None = None,
) -> None:
    """Grouped bar chart at selected (λ, δ) pairs: Random vs Backpressure with error bars."""
    if pairs is None:
        pairs = [(0.79, 0.2), (0.83, 0.3), (0.85, 0.4)]
    labels = [f"($\\lambda={lam}$, $\\delta={delta}$)" for lam, delta in pairs]
    x = range(len(pairs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(width_inch, 2.8))
    for i, (lam, delta) in enumerate(pairs):
        sub = combined[(combined["lam"] == lam) & (combined["delta"] == delta)]
        algos = [("random", "Random"), ("backpressure", "Backpressure")]
        for j, (algo, algo_label) in enumerate(algos):
            row = sub[sub["algorithm"] == algo]
            if row.empty:
                continue
            row = row.iloc[0]
            offset = (j - 0.5) * width
            ax.bar(
                i + offset,
                row["mean_sojourn_time"],
                width,
                yerr=row["ci_half_width"],
                capsize=2,
                label=algo_label if i == 0 else None,
                color="C0" if algo == "random" else "C1",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean sojourn time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Random vs Backpressure at selected $(\\lambda, \\delta)$", fontsize=9, y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot grid experiment results (Random vs Backpressure) for the report."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=repo_root / "figures",
        help="Output directory for PDF figures (default: figures)",
    )
    parser.add_argument(
        "--random",
        type=Path,
        default=repo_root / "results/random/numeric.csv",
        help="Path to Random algorithm numeric.csv",
    )
    parser.add_argument(
        "--backpressure",
        type=Path,
        default=repo_root / "results/backpressure_eps_1_40/numeric.csv",
        help="Path to Backpressure algorithm numeric.csv",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also save PNG versions for quick preview",
    )
    args = parser.parse_args()

    random_df, backpressure_df = load_results(args.random, args.backpressure)
    if random_df is None and backpressure_df is None:
        raise SystemExit(
            "No result files found. Run: python scripts/run_experiments.py\n"
            f"Expected: {args.random} and/or {args.backpressure}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    dfs_to_concat = [df for df in (random_df, backpressure_df) if df is not None]
    combined = pd.concat(dfs_to_concat, ignore_index=True)

    # 1. Sojourn vs λ (only if both algorithms present for comparison)
    sojourn_path = args.outdir / "sojourn_vs_lambda.pdf"
    plot_sojourn_vs_lambda(combined, sojourn_path)
    if args.png:
        plot_sojourn_vs_lambda(combined, args.outdir / "sojourn_vs_lambda.png")

    # 2. Heatmaps (one or both)
    heatmap_path = args.outdir / "heatmaps.pdf"
    plot_heatmaps(random_df, backpressure_df, heatmap_path)
    if args.png:
        plot_heatmaps(random_df, backpressure_df, args.outdir / "heatmaps.png")

    # 3. Bar comparison (only if both present)
    if random_df is not None and backpressure_df is not None:
        bar_path = args.outdir / "bar_comparison.pdf"
        plot_bar_comparison(combined, bar_path)
        if args.png:
            plot_bar_comparison(combined, args.outdir / "bar_comparison.png")

    print(f"Figures written to {args.outdir.absolute()}")


if __name__ == "__main__":
    main()
