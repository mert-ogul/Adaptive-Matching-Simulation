"""Command-line interface for Asymmetric(a, δ) simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from .config import Algorithm
from .experiments import (
    DEFAULT_DELTAS,
    DEFAULT_LAMS,
    format_console_table,
    run_grid_experiment,
    run_rep_experiment,
    run_single_experiment,
    write_grid_results,
)


def _add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lam", type=float, required=True, help="Arrival rate λ")
    parser.add_argument("--a", type=float, required=True, help="Asymmetry parameter a")
    parser.add_argument("--delta", type=float, required=True, help="Asymmetry parameter δ")
    parser.add_argument("--horizon", type=float, required=True, help="Simulation horizon")
    parser.add_argument("--warmup", type=float, required=True, help="Warm-up time")


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["random", "backpressure"],
        default="random",
        help="Dispatch policy to use.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Backpressure epsilon; required when --algorithm backpressure.",
    )


def _validate_policy_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.algorithm == "backpressure" and args.eps is None:
        parser.error("--eps is required when --algorithm backpressure")


def _add_adaptive_rep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adaptive-reps",
        action="store_true",
        help="Adaptively increase replications until CI precision target is reached.",
    )
    parser.add_argument(
        "--min-reps",
        type=int,
        default=30,
        help="Minimum reps per cell when --adaptive-reps is enabled.",
    )
    parser.add_argument(
        "--max-reps",
        type=int,
        default=300,
        help="Maximum reps per cell when --adaptive-reps is enabled.",
    )
    parser.add_argument(
        "--rep-batch-size",
        type=int,
        default=10,
        help="Extra reps per adaptive iteration after min-reps is reached.",
    )
    parser.add_argument(
        "--min-significant-digits",
        type=int,
        default=2,
        help="Target significant digits for mean_sojourn_time in each cell.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulation for the Asymmetric(a, δ) expert system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single run
    single = subparsers.add_parser("single", help="Run a single simulation.")
    _add_common_sim_args(single)
    _add_policy_args(single)
    single.add_argument("--seed", type=int, required=True, help="Random seed")

    # Replications
    reps = subparsers.add_parser("reps", help="Run multiple replications.")
    _add_common_sim_args(reps)
    _add_policy_args(reps)
    reps.add_argument("--n-reps", type=int, required=True, help="Number of replications")
    reps.add_argument("--base-seed", type=int, required=True, help="Base random seed")

    # Grid
    grid = subparsers.add_parser("grid", help="Run a full (λ, δ) experiment grid.")
    grid.add_argument("--a", type=float, required=True, help="Asymmetry parameter a")
    grid.add_argument("--horizon", type=float, required=True, help="Simulation horizon")
    grid.add_argument("--warmup", type=float, required=True, help="Warm-up time")
    _add_policy_args(grid)
    _add_adaptive_rep_args(grid)
    grid.add_argument("--n-reps", type=int, required=True, help="Number of replications")
    grid.add_argument("--base-seed", type=int, required=True, help="Base random seed")
    grid.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for CSV files.",
    )
    grid.add_argument(
        "--lams",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMS),
        help="Comma-separated list of λ values (default: 0.75,0.77,...,0.85).",
    )
    grid.add_argument(
        "--deltas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DELTAS),
        help="Comma-separated list of δ values (default: 0.1,0.2,...,0.5).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "single":
        _validate_policy_args(args, parser)
        algorithm = cast(Algorithm, args.algorithm)
        metrics = run_single_experiment(
            lam=args.lam,
            a=args.a,
            delta=args.delta,
            horizon=args.horizon,
            warmup=args.warmup,
            seed=args.seed,
            algorithm=algorithm,
            eps=args.eps,
        )
        print("Single run metrics:")
        for key, value in metrics.items():
            print(f"- {key}: {value}")

    elif args.command == "reps":
        _validate_policy_args(args, parser)
        algorithm = cast(Algorithm, args.algorithm)
        result = run_rep_experiment(
            lam=args.lam,
            a=args.a,
            delta=args.delta,
            horizon=args.horizon,
            warmup=args.warmup,
            n_reps=args.n_reps,
            base_seed=args.base_seed,
            algorithm=algorithm,
            eps=args.eps,
        )
        print("Replication results (mean sojourn time):")
        ci = result["mean_sojourn_time"]
        print(
            f"mean={ci['mean']:.4f}, std={ci['std']:.4f}, "
            f"95% CI=[{ci['low']:.4f}, {ci['high']:.4f}]",
        )

    elif args.command == "grid":
        _validate_policy_args(args, parser)
        algorithm = cast(Algorithm, args.algorithm)
        lam_values = [float(x) for x in args.lams.split(",") if x]
        delta_values = [float(x) for x in args.deltas.split(",") if x]

        summary_df, numeric_df = run_grid_experiment(
            a=args.a,
            lam_values=lam_values,
            delta_values=delta_values,
            horizon=args.horizon,
            warmup=args.warmup,
            n_reps=args.n_reps,
            base_seed=args.base_seed,
            algorithm=algorithm,
            eps=args.eps,
            adaptive_reps=args.adaptive_reps,
            min_reps=args.min_reps,
            max_reps=args.max_reps,
            rep_batch_size=args.rep_batch_size,
            min_significant_digits=args.min_significant_digits,
        )

        print("Mean sojourn time (95% CI half-width) over (λ, δ):")
        print(format_console_table(summary_df))

        write_grid_results(summary_df, numeric_df, args.outdir)

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
