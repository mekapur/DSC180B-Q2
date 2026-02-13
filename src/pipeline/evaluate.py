from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.compare import (
    evaluate_all,
    results_to_dataframe,
    detailed_results_to_dataframe,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic vs real query results")
    parser.add_argument("--real-dir", type=Path, default=Path("data/results/real"),
                        help="Directory with ground truth CSVs")
    parser.add_argument("--synth-dir", type=Path, required=True,
                        help="Directory with synthetic result CSVs")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV for summary results")
    parser.add_argument("--detailed-output", type=Path, default=None,
                        help="Output CSV for per-metric results")
    parser.add_argument("--rel-tol", type=float, default=0.25,
                        help="Relative error tolerance (default: 0.25)")
    parser.add_argument("--tv-tol", type=float, default=0.15,
                        help="Total variation tolerance (default: 0.15)")
    parser.add_argument("--rho-tol", type=float, default=0.5,
                        help="Spearman rho tolerance (default: 0.5)")
    args = parser.parse_args()

    print(f"Evaluating: {args.real_dir} vs {args.synth_dir}")
    results = evaluate_all(
        args.real_dir, args.synth_dir,
        rel_tol=args.rel_tol, tv_tol=args.tv_tol, rho_tol=args.rho_tol,
    )

    summary = results_to_dataframe(results)
    evaluated = summary[summary["error"] == ""]

    print("\nResults:")
    for _, row in evaluated.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  [{status}] {row['query']}: {row['n_passed']}/{row['n_metrics']} "
              f"metrics passed (score={row['score']:.2f})")

    n_evaluated = len(evaluated)
    n_passed = evaluated["passed"].sum()
    print(f"\nOverall: {n_passed}/{n_evaluated} queries passed")

    skipped = summary[summary["error"] != ""]
    if len(skipped) > 0:
        print(f"Skipped {len(skipped)} queries (missing CSVs)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"\nSummary saved to {args.output}")

    if args.detailed_output:
        detail = detailed_results_to_dataframe(results)
        args.detailed_output.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.detailed_output, index=False)
        print(f"Detailed metrics saved to {args.detailed_output}")


if __name__ == "__main__":
    main()
