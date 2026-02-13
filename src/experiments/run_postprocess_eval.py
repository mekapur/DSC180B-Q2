from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.postprocess_reporting import (
    PostprocessConfig,
    postprocess_reporting_dir,
)
from src.pipeline.run_benchmark import run_all
from src.eval.compare import evaluate_all, results_to_dataframe, detailed_results_to_dataframe


def main():
    parser = argparse.ArgumentParser(description="Postprocess synthetic reporting tables and evaluate.")
    parser.add_argument("--real-reporting-dir", type=Path, default=Path("data/reporting"))
    parser.add_argument("--synth-reporting-dir", type=Path, required=True)
    parser.add_argument("--queries-dir", type=Path, default=Path("docs/queries"))
    parser.add_argument("--out-root", type=Path, required=True)

    parser.add_argument("--disable-reference-categories", action="store_true")
    parser.add_argument("--fuzzy-cutoff", type=float, default=0.80)

    args = parser.parse_args()

    post_dir = args.out_root / "reporting"
    query_dir = args.out_root / "query_results"
    summary_csv = args.out_root / "evaluation.csv"
    detail_csv = args.out_root / "evaluation_detail.csv"

    cfg = PostprocessConfig(
        use_reference_categories=not args.disable_reference_categories,
        fuzzy_cutoff=args.fuzzy_cutoff,
    )

    print("Postprocessing synthetic reporting tables...")
    written = postprocess_reporting_dir(
        real_reporting_dir=args.real_reporting_dir,
        synth_reporting_dir=args.synth_reporting_dir,
        output_dir=post_dir,
        cfg=cfg,
    )
    print(f"  wrote {len(written)} postprocessed tables to {post_dir}")

    print("Running benchmark queries...")
    run_all(
        queries_dir=args.queries_dir,
        reporting_dir=post_dir,
        output_dir=query_dir,
        skip_infeasible=True,
        verbose=True,
    )

    print("Evaluating query outputs...")
    results = evaluate_all(
        real_dir=Path("data/results/real"),
        synth_dir=query_dir,
    )
    summary = results_to_dataframe(results)
    detail = detailed_results_to_dataframe(results)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    detail.to_csv(detail_csv, index=False)

    eval_rows = summary[summary["n_metrics"] > 0]
    passed = int(eval_rows["passed"].fillna(False).sum())
    total = len(eval_rows)
    avg_score = float(eval_rows["score"].mean()) if total else 0.0

    print(f"Evaluation complete: {passed}/{total} passed, avg_score={avg_score:.3f}")
    print(f"Summary: {summary_csv}")
    print(f"Detail:  {detail_csv}")


if __name__ == "__main__":
    main()

