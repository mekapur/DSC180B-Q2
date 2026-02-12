"""
Post-process PE batch generation results into evaluated synthetic data.

Usage:
    uv run python -m src.pipeline.pe_postprocess \
        --chunks-dir data/batch_jobs \
        --wide-table data/reporting/wide_training_table.parquet \
        --reporting-dir data/reporting \
        --queries-dir docs/queries \
        --output-dir data/results/synth_pe \
        --n-synth 50000 \
        --epsilon 4.0 \
        --delta 1e-5

Concatenates batch_random_chunk*.parquet files from PE generation,
runs the DP nearest-neighbor histogram and rank-based selection,
decomposes the selected candidates into reporting tables, executes
the 21 benchmark queries, and evaluates against ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.benchmark import run_benchmark
from src.eval.compare import QUERY_METADATA, evaluate_all, results_to_dataframe
from src.eval.decompose import decompose_wide_table
from src.pe.distance import WorkloadDistance
from src.pe.histogram import dp_nn_histogram, select_candidates
from src.pe.privacy import calibrate_sigma


def load_chunks(chunks_dir: Path) -> pd.DataFrame:
    parquets = sorted(chunks_dir.glob("batch_random_chunk*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No chunk parquets found in {chunks_dir}")
    dfs = [pd.read_parquet(p) for p in parquets]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(parquets)} chunks: {len(combined)} total records")
    return combined


def run_pe_postprocess(
    chunks_dir: Path,
    wide_table_path: Path,
    reporting_dir: Path,
    queries_dir: Path,
    output_dir: Path,
    n_synth: int = 50_000,
    epsilon: float = 4.0,
    delta: float = 1e-5,
    T: int = 1,
) -> None:
    population = load_chunks(chunks_dir)

    real_wide = pd.read_parquet(wide_table_path)
    print(f"Real wide table: {len(real_wide)} rows, {real_wide.shape[1]} columns")

    sigma = calibrate_sigma(epsilon, delta, T)
    print(f"Calibrated sigma={sigma:.4f} for (eps={epsilon}, delta={delta}, T={T})")

    dist = WorkloadDistance(real_wide)
    print(f"Distance function: {len(dist.cat_cols)} cat cols, {len(dist.num_cols)} num cols")

    print(f"Computing DP NN histogram ({len(real_wide)} real x {len(population)} synth)...")
    histogram = dp_nn_histogram(real_wide, population, dist, sigma)
    print(f"Histogram: max={histogram.max():.0f}, nonzero={np.count_nonzero(histogram)}")

    selected = select_candidates(population, histogram, n_synth, method="rank")
    print(f"Selected {len(selected)} candidates")

    synth_reporting_dir = output_dir / "reporting"
    counts = decompose_wide_table(selected, synth_reporting_dir)
    print(f"Decomposed into {len(counts)} reporting tables:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    query_names = list(QUERY_METADATA.keys())
    print(f"\nRunning {len(query_names)} benchmark queries on synthetic data...")
    results_dir = output_dir / "results"
    run_benchmark(query_names, queries_dir, synth_reporting_dir, results_dir)

    print("\nEvaluating against ground truth...")
    real_results_dir = reporting_dir.parent / "results" / "real"
    if not real_results_dir.exists():
        real_results_dir = Path("data/results/real")

    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    eval_path = output_dir / "evaluation_pe.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\nEvaluation saved to {eval_path}")

    passed = eval_df[eval_df["score"] >= 0.5]
    print(f"\nResults: {len(passed)}/{len(eval_df)} queries passed (score >= 0.5)")
    print(f"Average score: {eval_df['score'].mean():.3f}")
    print(f"Median score: {eval_df['score'].median():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Post-process PE batch results")
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/batch_jobs"))
    parser.add_argument("--wide-table", type=Path, default=Path("data/reporting/wide_training_table.parquet"))
    parser.add_argument("--reporting-dir", type=Path, default=Path("data/reporting"))
    parser.add_argument("--queries-dir", type=Path, default=Path("docs/queries"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/pe"))
    parser.add_argument("--n-synth", type=int, default=50_000)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()

    run_pe_postprocess(
        chunks_dir=args.chunks_dir,
        wide_table_path=args.wide_table,
        reporting_dir=args.reporting_dir,
        queries_dir=args.queries_dir,
        output_dir=args.output_dir,
        n_synth=args.n_synth,
        epsilon=args.epsilon,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()
