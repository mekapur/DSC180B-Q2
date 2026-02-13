from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.benchmark import run_benchmark
from src.eval.compare import QUERY_METADATA, evaluate_all, results_to_dataframe, detailed_results_to_dataframe
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
    real_subsample: int | None = None,
    from_checkpoint: Path | None = None,
) -> None:
    if from_checkpoint is not None:
        selected_path = from_checkpoint / "selected_iter0.parquet"
        population_path = from_checkpoint / "population_iter0.parquet"

        if selected_path.exists():
            print(f"Loading pre-selected candidates from {selected_path}")
            selected = pd.read_parquet(selected_path)
            if "guid" not in selected.columns:
                selected.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(selected))])
            print(f"Selected candidates: {len(selected):,} rows")
            return _evaluate_selected(
                selected, reporting_dir, queries_dir, output_dir
            )

        if population_path.exists():
            print(f"Loading population from checkpoint: {population_path}")
            population = pd.read_parquet(population_path)
            print(f"Population: {len(population):,} rows")
        else:
            raise FileNotFoundError(
                f"No checkpoint data at {from_checkpoint}. "
                "Run PE generation first."
            )
    else:
        population = load_chunks(chunks_dir)

    real_wide = pd.read_parquet(wide_table_path)
    print(f"Real wide table: {len(real_wide)} rows, {real_wide.shape[1]} columns")

    dist = WorkloadDistance(real_wide)
    print(f"Distance function: {len(dist.cat_cols)} cat cols, {len(dist.num_cols)} num cols")

    if real_subsample is not None and real_subsample < len(real_wide):
        voting_df = real_wide.sample(real_subsample, random_state=42).reset_index(drop=True)
        print(f"Subsampled {real_subsample:,} from {len(real_wide):,} real records for voting")
    else:
        voting_df = real_wide

    sigma = calibrate_sigma(epsilon, delta, T)
    print(f"Calibrated sigma={sigma:.4f} for (eps={epsilon}, delta={delta}, T={T})")

    print(f"Computing DP NN histogram ({len(voting_df)} real x {len(population)} synth)...")
    histogram = dp_nn_histogram(voting_df, population, dist, sigma)
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

    detail_df = detailed_results_to_dataframe(eval_results)
    detail_path = output_dir / "evaluation_pe_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Detailed evaluation saved to {detail_path}")

    std_eval = reporting_dir.parent / "results" / "evaluation_pe.csv"
    std_detail = reporting_dir.parent / "results" / "evaluation_pe_detail.csv"
    eval_df.to_csv(std_eval, index=False)
    detail_df.to_csv(std_detail, index=False)
    print(f"Also saved to {std_eval}")

    import shutil
    std_synth = reporting_dir.parent / "results" / "synth_pe"
    if results_dir != std_synth:
        std_synth.mkdir(parents=True, exist_ok=True)
        for csv_file in results_dir.glob("*.csv"):
            shutil.copy2(csv_file, std_synth / csv_file.name)
        print(f"Query results copied to {std_synth}")

    passed = eval_df[eval_df["score"] >= 0.5]
    print(f"\nResults: {len(passed)}/{len(eval_df)} queries passed (score >= 0.5)")
    print(f"Average score: {eval_df['score'].mean():.3f}")
    print(f"Median score: {eval_df['score'].median():.3f}")


def _evaluate_selected(
    selected: pd.DataFrame,
    reporting_dir: Path,
    queries_dir: Path,
    output_dir: Path,
) -> None:
    synth_reporting_dir = output_dir / "reporting"
    counts = decompose_wide_table(selected, synth_reporting_dir)
    print(f"Decomposed into {len(counts)} reporting tables:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} rows")

    query_names = list(QUERY_METADATA.keys())
    print(f"\nRunning {len(query_names)} benchmark queries...")
    results_dir = output_dir / "results"
    run_benchmark(query_names, queries_dir, synth_reporting_dir, results_dir)

    print("\nEvaluating against ground truth...")
    real_results_dir = reporting_dir.parent / "results" / "real"
    if not real_results_dir.exists():
        real_results_dir = Path("data/results/real")

    eval_results = evaluate_all(real_results_dir, results_dir)
    eval_df = results_to_dataframe(eval_results)
    detail_df = detailed_results_to_dataframe(eval_results)

    std_eval = reporting_dir.parent / "results" / "evaluation_pe.csv"
    std_detail = reporting_dir.parent / "results" / "evaluation_pe_detail.csv"
    eval_df.to_csv(std_eval, index=False)
    detail_df.to_csv(std_detail, index=False)
    print(f"Saved: {std_eval}")
    print(f"Saved: {std_detail}")

    import shutil
    std_synth = reporting_dir.parent / "results" / "synth_pe"
    std_synth.mkdir(parents=True, exist_ok=True)
    for csv_file in results_dir.glob("*.csv"):
        shutil.copy2(csv_file, std_synth / csv_file.name)
    print(f"Query results copied to {std_synth}")

    valid = eval_df[(eval_df["error"].isna()) | (eval_df["error"] == "")]
    passed = valid[valid["passed"] == True]
    print(f"\nResults: {len(passed)}/{len(valid)} queries passed (score >= 0.5)")
    print(f"Average score: {valid['score'].mean():.3f}")
    print(f"Median score: {valid['score'].median():.3f}")


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
    parser.add_argument("--real-subsample", type=int, default=None,
                        help="Subsample N real records for voting (speeds up NN computation)")
    parser.add_argument("--from-checkpoint", type=Path, default=None,
                        help="Resume from checkpoint dir (skips NN if selection done)")
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
        real_subsample=args.real_subsample,
        from_checkpoint=args.from_checkpoint,
    )


if __name__ == "__main__":
    main()
