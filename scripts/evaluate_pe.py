import shutil
from pathlib import Path

import pandas as pd

from src.eval.benchmark import run_benchmark
from src.eval.compare import QUERY_METADATA, evaluate_all, results_to_dataframe, detailed_results_to_dataframe
from src.eval.decompose import decompose_wide_table

PE_WIDE = Path("data/reporting/pe_wide_table.parquet")
PE_CHECKPOINT = Path("data/pe_checkpoints/selected_iter0.parquet")
PE_REPORTING = Path("data/reporting/pe")
QUERIES_DIR = Path("docs/queries")
REAL_RESULTS = Path("data/results/real")
SYNTH_PE_RESULTS = Path("data/results/synth_pe")
EVAL_DIR = Path("data/results")


def main():
    if PE_WIDE.exists():
        print(f"Loading PE wide table from {PE_WIDE}")
        synth_wide = pd.read_parquet(PE_WIDE)
    elif PE_CHECKPOINT.exists():
        print(f"Loading from checkpoint: {PE_CHECKPOINT}")
        synth_wide = pd.read_parquet(PE_CHECKPOINT)
        if "guid" not in synth_wide.columns:
            synth_wide.insert(0, "guid", [f"pe_{i:07d}" for i in range(len(synth_wide))])
    else:
        raise FileNotFoundError(
            f"No PE wide table found at {PE_WIDE} or {PE_CHECKPOINT}. "
            "Run notebook 06 first."
        )
    print(f"PE synthetic data: {len(synth_wide):,} rows x {synth_wide.shape[1]} columns")

    print(f"\nDecomposing into reporting tables...")
    counts = decompose_wide_table(synth_wide, PE_REPORTING)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count:,} rows")

    query_names = list(QUERY_METADATA.keys())
    print(f"\nRunning {len(query_names)} benchmark queries...")
    SYNTH_PE_RESULTS.mkdir(parents=True, exist_ok=True)
    run_benchmark(query_names, QUERIES_DIR, PE_REPORTING, SYNTH_PE_RESULTS)

    n_produced = len(list(SYNTH_PE_RESULTS.glob("*.csv")))
    print(f"Produced {n_produced} query result CSVs")

    print(f"\nEvaluating against ground truth...")
    eval_results = evaluate_all(REAL_RESULTS, SYNTH_PE_RESULTS)
    eval_df = results_to_dataframe(eval_results)
    detail_df = detailed_results_to_dataframe(eval_results)

    eval_path = EVAL_DIR / "evaluation_pe.csv"
    detail_path = EVAL_DIR / "evaluation_pe_detail.csv"
    eval_df.to_csv(eval_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved: {eval_path}")
    print(f"Saved: {detail_path}")

    valid = eval_df[eval_df["error"].isna() | (eval_df["error"] == "")]
    n_passed = int(valid["passed"].sum())
    n_eval = len(valid)
    avg_score = valid["score"].mean()
    med_score = valid["score"].median()
    print(f"\nResults: {n_passed}/{n_eval} queries passed (score >= 0.5)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Median score: {med_score:.3f}")

    print("\nPer-query breakdown:")
    for _, row in eval_df.iterrows():
        if row["error"]:
            status = f"ERROR: {row['error']}"
        elif row["passed"]:
            status = f"PASS  score={row['score']:.3f}"
        else:
            status = f"FAIL  score={row['score']:.3f}"
        print(f"  {row['query'][:55]:55s} {status}")


if __name__ == "__main__":
    main()
