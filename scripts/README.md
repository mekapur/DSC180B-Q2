# scripts/

Standalone utilities that complement the notebooks with non-interactive entry points.

`evaluate_pe.py` evaluates PE synthetic data from a pre-saved wide table or checkpoint. It skips the expensive nearest-neighbor histogram computation (already done in notebook 06) and runs the remaining steps: decompose into reporting tables, execute 21 benchmark queries, and score against ground truth. It reads from `data/reporting/pe_wide_table.parquet` or falls back to `data/pe_checkpoints/selected_iter0.parquet`, and produces evaluation CSVs in `data/results/`.

`generate_report_data.py` reads all `evaluation_*.csv` files and prints the numbers needed for the report's TikZ figures and LaTeX longtable to stdout. It does not produce any files.
