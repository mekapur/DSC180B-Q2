# src/

Three packages: `src.pe` (Private Evolution), `src.eval` (evaluation), and `src.pipeline` (CLI stages). Notebooks 06-10 import from these packages. Notebooks 01-05 use inline logic and do not depend on `src/`.

## pe/ — Private Evolution

Implements Private Evolution (Lin et al. 2024) adapted for tabular data (Swanberg et al. 2025).

`privacy.py` calibrates the Gaussian noise parameter $\sigma$ for $(\varepsilon, \delta)$-differential privacy using the analytic Gaussian mechanism from Balle and Wang (2018). The function `calibrate_sigma(epsilon, delta, T)` finds the minimum $\sigma$ satisfying the privacy constraint across $T$ iterations, and `compute_epsilon(sigma, delta, T)` computes the achieved $\varepsilon$ for a given $\sigma$.

`distance.py` implements the workload-aware weighted $\ell_1$ distance function. Nine categorical columns are scored by mismatch, weighted by how many benchmark queries reference each attribute. Fifty-nine numeric columns are scored by $\ell_1$ distance on min-max normalized values. The `WorkloadDistance` class fits normalization bounds from the real dataset, and `nearest_neighbors(real_df, synth_df)` returns the index of the closest synthetic record for each real record using chunked `scipy.cdist` with thread parallelism.

`histogram.py` contains the DP nearest-neighbor histogram and PE orchestration logic. The function `dp_nn_histogram()` tallies votes from real records for their nearest synthetic candidates and adds calibrated Gaussian noise $\mathcal{N}(0, \sigma^2)$. The function `select_candidates()` picks the top $N_\text{synth}$ candidates by noisy vote count. The function `private_evolution()` runs the full PE loop: generate $N_\text{synth} \times L$ candidates via the API, compute the histogram, and select the final population. The `PECheckpoint` class provides stage-based resume so that interrupted runs can skip completed work.

`api.py` is the OpenAI interface for generating synthetic telemetry records. It uses Pydantic Structured Outputs for guaranteed schema compliance. Two modes are supported: real-time async mode for smoke tests and small runs, and Batch API mode for full generation runs at 50% reduced cost. In batch mode, requests are split into chunks of 800 with parquet checkpointing so that interrupted runs can resume without re-paying for completed batches.

## eval/ — Evaluation

`benchmark.py` executes SQL queries from `docs/queries/*.json` against reporting parquet tables via DuckDB. The function `adapt_sql()` rewrites `reporting.table_name` references in the SQL to `read_parquet()` calls pointing at the correct parquet files. The function `run_benchmark()` runs a list of queries and saves each result as a CSV.

`compare.py` scores synthetic query results against ground truth using metrics matched to each query type: relative error for aggregate queries, total variation distance for distributions, Spearman's $\rho$ for rankings, and Jaccard similarity for group coverage. A query passes if at least 50% of its per-column metrics meet their respective thresholds. The constant `QUERY_METADATA` maps all 21 query names to their type classification and evaluation parameters.

`decompose.py` splits a synthetic wide table (one row per guid, 70 columns) back into 12 individual reporting parquet files. Sparsity masks are applied so that only guids with nonzero data appear in each output table.

## pipeline/ — CLI stages

Each module has a `main()` function and can be invoked via `uv run python -m src.pipeline.<module>`.

`build_reporting.py` builds all 19 reporting tables from raw DCA data using DuckDB SQL aggregation. It is the CLI equivalent of notebook 03.

`run_benchmark.py` executes all benchmark SQL queries against a given reporting directory and saves the results as CSVs. It is the CLI equivalent of notebook 04.

`evaluate.py` compares real and synthetic query result CSVs and outputs summary and detail evaluation CSVs.

`pe_postprocess.py` runs the full PE post-processing pipeline end to end. It loads the PE population from batch chunks or a checkpoint, runs the DP nearest-neighbor histogram, selects candidates, decomposes the result into reporting tables, executes the benchmark queries, and evaluates against ground truth. Passing `--from-checkpoint` skips the histogram computation if selection has already been completed.
