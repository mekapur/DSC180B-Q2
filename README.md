<div align="center">

# Training-based versus training-free differential privacy for data synthesis

[Mehak Kapur](https://github.com/mekapur), [Hana Tjendrawasi](https://github.com/hanajuliatj), [Jason Tran](https://github.com/jktrn), [Phuc Tran](https://github.com/21phuctran), Yu-Xiang Wang
`{mekapur,htjendrawasi,jat037,pct001,yuxiangw}@ucsd.edu`

</div>

### Abstract

Differentially private synthetic data generation promises to resolve the tension between data utility and individual privacy, enabling the release of datasets that preserve the statistical properties analysts need while bounding what any adversary can learn about a single record. Two paradigms have emerged to fulfill this promise. Training-based methods inject calibrated noise during model optimization, coupling privacy to the learning process itself. Training-free methods instead leverage foundation models through black-box API access, achieving privacy through selection mechanisms that never touch the model's parameters. Both have demonstrated success on image and text benchmarks, yet their behavior on realistic, multi-table relational data remains largely unexplored. We investigate both approaches on Intel's Driver and Client Applications (DCA) telemetry corpus, evaluating against a benchmark of 21 analytical SQL queries representative of production business intelligence workloads.

View the full report [here](report/q2-report.pdf).

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

1. If you don't have uv installed:

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or with Homebrew
   brew install uv

   # Or with pip
   pip install uv
   ```

2. Create your virtual environment:

   ```bash
   uv venv
   uv sync
   ```

   This creates a `.venv` directory and installs all dependencies from `pyproject.toml`.

3. Activate the environment:

   ```bash
   # macOS/Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Windows (cmd)
   .venv\Scripts\activate.bat
   ```

4. To run the notebooks in Jupyter/Visual Studio Code, register the virtual environment as a Jupyter kernel. Make sure the `.venv` is activated, then:

   ```bash
   python -m ipykernel install --user --name dsc180-q2 --display-name "DSC 180B Q2"
   ```

   Now you can select "DSC 180B Q2" as the kernel when opening notebooks.

### Data acquisition

The DCA telemetry data (~20.7 GiB) is hosted on the HDSI Industry Data Repository and transferred via [Globus](https://www.globus.org/). The data is not included in this repository. See [`data/README.md`](data/README.md) for the complete download list and reproduction instructions.

### Project structure

```
dsc180-q2/
├── src/
│   ├── pe/                                # Private Evolution pipeline
│   │   ├── api.py                         # RANDOM_API / VARIATION_API via OpenAI Batch
│   │   ├── distance.py                    # Workload-aware distance function
│   │   ├── histogram.py                   # DP nearest-neighbor histogram + PE loop
│   │   └── privacy.py                     # Analytic Gaussian mechanism (Balle-Wang)
│   ├── eval/                              # Evaluation framework
│   │   ├── benchmark.py                   # SQL query runner with table adaptation
│   │   ├── compare.py                     # Query discrepancy metrics (RE, TV, Spearman)
│   │   └── decompose.py                   # Wide table -> reporting table decomposition
│   └── pipeline/                          # CLI-runnable pipeline stages
│       ├── build_reporting.py             # Raw data -> 19 reporting tables
│       ├── run_benchmark.py               # Execute SQL queries on any reporting dir
│       └── evaluate.py                    # Compare real vs synthetic results
│
├── notebooks/
│   ├── 01-data-exploration.ipynb          # Validate downloaded data, check schemas
│   ├── 02-query-feasibility.ipynb         # Verify benchmark queries against schemas
│   ├── 03-build-reporting-tables.ipynb    # Build 19 reporting tables via DuckDB
│   ├── 04-run-benchmark-queries.ipynb     # Execute queries, save ground truth CSVs
│   ├── 05-dp-sgd.ipynb                    # Wide-table DP-VAE (epsilon=4.0)
│   ├── 06-private-evolution.ipynb         # PE via OpenAI Batch API (gpt-5-nano)
│   ├── 07-pe-chunk-analysis.ipynb         # Inspect PE batch generation progress
│   ├── 08-per-table-dpsgd.ipynb           # Per-table DP histogram synthesis
│   └── 09-mst-baseline.ipynb             # MST marginal-based baseline
│
├── report/
│   ├── q2-report.tex                      # Main Q2 report (XeLaTeX)
│   ├── q2-proposal.tex
│   ├── q1-report.tex
│   ├── reference.bib
│   └── style/dsc180reportstyle.sty
│
├── docs/
│   ├── queries/                           # 24 benchmark SQL queries (JSON)
│   └── papers/                            # Reference papers and reading notes
│
├── data/                                  # Gitignored except README and manifests
│   ├── README.md                          # Download instructions and file manifest
│   ├── raw/                               # Parquet and gzipped source files (~20.7 GiB)
│   ├── reporting/                         # 19 derived reporting tables (~11.5 GiB)
│   ├── models/                            # Model checkpoints (DP-VAE, per-table VAEs)
│   ├── batch_jobs/                        # OpenAI Batch API artifacts (PE chunks)
│   └── results/
│       ├── real/                          # Ground truth query results (24 CSVs)
│       ├── synthetic/                     # Wide-table DP-SGD results
│       └── synth_pertable/                # Per-table DP-SGD results
│
├── dsc-180a-q1/                           # Git submodule: Q1 DP-VAE implementation
├── pyproject.toml
└── uv.lock
```

---

## Usage

### 1. Download the data

Follow the instructions in [`data/README.md`](data/README.md) to transfer files from Globus. Total download is approximately 20.7 GiB.

### 2. Run the pipeline

The pipeline can be run via the numbered notebooks (for interactive exploration) or via standalone scripts (for batch execution). Each stage reads from the outputs of the previous one.

Standalone scripts:

```bash
# Build 19 reporting tables from raw data
uv run python -m src.pipeline.build_reporting --raw-dir data/raw --out-dir data/reporting

# Execute benchmark queries on real data (ground truth)
uv run python -m src.pipeline.run_benchmark \
    --reporting-dir data/reporting \
    --output-dir data/results/real

# Execute benchmark queries on synthetic data
uv run python -m src.pipeline.run_benchmark \
    --reporting-dir data/reporting/synth_pertable \
    --output-dir data/results/synth_pertable

# Compare real vs synthetic results
uv run python -m src.pipeline.evaluate \
    --real-dir data/results/real \
    --synth-dir data/results/synth_pertable \
    --output data/results/evaluation_pertable.csv
```

Equivalent notebook workflow:

```bash
jupyter execute notebooks/03-build-reporting-tables.ipynb
jupyter execute notebooks/04-run-benchmark-queries.ipynb
jupyter execute notebooks/05-dp-sgd.ipynb          # Wide-table DP-VAE
jupyter execute notebooks/08-per-table-dpsgd.ipynb  # Per-table DP histograms
```

Notebooks 01 and 02 are validation/exploration notebooks and do not need to be re-run for the pipeline to work.

### 3. Build the report

The report requires XeLaTeX and the [Fira Code](https://github.com/tonsky/FiraCode) font.

```bash
# Install Fira Code (macOS)
brew install --cask font-fira-code

# Build the PDF
cd report
latexmk q2-report.tex
```

The `.latexmkrc` in `report/` configures latexmk to use XeLaTeX.

---

## Pipeline overview

The pipeline transforms raw DCA telemetry into differentially private synthetic data and evaluates it against a SQL benchmark.

Data preparation (notebooks 01-04): Raw Parquet and gzipped text files are ingested via DuckDB, aggregated into 19 reporting tables matching Intel's `reporting.system_*` schema, and used to execute 24 analytical SQL queries that produce ground truth results.

Wide-table DP-SGD (notebook 05): All 19 reporting tables are joined into a single wide table (1,000,000 rows, 70 columns) keyed on `guid`. A variational autoencoder is trained with DP-SGD via Opacus (epsilon=4.0, delta=1e-5). The synthetic wide table is decomposed back into reporting table schemas and the same benchmark queries are re-executed for comparison.

Per-table DP-SGD (notebook 08): Each reporting table is synthesized independently using DP histogram mechanisms calibrated to epsilon=4.0 per table. This avoids the zero-inflation problem of the wide table but sacrifices cross-table correlations.

Private Evolution (notebook 06): A training-free approach using the OpenAI Batch API (gpt-5-nano) with workload-aware distance functions (Swanberg et al., 2025). The foundation model generates synthetic records that are selected via a differentially private nearest-neighbor histogram. Uses T=1 iteration (optimal for tabular per convergence theory).

MST baseline (notebook 09): Marginal-based synthetic data generation using the MST algorithm (McKenna et al., 2021) as a non-neural baseline.

Evaluation (src/eval/compare.py): Each synthetic method is scored on query discrepancy using relative error for scalars, total variation distance for distributions, and Spearman rank correlation for rankings.

---

## Benchmark queries

The evaluation benchmark consists of 21 feasible SQL queries developed by Intel analysts, spanning five categories:

| Category | Count | Tests |
|---|---|---|
| Aggregate statistics with joins | 6 | Cross-table correlation preservation |
| Ranked top-k | 7 | Relative ordering preservation |
| Geographic/demographic breakdowns | 4 | Conditional distribution preservation |
| Histograms and distributions | 2 | Distributional shape preservation |
| Complex multi-way pivots | 2 | High-dimensional joint distributions |

Three additional queries are permanently infeasible due to missing power consumption data (single-guid stub only).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `duckdb` | >=1.4.4 | SQL engine for data ingestion and query execution |
| `pandas`, `pyarrow` | >=3.0.0, >=23.0.0 | Data manipulation and Parquet I/O |
| `torch` | >=2.10.0 | DP-VAE model architecture |
| `opacus` | >=1.5.4 | DP-SGD (per-sample gradient clipping + noise injection) |
| `openai` | >=2.20.0 | PE pipeline: Batch API with structured outputs (gpt-5-nano) |
| `scikit-learn` | >=1.8.0 | Preprocessing (ColumnTransformer, StandardScaler, OneHotEncoder) |
| `scipy` | >=1.17.0 | Spearman correlation, Gaussian mechanism calibration |
| `matplotlib` | >=3.10.8 | Evaluation visualizations |
| `jupyter`, `ipykernel` | >=1.1.1, >=7.2.0 | Notebook execution |
