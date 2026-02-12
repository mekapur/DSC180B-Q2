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
├── notebooks/
│   ├── 01-data-exploration.ipynb          # Validate downloaded data, check schemas and guid overlap
│   ├── 02-query-feasibility.ipynb         # Verify all 24 benchmark queries against reporting schemas
│   ├── 03-build-reporting-tables.ipynb    # Aggregate raw data into 19 reporting tables via DuckDB
│   ├── 04-run-benchmark-queries.ipynb     # Execute all queries on real data, save ground truth CSVs
│   └── 05-dp-sgd.ipynb                    # DP-VAE training, synthetic generation, benchmark evaluation
│
├── report/
│   ├── q2-report.tex
│   ├── q2-proposal.tex
│   ├── q1-report.tex
│   ├── reference.bib
│   └── style/
│
├── docs/
│   ├── queries/                           # 24 benchmark SQL queries, in JSON
│   ├── papers/                            # Reference papers and reading notes
│   └── references/                        # Additional documentation
│
├── data/                                  # Gitignored except README and manifests
│   ├── README.md                          # Download instructions and file manifest
│   ├── raw/                               # Parquet and gzipped source files
│   ├── reporting/                         # 19 derived reporting tables (~11.5 GiB)
│   ├── models/                            # DP-VAE checkpoint and sklearn transformer
│   └── results/
│       ├── real/                          # Ground truth query results (24 CSVs)
│       └── synthetic/                     # Synthetic query results
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

The notebooks are numbered and should be run in order. Each notebook reads from the outputs of the previous one.

```bash
# Build 19 reporting tables from raw data
jupyter execute notebooks/03-build-reporting-tables.ipynb

# Execute all 24 benchmark queries on real data
jupyter execute notebooks/04-run-benchmark-queries.ipynb

# Train DP-VAE, generate synthetic data, evaluate against benchmark
jupyter execute notebooks/05-dp-sgd.ipynb
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

**Data preparation** (notebooks 01-04): Raw Parquet and gzipped text files are ingested via DuckDB, aggregated into 19 reporting tables matching Intel's `reporting.system_*` schema, and used to execute 24 analytical SQL queries that produce ground truth results.

**DP-SGD synthesis** (notebook 05): All 19 reporting tables are joined into a single wide table (1,000,000 rows, 70 columns) keyed on `guid`. A variational autoencoder is trained with DP-SGD using Opacus ($\varepsilon = 4.0$, $\delta = 10^{-5}$). The synthetic wide table is decomposed back into reporting table schemas and the same benchmark queries are re-executed for comparison.

**Private Evolution** (planned): A training-free alternative using black-box API access to foundation models, with privacy achieved through differentially private nearest-neighbor histograms.

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

| Package | Purpose |
|---|---|
| `duckdb` | SQL engine for data ingestion and query execution |
| `pandas`, `pyarrow` | Data manipulation and Parquet I/O |
| `torch` | DP-VAE model |
| `opacus` | DP-SGD (per-sample gradient clipping + noise injection) |
| `scikit-learn` | Preprocessing (ColumnTransformer, StandardScaler, OneHotEncoder) |
| `matplotlib` | Evaluation visualizations |
| `jupyter`, `ipykernel` | Notebook execution |
