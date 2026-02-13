# data/

All data files in this directory are gitignored. Raw download size is ~20.7 GiB, and the full populated `data/` directory is ~32 GiB after running synthesis/evaluation stages. Only this README and manifest JSON files are tracked. The directory is populated by downloading raw data from Globus, then running the pipeline stages described below.

## How to reproduce

1. Set up [Globus Connect Personal](https://www.globus.org/globus-connect-personal)
2. Access the HDSI Industry Data Repository collection at path `/projects/dca/`
3. Transfer the files listed in the "Download list" section below into `data/raw/`
4. Run the pipeline (either notebooks or CLI):

```bash
# Option A: notebooks (interactive)
# Run notebooks 03 and 04 in order

# Option B: CLI (batch)
uv run python -m src.pipeline.build_reporting --raw-dir data/raw --out-dir data/reporting
uv run python -m src.pipeline.run_benchmark --reporting-dir data/reporting --output-dir data/results/real
```

## Directory structure

After running the full pipeline, the following directories are populated. All data files are gitignored.

### `raw/` — Downloaded from Globus (~20.7 GiB)

| Contents | Notes |
|---|---|
| `system_sysinfo_unique_normalized/` (8 parquets) | Anchor table, 1M guids |
| `data_dictionary/` (1 parquet) | Schema metadata |
| `hw_metric_stats/` (1 parquet + manifest) | Power, C0, frequency, temperature metrics |
| `os_network_consumption_v2/` (1 parquet + manifest) | Network bytes/sec |
| `os_memsam_avail_percent/` (1 parquet + manifest) | Memory utilization |
| `web_cat_usage_v2/` (1 parquet + manifest) | Browser usage by category |
| `web_cat_pivot/` (8 parquets + manifest) | Pre-pivoted web categories |
| `userwait_v2/` (1 parquet + manifest) | User wait events |
| 8 `*.txt000.gz` files | Dec 2024 update (CPU metadata, battery, display, blockers, etc.) |

### `reporting/` — Built by notebook 03 (~11.5 GiB)

19 reporting parquet files aggregated from raw data, each named `system_*.parquet`. Also contains:

| Contents | Producer |
|---|---|
| `wide_training_table.parquet` (1M rows, 70 cols) | Notebook 05 |
| `synth_wide_training_table.parquet` | Notebook 05 |
| `synthetic/` (12 parquets) | Notebook 05 (wide-table DP-SGD decomposition) |
| `synth_pertable/` (19 parquets) | Notebook 08 (per-table DP-SGD) |
| `synth_mst/` (19 parquets) | Notebook 09 (MST baseline) |
| `pe_wide_table.parquet` | Notebook 06 (Private Evolution selected wide table) |
| `pe/` (up to 12 parquets) | Notebook 06 / `scripts/evaluate_pe.py` |

### `models/` — Model checkpoints

| File | Producer |
|---|---|
| `dp_vae_checkpoint.pt`, `transformer.pkl`, `training_curves.png` | Notebook 05 |
| `pertable/sysinfo_vae.pt`, `pertable/cpu_metadata_vae.pt` | Notebook 08 |

### `batch_jobs/` — OpenAI Batch API artifacts

Up to ~24 `batch_random_chunk*.parquet` files (150K PE records total). Produced by notebook 06.

### `pe_checkpoints/` — PE checkpoint state

| File | Contents |
|---|---|
| `checkpoint.json` | Stage metadata (current stage, timing) |
| `population_iter0.parquet` | Full 150K PE population |
| `histogram_iter0.npy` | DP NN histogram (150K entries) |
| `selected_iter0.parquet` | Top 50K selected candidates |

### `results/` — Query evaluation results

| Directory / file | Producer | Contents |
|---|---|---|
| `real/` | Notebook 04 | 24 ground truth CSVs |
| `synthetic/` | Notebook 05 | 8 wide-table DP-SGD query result CSVs |
| `synth_pertable/` | Notebook 08 | 21 per-table DP-SGD query result CSVs |
| `synth_mst/` | Notebook 09 | 21 MST query result CSVs |
| `synth_pe/` | `scripts/evaluate_pe.py` | Up to 21 PE query result CSVs |
| `evaluation_widetable.csv` / `*_detail.csv` | Notebook 05 | Scoring for wide-table DP-SGD |
| `evaluation_pertable.csv` / `*_detail.csv` | Notebook 08 | Scoring for per-table DP-SGD |
| `evaluation_mst.csv` / `*_detail.csv` | Notebook 09 | Scoring for MST baseline |
| `evaluation_pe.csv` / `*_detail.csv` | `scripts/evaluate_pe.py` | Scoring for Private Evolution |

## What produces what

| Producer | Inputs | Outputs |
|---|---|---|
| Globus download | HDSI repository | `raw/` (parquets + gzips) |
| Notebook 03 / `build_reporting.py` | `raw/` | `reporting/*.parquet` (19 tables) |
| Notebook 04 / `run_benchmark.py` | `reporting/`, `docs/queries/` | `results/real/*.csv` (24 CSVs) |
| Notebook 05 (wide-table DP-SGD) | `reporting/` | `reporting/wide_training_table.parquet`, `reporting/synth_wide_training_table.parquet`, `reporting/synthetic/*.parquet`, `results/synthetic/*.csv`, `results/evaluation_widetable.csv`, `models/dp_vae_checkpoint.pt` |
| Notebook 06 (Private Evolution) | `reporting/wide_training_table.parquet`, `.env` (API key) | `batch_jobs/batch_random_chunk*.parquet`, `pe_checkpoints/`, `reporting/pe_wide_table.parquet`, `reporting/pe/*.parquet`, `results/synth_pe/*.csv` |
| Notebook 07 (PE chunk analysis) | `batch_jobs/` | (analysis only, no output files) |
| Notebook 08 (per-table DP-SGD) | `reporting/` | `reporting/synth_pertable/*.parquet`, `results/synth_pertable/*.csv`, `results/evaluation_pertable.csv`, `models/pertable/*.pt` |
| Notebook 09 (MST baseline) | `reporting/` | `reporting/synth_mst/*.parquet`, `results/synth_mst/*.csv`, `results/evaluation_mst.csv` |
| Notebook 10 (evaluation) | `results/evaluation_*.csv` | (analysis only, no output files) |
| `src.pipeline.pe_postprocess` | `reporting/wide_training_table.parquet` and/or `pe_checkpoints/` | `reporting/pe_wide_table.parquet`, `reporting/pe/*.parquet`, `results/synth_pe/*.csv`, `results/evaluation_pe.csv` |
| `scripts/evaluate_pe.py` | `reporting/pe_wide_table.parquet` or `pe_checkpoints/` | `reporting/pe/*.parquet`, `results/synth_pe/*.csv`, `results/evaluation_pe.csv` |
| `scripts/generate_report_data.py` | `results/evaluation_*.csv` | (stdout: TikZ data for report figures) |

## Evaluation CSV format

Summary CSVs (`evaluation_*.csv`) have columns: `query`, `type`, `score`, `passed`, `n_metrics`, `error`.

Detail CSVs (`evaluation_*_detail.csv`) have columns: `query`, `column`, `metric`, `value`, `threshold`, `passed`.

A query passes if $\geq 50\%$ of its per-column metrics meet their thresholds: relative error $\leq 0.25$, total variation $\leq 0.15$, Spearman's $\rho \geq 0.5$, Jaccard $\geq 0.5$.

## Download list

### From `university_analysis_pad/`

| Path | Size | Notes |
|---|---|---|
| `system_sysinfo_unique_normalized/` (all 8 parquets) | 77.5 MiB | Anchor table, 1M clients |
| `data_dictionary/` (all parquets) | 40 KiB | Schema reference |

### From `university_prod/`

Download only the first parquet file from each (partial samples):

| Path | Size | Notes |
|---|---|---|
| `hw_metric_stats/0000_part_00.parquet` | 1.17 GiB | Hardware metrics (power, C0, freq, temp) |
| `os_network_consumption_v2/0000_part_00.parquet` | 1.81 GiB | Network bytes/sec |
| `os_memsam_avail_percent/0000_part_00.parquet` | 2.03 GiB | Memory utilization |
| `web_cat_usage_v2/0000_part_00.parquet` | 864 MiB | Browser usage by category |
| `web_cat_pivot/` (all 8 parquets) | 55.4 MiB | Pre-pivoted web categories |
| `userwait_v2/0000_part_00.parquet` | 4.89 GiB | User wait events (1 of 16 files) |

Also download the `-manifest.json` for each table (a few KB each).

### From `university_prod/dca_update_dec_2024/`

Gzipped text files from Intel's December 2024 data update. Place in `data/raw/`:

| Path | Size | Notes |
|---|---|---|
| `system_cpu_metadata/system_cpu_metadata.txt000.gz` | 42.5 MiB | CPU specs per client |
| `system_os_codename_history/system_os_codename_history.txt000.gz` | 17.6 MiB | Windows version history |
| `guids_on_off_suspend_time_day/guids_on_off_suspend_time_day.txt000.gz` | 16.8 MiB | Daily on/off/sleep times |
| `mods_sleepstudy_top_blocker_hist/mods_sleepstudy_top_blocker_hist.txt000.gz` | 1.89 GiB | Modern standby blockers |
| `__tmp_batt_dc_events/__tmp_batt_dc_events.txt000.gz` | 12 MiB | Battery DC events |
| `display_devices/display_devices.txt000.gz` | 6.16 GiB | Display device usage |
| `__tmp_fgnd_apps_date/__tmp_fgnd_apps_date.txt003.gz` | 1.53 GiB | Foreground app usage (file 3 of 4) |
| `mods_sleepstudy_power_estimation_data_13wks/mods_sleepstudy_power_estimation_data_13wks.txt000.gz` | 218 KB | Power consumption (stub: 1 guid only) |

Expected total download: ~20.7 GiB.
