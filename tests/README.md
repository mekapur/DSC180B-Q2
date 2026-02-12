# tests/

Unit tests for `src/` library code. Run with:

```bash
uv run pytest tests/ -v
```

All 41 tests run without downloaded data or API keys.

## Test files

| File | Module tested | Tests |
|---|---|---|
| `test_privacy.py` | `src.pe.privacy` | `analytic_gaussian_delta` monotonicity and edge cases; `calibrate_sigma` known values and round-trip with `compute_epsilon`; `compute_epsilon` monotonicity in $\sigma$ |
| `test_distance.py` | `src.pe.distance` | `WorkloadDistance` initialization and weight normalization; self nearest-neighbor returns identity; output shape matches input; chunked vs unchunked consistency; query weight application (chassistype=6, country=4) |
| `test_compare.py` | `src.eval.compare` | `relative_error` (equal values, zeros, positive); `total_variation_distance` (identical, disjoint, unnormalized); Spearman's $\rho$ (perfect correlation, anticorrelation, single element, constants); Jaccard similarity (identical, disjoint, partial overlap); `categorical_accuracy` (perfect, none, partial) |
| `test_decompose.py` | `src.eval.decompose` | `snap_ram` exact standard values; rounding to nearest standard value; boundary cases; large values |
| `test_benchmark.py` | `src.eval.benchmark` | `adapt_sql` single table replacement; multiple table JOINs; SQL without `reporting.` references (no-op); WHERE clause preservation; custom path handling |

## What is not tested

- `src.pe.api` (requires OpenAI API key and network access)
- `src.pe.histogram` (requires large DataFrames; tested implicitly via notebook 06)
- `src.pipeline.*` (integration-level; tested by running the notebooks end-to-end)
- `src.eval.decompose.decompose_wide_table` (requires a wide-table DataFrame; `snap_ram` is the unit-testable component)
