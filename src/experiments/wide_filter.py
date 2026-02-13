from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "HUGEINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
}


@dataclass
class WideFilterSummary:
    input_rows: int
    output_rows: int
    numeric_cols: int
    threshold: int
    output_path: Path


def _numeric_columns_from_parquet(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> list[str]:
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").df()
    numeric_cols = []
    for _, row in desc.iterrows():
        c = str(row["column_name"])
        t = str(row["column_type"]).upper()
        if c == "guid":
            continue
        if any(t.startswith(nt) for nt in NUMERIC_TYPES):
            numeric_cols.append(c)
    return numeric_cols


def build_coverage_filtered_wide_table(
    input_path: Path,
    output_path: Path,
    min_nonzero_numeric: int = 3,
) -> WideFilterSummary:
    con = duckdb.connect()
    numeric_cols = _numeric_columns_from_parquet(con, input_path)
    if not numeric_cols:
        raise ValueError("No numeric columns found in wide table.")

    coverage_expr = " + ".join([f"CASE WHEN {c} IS NOT NULL AND {c} <> 0 THEN 1 ELSE 0 END" for c in numeric_cols])
    query = f"""
        COPY (
            SELECT *
            FROM read_parquet('{input_path}')
            WHERE ({coverage_expr}) >= {min_nonzero_numeric}
        ) TO '{output_path}' (FORMAT PARQUET)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(query)

    input_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{input_path}')").fetchone()[0])
    output_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0])

    return WideFilterSummary(
        input_rows=input_rows,
        output_rows=output_rows,
        numeric_cols=len(numeric_cols),
        threshold=min_nonzero_numeric,
        output_path=output_path,
    )


def summarize_numeric_sparsity(parquet_path: Path, top_k: int = 20) -> pd.DataFrame:
    con = duckdb.connect()
    numeric_cols = _numeric_columns_from_parquet(con, parquet_path)
    total_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()[0])
    rows = []
    for c in numeric_cols:
        nonzero = int(
            con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{parquet_path}') WHERE {c} IS NOT NULL AND {c} <> 0"
            ).fetchone()[0]
        )
        rows.append(
            {
                "column": c,
                "nonzero_rows": nonzero,
                "nonzero_rate": nonzero / total_rows if total_rows else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("nonzero_rate").head(top_k)

