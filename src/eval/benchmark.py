"""SQL benchmark runner for evaluating synthetic data quality.

Executes the 21-query DCA benchmark workload against either real or
synthetic reporting tables. The adapt_sql function rewrites
'reporting.table_name' references to read_parquet('path') calls so
that DuckDB can execute the original Intel SQL unchanged.
"""

import json
import re
from pathlib import Path

import duckdb
import pandas as pd


def adapt_sql(sql: str, reporting_path: Path) -> str:
    """Rewrite reporting.table_name references to read_parquet() calls."""
    def replacer(match):
        table = match.group(1)
        path = reporting_path / f"{table}.parquet"
        return f"read_parquet('{path}')"

    return re.sub(r"reporting\.(\w+)", replacer, sql)


def run_query(
    name: str,
    queries_dir: Path,
    reporting_path: Path,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame | None:
    qfile = queries_dir / f"{name}.json"
    if not qfile.exists():
        print(f"Query {name}: file not found at {qfile}")
        return None
    with open(qfile) as f:
        data = json.load(f)
        if isinstance(data, list):
            data = data[0]

    sql = data["sql"]
    adapted = adapt_sql(sql, reporting_path)

    if con is None:
        con = duckdb.connect()

    try:
        return con.execute(adapted).df()
    except Exception as e:
        print(f"Query {name} failed: {e}")
        return None


def run_benchmark(
    query_names: list[str],
    queries_dir: Path,
    reporting_path: Path,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    results = {}

    for name in query_names:
        df = run_query(name, queries_dir, reporting_path, con)
        if df is not None:
            results[name] = df
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / f"{name}.csv", index=False)

    return results
