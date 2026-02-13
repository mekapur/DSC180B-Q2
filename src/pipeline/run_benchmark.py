from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import duckdb
import pandas as pd


ALL_QUERIES = [
    "avg_platform_power_c0_freq_temp_by_chassis",
    "battery_on_duration_cpu_family_gen",
    "battery_power_on_geographic_summary",
    "display_devices_connection_type_resolution_durations_ac_dc",
    "display_devices_vendors_percentage",
    "mods_blockers_by_osname_and_codename",
    "most_popular_browser_in_each_country_by_system_count",
    "on_off_mods_sleep_summary_by_cpu_marketcodename_gen",
    "persona_web_cat_usage_analysis",
    "pkg_power_by_country",
    "popular_browsers_by_count_usage_percentage",
    "ram_utilization_histogram",
    "ranked_process_classifications",
    "server_exploration_1",
    "top_10_applications_by_app_type_ranked_by_focal_time",
    "top_10_applications_by_app_type_ranked_by_system_count",
    "top_10_applications_by_app_type_ranked_by_total_detections",
    "top_10_processes_per_user_id_ranked_by_total_power_consumption",
    "top_20_most_power_consuming_processes_by_avg_power_consumed",
    "top_mods_blocker_types_durations_by_osname_and_codename",
    "userwait_top_10_wait_processes",
    "userwait_top_10_wait_processes_wait_type_ac_dc",
    "userwait_top_20_wait_processes_compare_ac_dc_unknown_durations",
    "Xeon_network_consumption",
]


INFEASIBLE_QUERIES = {
    "ranked_process_classifications",
    "top_10_processes_per_user_id_ranked_by_total_power_consumption",
    "top_20_most_power_consuming_processes_by_avg_power_consumed",
}


def adapt_sql(sql: str, reporting_dir: Path) -> str:
    def replacer(match):
        table = match.group(1)
        path = reporting_dir / f"{table}.parquet"
        return f"read_parquet('{path}')"
    return re.sub(r"reporting\.(\w+)", replacer, sql)


def run_query(
    name: str,
    queries_dir: Path,
    reporting_dir: Path,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame | None:
    qfile = queries_dir / f"{name}.json"
    if not qfile.exists():
        print(f"  SKIP {name}: query file not found")
        return None

    with open(qfile) as f:
        data = json.load(f)
        if isinstance(data, list):
            data = data[0]

    sql = data["sql"]
    adapted = adapt_sql(sql, reporting_dir)

    try:
        return con.execute(adapted).df()
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return None


def run_all(
    queries_dir: Path,
    reporting_dir: Path,
    output_dir: Path,
    query_names: list[str] | None = None,
    skip_infeasible: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    results = {}

    run_list = query_names or ALL_QUERIES

    for name in run_list:
        if skip_infeasible and name in INFEASIBLE_QUERIES:
            if verbose:
                print(f"  SKIP {name}: infeasible (stub data)")
            continue

        t0 = time.time()
        df = run_query(name, queries_dir, reporting_dir, con)
        elapsed = time.time() - t0

        if df is not None:
            results[name] = df
            df.to_csv(output_dir / f"{name}.csv", index=False)
            if verbose:
                print(f"  OK {name}: {len(df)} rows x {len(df.columns)} cols ({elapsed:.1f}s)")
        elif verbose:
            print(f"  FAIL {name} ({elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run SQL benchmark queries")
    parser.add_argument("--reporting-dir", type=Path, default=Path("data/reporting"),
                        help="Directory containing reporting Parquet files")
    parser.add_argument("--queries-dir", type=Path, default=Path("docs/queries"),
                        help="Directory containing query JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/real"),
                        help="Output directory for result CSVs")
    parser.add_argument("--queries", nargs="*", default=None,
                        help="Specific queries to run (default: all)")
    parser.add_argument("--include-infeasible", action="store_true",
                        help="Include queries that rely on stub data")
    args = parser.parse_args()

    print(f"Running benchmark: {args.reporting_dir} -> {args.output_dir}")
    results = run_all(
        args.queries_dir, args.reporting_dir, args.output_dir,
        query_names=args.queries,
        skip_infeasible=not args.include_infeasible,
    )
    print(f"\n{len(results)}/{len(ALL_QUERIES)} queries succeeded.")


if __name__ == "__main__":
    main()
