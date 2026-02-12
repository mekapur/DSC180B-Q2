"""
Build reporting tables from raw DCA Parquet/CSV data.

Usage:
    uv run python -m src.pipeline.build_reporting --raw-dir data/raw --out-dir data/reporting

Reads raw Parquet files and gzipped CSVs from the DCA dataset,
applies the transformations documented in Intel's ETL SQL, and
writes per-table Parquet files matching the `reporting.system_*`
schema that the 24 benchmark queries expect.
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import duckdb


def _find_raw_parquets(raw_dir: Path, table_name: str) -> str:
    folder = raw_dir / table_name
    if folder.is_dir():
        return f"'{folder}/*.parquet'"
    single = raw_dir / f"{table_name}.parquet"
    if single.exists():
        return f"'{single}'"
    raise FileNotFoundError(f"No parquet data found for {table_name} in {raw_dir}")


def _find_gz(raw_dir: Path, table_name: str) -> str:
    folder = raw_dir / table_name
    if folder.is_dir():
        gz_files = sorted(folder.glob("*.gz"))
        if gz_files:
            return f"'{gz_files[0]}'"
    for pattern in [f"{table_name}.txt000.gz", f"{table_name}.txt*.gz"]:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return f"'{matches[0]}'"
    update_dir = raw_dir / "dca_update_dec_2024" / table_name
    if update_dir.is_dir():
        gz_files = sorted(update_dir.glob("*.gz"))
        if gz_files:
            return f"'{gz_files[0]}'"
    raise FileNotFoundError(f"No gz data found for {table_name} in {raw_dir}")


REPORTING_TABLE_BUILDERS: dict[str, dict] = {}


def _register(name: str, source_type: str = "parquet", source_table: str | None = None):
    def decorator(fn):
        REPORTING_TABLE_BUILDERS[name] = {
            "fn": fn,
            "source_type": source_type,
            "source_table": source_table or name,
        }
        return fn
    return decorator


# Builders that need access to raw_dir (for cross-table joins) accept it as
# a third parameter.  The build_all() dispatcher passes raw_dir when the
# builder's signature expects it.


@_register("system_sysinfo_unique_normalized", "parquet", "system_sysinfo_unique_normalized")
def _build_sysinfo(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_parquet({src})"


@_register("system_cpu_metadata", "gz", "system_cpu_metadata")
def _build_cpu_metadata(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_os_codename_history", "gz", "system_os_codename_history")
def _build_os_codename(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_on_off_suspend_time_day", "gz", "guids_on_off_suspend_time_day")
def _build_onoff(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_display_devices", "gz", "display_devices")
def _build_display(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_frgnd_apps_types", "gz", "__tmp_fgnd_apps_date")
def _build_frgnd(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_mods_top_blocker_hist", "gz", "mods_sleepstudy_top_blocker_hist")
def _build_blocker(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"""
        SELECT *, dt_utc AS dt
        FROM read_csv({src}, auto_detect=true, ignore_errors=true)
    """


@_register("system_mods_power_consumption", "gz", "mods_sleepstudy_power_estimation_data_13wks")
def _build_mods_power(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_csv({src}, auto_detect=true, ignore_errors=true)"


@_register("system_batt_dc_events", "gz", "__tmp_batt_dc_events")
def _build_batt(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"""
        SELECT
            guid,
            CAST(power_on_dc_ts AS DATE) AS dt,
            SUM(duration_mins) AS duration_mins,
            COUNT(*) AS num_power_ons,
            MAX(start_batt_prcnt) AS max_start_batt_prcnt,
            MIN(start_batt_prcnt) AS min_start_batt_prcnt,
            AVG(start_batt_prcnt) AS avg_start_batt_prcnt,
            MAX(end_batt_prcnt) AS max_end_batt_prcnt,
            MIN(end_batt_prcnt) AS min_end_batt_prcnt,
            AVG(end_batt_prcnt) AS avg_end_batt_prcnt
        FROM read_csv({src}, auto_detect=true, ignore_errors=true)
        GROUP BY guid, CAST(power_on_dc_ts AS DATE)
    """


@_register("system_network_consumption", "parquet", "os_network_consumption_v2")
def _build_network(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"""
        SELECT
            guid, dt, input_description AS input_desc,
            SUM(nr_samples) AS nrs,
            MIN(min_bytes_sec) AS min_bytes_sec,
            MAX(max_bytes_sec) AS max_bytes_sec,
            SUM(nr_samples * avg_bytes_sec) / SUM(nr_samples) AS avg_bytes_sec
        FROM read_parquet({src})
        GROUP BY guid, dt, input_description
    """


@_register("system_userwait", "parquet", "userwait_v2")
def _build_userwait(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"""
        SELECT
            guid, dt, event_name, ac_dc_event_name,
            UPPER(SUBSTRING(ac_dc_event_name, 1, 2)) AS acdc,
            proc_name_current AS proc_name,
            COUNT(*) AS number_of_instances,
            SUM(duration_ms) AS total_duration_ms
        FROM read_parquet({src})
        GROUP BY guid, dt, event_name, ac_dc_event_name, proc_name_current
    """


@_register("system_web_cat_usage", "parquet", "web_cat_usage_v2")
def _build_web_usage(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"SELECT * FROM read_parquet({src})"


@_register("system_memory_utilization", "parquet", "os_memsam_avail_percent")
def _build_memory(con: duckdb.DuckDBPyConnection, src: str, raw_dir: Path | None = None) -> str:
    if raw_dir is None:
        raise ValueError("system_memory_utilization requires raw_dir for sysinfo JOIN")
    sysinfo_src = _find_raw_parquets(raw_dir, "system_sysinfo_unique_normalized")
    return f"""
        SELECT
            a.guid, a.dt,
            SUM(a.sample_count) AS nrs,
            CAST(b.ram * 1024 AS BIGINT) AS sysinfo_ram,
            ROUND(
                (CAST(b.ram * 1024 AS BIGINT) -
                 SUM(a.sample_count * a.average) / SUM(a.sample_count))
                * 100.0 / CAST(b.ram * 1024 AS BIGINT)
            ) AS avg_percentage_used
        FROM read_parquet({src}) a
        INNER JOIN read_parquet({sysinfo_src}) b ON a.guid = b.guid
        WHERE b.ram != 0
        GROUP BY a.guid, a.dt, b.ram
    """


def _hw_metric_sql(src: str, metric_names: list[str], prefix: str) -> str:
    name_filter = " OR ".join(f"name = '{n}'" for n in metric_names)
    return f"""
        SELECT guid, dt,
            SUM(nrs) AS nrs,
            SUM(nrs * min) / SUM(nrs) AS min_{prefix},
            SUM(nrs * mean) / SUM(nrs) AS avg_{prefix},
            SUM(nrs * max) / SUM(nrs) AS max_{prefix}
        FROM read_parquet({src})
        WHERE {name_filter}
        GROUP BY guid, dt
    """


@_register("system_psys_rap_watts", "parquet", "hw_metric_stats")
def _build_psys(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return _hw_metric_sql(
        src,
        ["HW::PACKAGE:RAP:WATTS:", "HW:::PSYS_RAP:WATTS:"],
        "psys_rap_watts",
    )


@_register("system_pkg_C0", "parquet", "hw_metric_stats")
def _build_c0(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return _hw_metric_sql(
        src,
        ["HW::PACKAGE:C0_RESIDENCY:PERCENT:"],
        "pkg_c0",
    )


@_register("system_pkg_avg_freq_mhz", "parquet", "hw_metric_stats")
def _build_freq(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return _hw_metric_sql(
        src,
        ["HW::CORE:AVG_FREQ:MHZ:"],
        "avg_freq_mhz",
    )


@_register("system_pkg_temp_centigrade", "parquet", "hw_metric_stats")
def _build_temp(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return _hw_metric_sql(
        src,
        ["HW::CORE:TEMPERATURE:CENTIGRADE:"],
        "temp_centigrade",
    )


@_register("system_hw_pkg_power", "parquet", "hw_metric_stats")
def _build_pkg_power(con: duckdb.DuckDBPyConnection, src: str) -> str:
    return f"""
        SELECT guid, dt, instance,
            SUM(nrs) AS nrs,
            SUM(nrs * mean) / SUM(nrs) AS mean,
            MAX(max) AS max
        FROM read_parquet({src})
        WHERE name = 'HW::PACKAGE:IA_POWER:WATTS:'
        GROUP BY guid, dt, instance
    """


WEB_CATEGORIES = [
    ("education", "education_education"),
    ("finance", "finance_banking_and_accounting"),
    ("mail", "mail_mail"),
    ("news", "news_news_media"),
    ("unclassified", "unclassified_unclassified"),
    ("private", "private_private"),
    ("reference", "reference_reference"),
    ("search", "search_search"),
    ("shopping", "shopping_ecommerce_auction"),
    ("recreation_travel", "recreation_travel_recreation_travel"),
    ("content_creation_photo_edit_creation", "content_creation_photo_edit_creation"),
    ("content_creation_video_audio_edit_creation", "content_creation_video_audio_edit_creation"),
    ("content_creation_web_design_development", "content_creation_web_design_development"),
    ("entertainment_music_audio_streaming", "entertainment_music_audio_streaming"),
    ("entertainment_other", "entertainment_other"),
    ("entertainment_video_streaming", "entertainment_video_streaming"),
    ("games_other", "games_other"),
    ("games_video_games", "games_video_games"),
    ("productivity_crm", "productivity_crm"),
    ("productivity_other", "productivity_other"),
    ("productivity_presentations", "productivity_presentations"),
    ("productivity_programming", "productivity_programming"),
    ("productivity_project_management", "productivity_project_management"),
    ("productivity_spreadsheets", "productivity_spreadsheets"),
    ("productivity_word_processing", "productivity_word_processing"),
    ("social_social_network", "social_social_network"),
    ("social_communication", "social_communication"),
    ("social_communication_live", "social_communication_live"),
]


@_register("system_web_cat_pivot_duration", "parquet", "web_cat_pivot")
def _build_pivot(con: duckdb.DuckDBPyConnection, src: str) -> str:
    aliases = []
    for short_name, long_name in WEB_CATEGORIES:
        aliases.append(f'    "{long_name}" AS {short_name}')
    alias_str = ",\n".join(aliases)
    return f"""
        SELECT guid, dt,
{alias_str}
        FROM read_parquet({src})
    """


def build_all(
    raw_dir: Path,
    out_dir: Path,
    tables: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    counts = {}

    build_list = tables or list(REPORTING_TABLE_BUILDERS.keys())

    for name in build_list:
        if name not in REPORTING_TABLE_BUILDERS:
            if verbose:
                print(f"  SKIP {name}: no builder registered")
            continue

        info = REPORTING_TABLE_BUILDERS[name]
        source_table = info["source_table"]
        source_type = info["source_type"]
        builder_fn = info["fn"]

        try:
            if source_type == "parquet":
                src = _find_raw_parquets(raw_dir, source_table)
            else:
                src = _find_gz(raw_dir, source_table)
        except FileNotFoundError as e:
            if verbose:
                print(f"  SKIP {name}: {e}")
            continue

        t0 = time.time()
        try:
            sig = inspect.signature(builder_fn)
            if "raw_dir" in sig.parameters:
                sql = builder_fn(con, src, raw_dir=raw_dir)
            else:
                sql = builder_fn(con, src)

            df = con.execute(sql).df()
            out_path = out_dir / f"{name}.parquet"
            df.to_parquet(out_path, index=False)
            counts[name] = len(df)

            elapsed = time.time() - t0
            if verbose:
                print(f"  OK {name}: {len(df):,} rows ({elapsed:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"  FAIL {name}: {e}")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Build reporting tables from raw DCA data")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="Directory containing raw Parquet/gz files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/reporting"),
                        help="Output directory for reporting Parquet files")
    parser.add_argument("--tables", nargs="*", default=None,
                        help="Specific tables to build (default: all)")
    args = parser.parse_args()

    print(f"Building reporting tables from {args.raw_dir} -> {args.out_dir}")
    counts = build_all(args.raw_dir, args.out_dir, args.tables)
    print(f"\nBuilt {len(counts)} tables successfully.")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count:,} rows")


if __name__ == "__main__":
    main()
