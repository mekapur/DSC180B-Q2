import duckdb

# EDIT THESE PATHS to your parquet folders
CPU_PARQUET = "/path/to/system_cpu_metadata/*.parquet"
BATT_PARQUET = "/path/to/system_batt_dc_events/*.parquet"

OUT_PARQUET = "train_query2_rows.parquet"

con = duckdb.connect()

con.execute("CREATE SCHEMA IF NOT EXISTS reporting;")
con.execute(f"CREATE OR REPLACE VIEW reporting.system_cpu_metadata AS SELECT * FROM read_parquet('{CPU_PARQUET}');")
con.execute(f"CREATE OR REPLACE VIEW reporting.system_batt_dc_events AS SELECT * FROM read_parquet('{BATT_PARQUET}');")

# Build row-level training data (one row per guid-day battery event joined to cpu metadata)
# Keep the same filters your query implies
con.execute(f"""
COPY (
  SELECT
    b.guid,
    a.marketcodename,
    a.cpugen,
    b.dt,
    b.duration_mins
  FROM reporting.system_cpu_metadata a
  JOIN reporting.system_batt_dc_events b
    ON a.guid = b.guid
  WHERE a.cpugen IS NOT NULL
    AND a.cpugen <> 'Unknown'
    AND a.marketcodename IS NOT NULL
    AND b.duration_mins IS NOT NULL
    AND b.duration_mins >= 0
) TO '{OUT_PARQUET}' (FORMAT PARQUET);
""")

print(f"Wrote: {OUT_PARQUET}")

