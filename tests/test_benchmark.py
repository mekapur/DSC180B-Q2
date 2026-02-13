from pathlib import Path

import pytest

from src.eval.benchmark import adapt_sql


class TestAdaptSql:
    def test_single_table(self):
        sql = "SELECT * FROM reporting.system_sysinfo_unique_normalized"
        result = adapt_sql(sql, Path("/data/reporting"))
        assert "read_parquet('/data/reporting/system_sysinfo_unique_normalized.parquet')" in result
        assert "reporting." not in result

    def test_multiple_tables(self):
        sql = (
            "SELECT a.guid FROM reporting.system_sysinfo_unique_normalized a "
            "JOIN reporting.system_network_consumption b ON a.guid = b.guid"
        )
        result = adapt_sql(sql, Path("/data/reporting"))
        assert "reporting." not in result
        assert "system_sysinfo_unique_normalized.parquet" in result
        assert "system_network_consumption.parquet" in result

    def test_no_reporting_reference(self):
        sql = "SELECT 1 + 1 AS result"
        result = adapt_sql(sql, Path("/data/reporting"))
        assert result == sql

    def test_preserves_other_sql(self):
        sql = "SELECT COUNT(*) FROM reporting.system_userwait WHERE guid = '123'"
        result = adapt_sql(sql, Path("/tmp"))
        assert "WHERE guid = '123'" in result
        assert "read_parquet('/tmp/system_userwait.parquet')" in result

    def test_uses_provided_path(self):
        sql = "SELECT * FROM reporting.my_table"
        result = adapt_sql(sql, Path("/custom/path"))
        assert "/custom/path/my_table.parquet" in result
