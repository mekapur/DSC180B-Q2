from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import difflib

import numpy as np
import pandas as pd
import duckdb


NONNEGATIVE_HINTS = (
    "count",
    "number",
    "duration",
    "bytes",
    "power",
    "time",
    "num",
    "rank",
    "total",
    "nrs",
)

PERCENT_HINTS = ("percent", "percentage")

INTLIKE_HINTS = ("count", "number", "num", "rank", "nrs", "instances")

CANONICALIZE_COLUMNS = {
    "chassistype",
    "countryname_normalized",
    "country",
    "os",
    "processor_class",
    "browser",
    "vendor_name",
    "app_type",
    "acdc",
    "event_name",
    "persona",
}


@dataclass
class PostprocessConfig:
    use_reference_categories: bool = True
    fuzzy_cutoff: float = 0.80
    enforce_nonnegative: bool = True
    enforce_percent_bounds: bool = True
    enforce_intlike_rounding: bool = True
    fill_numeric_nan: bool = True
    fill_string_nan: bool = True


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _has_hint(name: str, hints: tuple[str, ...]) -> bool:
    lname = name.lower()
    return any(h in lname for h in hints)


def _coerce_numeric(series: pd.Series, col: str, cfg: PostprocessConfig) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")

    if cfg.fill_numeric_nan and out.isna().any():
        finite = out[np.isfinite(out)]
        fill = float(finite.median()) if len(finite) else 0.0
        out = out.fillna(fill)

    if cfg.enforce_nonnegative and _has_hint(col, NONNEGATIVE_HINTS):
        out = out.clip(lower=0.0)

    if cfg.enforce_percent_bounds and _has_hint(col, PERCENT_HINTS):
        out = out.clip(lower=0.0, upper=100.0)

    if cfg.enforce_intlike_rounding and _has_hint(col, INTLIKE_HINTS):
        out = np.rint(out).astype(np.int64)

    return out


def _normalize_text(series: pd.Series, cfg: PostprocessConfig) -> pd.Series:
    out = series.astype("string")
    if cfg.fill_string_nan:
        out = out.fillna("Unknown")
    out = out.str.strip()
    out = out.replace("", "Unknown")
    return out


def _canonicalize_to_reference(
    series: pd.Series,
    reference_values: set[str],
    cutoff: float,
) -> pd.Series:
    if not reference_values:
        return series

    ref_lower_to_canonical = {v.lower(): v for v in reference_values}
    ref_lowers = list(ref_lower_to_canonical.keys())

    def map_one(val: str) -> str:
        key = val.lower()
        if key in ref_lower_to_canonical:
            return ref_lower_to_canonical[key]

        close = difflib.get_close_matches(key, ref_lowers, n=1, cutoff=cutoff)
        if close:
            return ref_lower_to_canonical[close[0]]

        return val

    return series.apply(map_one)


def postprocess_table(
    synth_df: pd.DataFrame,
    table_name: str,
    reference_categories: dict[str, set[str]] | None = None,
    cfg: PostprocessConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or PostprocessConfig()
    out = synth_df.copy()

    for col in out.columns:
        s = out[col]
        if _is_numeric(s):
            out[col] = _coerce_numeric(s, col, cfg)
        else:
            out[col] = _normalize_text(s, cfg)

    if cfg.use_reference_categories and reference_categories:
        for col in out.columns:
            if col not in CANONICALIZE_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                continue

            ref_vals = reference_categories.get(col, set())
            out[col] = _canonicalize_to_reference(
                out[col].astype(str),
                reference_values=ref_vals,
                cutoff=cfg.fuzzy_cutoff,
            )

    if "guid" in out.columns:
        out["guid"] = out["guid"].astype(str).str.lower().str.replace(r"[^0-9a-f]", "", regex=True)

    return out


def _collect_reference_categories(real_path: Path, columns: list[str]) -> dict[str, set[str]]:
    if not real_path.exists() or not columns:
        return {}

    refs: dict[str, set[str]] = {}
    con = duckdb.connect()
    for col in columns:
        try:
            q = f"SELECT DISTINCT CAST({col} AS VARCHAR) AS v FROM read_parquet('{real_path}') WHERE {col} IS NOT NULL"
            vals = con.execute(q).df()["v"].astype(str).str.strip()
            refs[col] = {v for v in vals.tolist() if v}
        except Exception:
            refs[col] = set()
    return refs


def postprocess_reporting_dir(
    real_reporting_dir: Path,
    synth_reporting_dir: Path,
    output_dir: Path,
    cfg: PostprocessConfig | None = None,
) -> dict[str, Path]:
    cfg = cfg or PostprocessConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for synth_path in sorted(synth_reporting_dir.glob("*.parquet")):
        table_name = synth_path.stem
        real_path = real_reporting_dir / synth_path.name

        synth_df = pd.read_parquet(synth_path)
        candidate_cols = [
            c for c in synth_df.columns
            if c in CANONICALIZE_COLUMNS and not pd.api.types.is_numeric_dtype(synth_df[c])
        ]
        ref_categories = _collect_reference_categories(real_path, candidate_cols) if cfg.use_reference_categories else {}
        post_df = postprocess_table(
            synth_df,
            table_name,
            reference_categories=ref_categories,
            cfg=cfg,
        )

        out_path = output_dir / synth_path.name
        post_df.to_parquet(out_path, index=False)
        written[table_name] = out_path

    return written

