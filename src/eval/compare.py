"""
Query discrepancy metrics for comparing real vs synthetic benchmark results.

Implements the evaluation framework from the Q2 proposal:
  - Relative error for scalar/aggregate columns
  - Spearman rank correlation for ranked result sets
  - Total variation distance for categorical distributions
  - Aggregate pass/fail scoring with configurable tolerances

Each query result pair (real, synthetic) is evaluated column-by-column,
producing per-column metrics and an overall query score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Query classification: maps each query to its result type and key columns
# ---------------------------------------------------------------------------

QUERY_METADATA: dict[str, dict] = {
    "avg_platform_power_c0_freq_temp_by_chassis": {
        "type": "aggregate",
        "group_cols": ["chassistype"],
        "metric_cols": [
            "number_of_systems",
            "avg_psys_rap_watts",
            "avg_pkg_c0",
            "avg_freq_mhz",
            "avg_temp_centigrade",
        ],
        "count_col": "number_of_systems",
    },
    "battery_on_duration_cpu_family_gen": {
        "type": "aggregate",
        "group_cols": ["marketcodename", "cpugen"],
        "metric_cols": [
            "number_of_systems",
            "avg_duration_mins_on_battery",
        ],
        "count_col": "number_of_systems",
    },
    "battery_power_on_geographic_summary": {
        "type": "aggregate",
        "group_cols": ["country"],
        "metric_cols": [
            "number_of_systems",
            "avg_number_of_dc_powerons",
            "avg_duration",
        ],
        "count_col": "number_of_systems",
    },
    "display_devices_connection_type_resolution_durations_ac_dc": {
        "type": "aggregate",
        "group_cols": ["connection_type", "resolution"],
        "metric_cols": [
            "number_of_systems",
            "average_duration_on_ac_in_seconds",
            "average_duration_on_dc_in_seconds",
        ],
        "count_col": "number_of_systems",
    },
    "display_devices_vendors_percentage": {
        "type": "distribution",
        "category_col": "vendor_name",
        "value_col": "percentage_of_systems",
    },
    "mods_blockers_by_osname_and_codename": {
        "type": "aggregate",
        "group_cols": ["os_name", "os_codename"],
        "metric_cols": ["num_entries", "number_of_systems", "entries_per_system"],
        "count_col": "number_of_systems",
    },
    "most_popular_browser_in_each_country_by_system_count": {
        "type": "ranking_categorical",
        "group_col": "country",
        "value_col": "browser",
    },
    "on_off_mods_sleep_summary_by_cpu_marketcodename_gen": {
        "type": "aggregate",
        "group_cols": ["marketcodename", "cpugen"],
        "metric_cols": [
            "number_of_systems",
            "avg_on_time",
            "avg_off_time",
            "avg_modern_sleep_time",
            "avg_sleep_time",
            "avg_total_time",
            "avg_pcnt_on_time",
            "avg_pcnt_off_time",
            "avg_pcnt_mods_time",
            "avg_pcnt_sleep_time",
        ],
        "count_col": "number_of_systems",
    },
    "persona_web_cat_usage_analysis": {
        "type": "aggregate",
        "group_cols": ["persona"],
        "metric_cols": [
            "number_of_systems",
            "days",
            "content_creation_photo_edit_creation",
            "content_creation_video_audio_edit_creation",
            "content_creation_web_design_development",
            "education",
            "entertainment_music_audio_streaming",
            "entertainment_other",
            "entertainment_video_streaming",
            "finance",
            "games_other",
            "games_video_games",
            "mail",
            "news",
            "unclassified",
            "private",
            "productivity_crm",
            "productivity_other",
            "productivity_presentations",
            "productivity_programming",
            "productivity_project_management",
            "productivity_spreadsheets",
            "productivity_word_processing",
            "recreation_travel",
            "reference",
            "search",
            "shopping",
            "social_social_network",
            "social_communication",
            "social_communication_live",
        ],
        "count_col": "number_of_systems",
    },
    "pkg_power_by_country": {
        "type": "aggregate",
        "group_cols": ["countryname_normalized"],
        "metric_cols": [
            "number_of_systems",
            "avg_pkg_power_consumed",
        ],
        "count_col": "number_of_systems",
    },
    "popular_browsers_by_count_usage_percentage": {
        "type": "distribution",
        "category_col": "browser",
        "value_col": "percent_systems",
        "extra_value_cols": ["percent_instances", "percent_duration"],
    },
    "ram_utilization_histogram": {
        "type": "histogram",
        "bin_col": "ram_gb",
        "count_col": "count(DISTINCT guid)",
        "metric_cols": ["avg_percentage_used"],
    },
    "server_exploration_1": {
        "type": "row_level",
        "id_col": "guid",
        "metric_cols": ["nrs", "received_bytes", "sent_bytes"],
        "cat_cols": ["chassistype", "vendor", "os"],
    },
    "Xeon_network_consumption": {
        "type": "aggregate",
        "group_cols": ["processor_class", "os"],
        "metric_cols": [
            "number_of_systems",
            "avg_bytes_received",
            "avg_bytes_sent",
        ],
        "count_col": "number_of_systems",
    },
    "top_10_applications_by_app_type_ranked_by_focal_time": {
        "type": "ranking_numeric",
        "group_col": "app_type",
        "rank_col": "rank",
        "item_col": "exe_name",
        "value_col": "average_focal_sec_per_day",
    },
    "top_10_applications_by_app_type_ranked_by_system_count": {
        "type": "ranking_numeric",
        "group_col": "app_type",
        "rank_col": "rank",
        "item_col": "exe_name",
        "value_col": "number_of_systems",
    },
    "top_10_applications_by_app_type_ranked_by_total_detections": {
        "type": "ranking_numeric",
        "group_col": "app_type",
        "rank_col": "rank",
        "item_col": "exe_name",
        "value_col": "total_number_of_detections",
    },
    "top_mods_blocker_types_durations_by_osname_and_codename": {
        "type": "aggregate",
        "group_cols": ["os_name", "os_codename", "blocker_name", "blocker_type", "activity_level"],
        "metric_cols": [
            "number_of_clients",
            "average_active_time_in_seconds",
            "number_of_occurences",
        ],
        "count_col": "number_of_clients",
    },
    "userwait_top_10_wait_processes": {
        "type": "ranking_numeric",
        "group_col": None,
        "rank_col": "rank",
        "item_col": "proc_name",
        "value_col": "total_duration_sec_per_instance",
    },
    "userwait_top_10_wait_processes_wait_type_ac_dc": {
        "type": "ranking_numeric",
        "group_col": "acdc",
        "rank_col": "rank",
        "item_col": "proc_name",
        "value_col": "total_duration_sec_per_instance",
        "extra_group_col": "event_name",
    },
    "userwait_top_20_wait_processes_compare_ac_dc_unknown_durations": {
        "type": "ranking_numeric",
        "group_col": None,
        "rank_col": None,
        "item_col": "proc_name",
        "value_col": "ac_duration",
        "extra_value_cols": ["dc_duration", "unknown_duration"],
    },
    "ranked_process_classifications": {
        "type": "ranking_numeric",
        "group_col": None,
        "rank_col": "rnk",
        "item_col": "user_id",
        "value_col": "total_power_consumption",
    },
    "top_10_processes_per_user_id_ranked_by_total_power_consumption": {
        "type": "ranking_numeric",
        "group_col": "user_id",
        "rank_col": "rnk",
        "item_col": "app_id",
        "value_col": "total_power_consumption",
    },
    "top_20_most_power_consuming_processes_by_avg_power_consumed": {
        "type": "ranking_numeric",
        "group_col": None,
        "rank_col": "rnk",
        "item_col": "app_id",
        "value_col": "total_power_consumption",
    },
}


# ---------------------------------------------------------------------------
# Metric result containers
# ---------------------------------------------------------------------------


@dataclass
class ColumnMetric:
    column: str
    metric_type: str
    value: float
    passed: bool
    detail: str = ""


@dataclass
class QueryResult:
    query_name: str
    query_type: str
    metrics: list[ColumnMetric] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    error: str = ""

    @property
    def n_passed(self) -> int:
        return sum(1 for m in self.metrics if m.passed)

    @property
    def n_total(self) -> int:
        return len(self.metrics)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def relative_error(real: float, synth: float) -> float:
    if real == 0 and synth == 0:
        return 0.0
    if real == 0:
        return float("inf")
    return abs(real - synth) / abs(real)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 and q_sum == 0:
        return 0.0
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum
    return 0.5 * np.abs(p - q).sum()


def spearman_rho(real_ranks: np.ndarray, synth_ranks: np.ndarray) -> float:
    if len(real_ranks) < 2 or len(synth_ranks) < 2:
        return 0.0
    rho, _ = stats.spearmanr(real_ranks, synth_ranks)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    if len(set_a | set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def categorical_accuracy(real_series: pd.Series, synth_series: pd.Series) -> float:
    real_keys = set(real_series.index)
    synth_keys = set(synth_series.index)
    common = real_keys & synth_keys
    if len(common) == 0:
        return 0.0
    matches = sum(1 for k in common if real_series[k] == synth_series[k])
    return matches / len(real_keys)


# ---------------------------------------------------------------------------
# Per-query-type evaluation
# ---------------------------------------------------------------------------


def _eval_aggregate(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
    rel_tol: float = 0.25,
) -> list[ColumnMetric]:
    metrics = []
    group_cols = meta["group_cols"]
    metric_cols = [c for c in meta["metric_cols"] if c in real_df.columns and c in synth_df.columns]

    real_groups = set(real_df[group_cols].apply(tuple, axis=1))
    synth_groups = set(synth_df[group_cols].apply(tuple, axis=1))
    overlap = real_groups & synth_groups

    group_jaccard = jaccard_similarity(real_groups, synth_groups)
    metrics.append(ColumnMetric(
        column="group_coverage",
        metric_type="jaccard",
        value=group_jaccard,
        passed=group_jaccard >= 0.5,
        detail=f"{len(overlap)}/{len(real_groups)} real groups matched, "
               f"{len(synth_groups)} synth groups total",
    ))

    if len(overlap) == 0:
        for col in metric_cols:
            metrics.append(ColumnMetric(
                column=col,
                metric_type="relative_error",
                value=float("inf"),
                passed=False,
                detail="no overlapping groups",
            ))
        return metrics

    real_indexed = real_df.set_index(group_cols)
    synth_indexed = synth_df.set_index(group_cols)

    def _unwrap_key(key: tuple):
        return key[0] if len(key) == 1 else key

    for col in metric_cols:
        errors = []
        for group_key in overlap:
            idx_key = _unwrap_key(group_key)
            r_val = float(real_indexed.loc[idx_key, col]) if idx_key in real_indexed.index else np.nan
            s_val = float(synth_indexed.loc[idx_key, col]) if idx_key in synth_indexed.index else np.nan
            if np.isnan(r_val) or np.isnan(s_val):
                continue
            errors.append(relative_error(r_val, s_val))

        if errors:
            median_re = float(np.median(errors))
            mean_re = float(np.mean(errors))
            metrics.append(ColumnMetric(
                column=col,
                metric_type="median_relative_error",
                value=median_re,
                passed=median_re <= rel_tol,
                detail=f"median RE={median_re:.4f}, mean RE={mean_re:.4f}, "
                       f"n_groups={len(errors)}",
            ))
        else:
            metrics.append(ColumnMetric(
                column=col,
                metric_type="median_relative_error",
                value=float("inf"),
                passed=False,
                detail="no valid comparisons",
            ))

    return metrics


def _eval_distribution(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
    tv_tol: float = 0.15,
) -> list[ColumnMetric]:
    metrics = []
    cat_col = meta["category_col"]
    value_cols = [meta["value_col"]]
    if "extra_value_cols" in meta:
        value_cols.extend(meta["extra_value_cols"])

    for vcol in value_cols:
        if vcol not in real_df.columns or vcol not in synth_df.columns:
            continue

        all_cats = sorted(set(real_df[cat_col].tolist() + synth_df[cat_col].tolist()))
        real_map = dict(zip(real_df[cat_col], real_df[vcol]))
        synth_map = dict(zip(synth_df[cat_col], synth_df[vcol]))

        p = np.array([real_map.get(c, 0.0) for c in all_cats], dtype=float)
        q = np.array([synth_map.get(c, 0.0) for c in all_cats], dtype=float)

        tvd = total_variation_distance(p, q)
        metrics.append(ColumnMetric(
            column=vcol,
            metric_type="total_variation",
            value=tvd,
            passed=tvd <= tv_tol,
            detail=f"TV={tvd:.4f}, {len(all_cats)} categories",
        ))

    return metrics


def _eval_histogram(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
    tv_tol: float = 0.15,
    rel_tol: float = 0.25,
) -> list[ColumnMetric]:
    metrics = []
    bin_col = meta["bin_col"]
    count_col = meta["count_col"]

    all_bins = sorted(set(
        real_df[bin_col].dropna().tolist() + synth_df[bin_col].dropna().tolist()
    ))
    real_counts = dict(zip(real_df[bin_col], real_df[count_col]))
    synth_counts = dict(zip(synth_df[bin_col], synth_df[count_col]))

    p = np.array([real_counts.get(b, 0) for b in all_bins], dtype=float)
    q = np.array([synth_counts.get(b, 0) for b in all_bins], dtype=float)

    tvd = total_variation_distance(p, q)
    metrics.append(ColumnMetric(
        column=f"{count_col}_distribution",
        metric_type="total_variation",
        value=tvd,
        passed=tvd <= tv_tol,
        detail=f"TV={tvd:.4f}, {len(all_bins)} bins",
    ))

    for mcol in meta.get("metric_cols", []):
        if mcol not in real_df.columns or mcol not in synth_df.columns:
            continue
        real_map = dict(zip(real_df[bin_col], real_df[mcol]))
        synth_map = dict(zip(synth_df[bin_col], synth_df[mcol]))
        common_bins = set(real_map.keys()) & set(synth_map.keys())
        errors = []
        for b in common_bins:
            rv = real_map[b]
            sv = synth_map[b]
            if pd.notna(rv) and pd.notna(sv):
                errors.append(relative_error(float(rv), float(sv)))
        if errors:
            median_re = float(np.median(errors))
            metrics.append(ColumnMetric(
                column=mcol,
                metric_type="median_relative_error",
                value=median_re,
                passed=median_re <= rel_tol,
                detail=f"median RE={median_re:.4f}, n_bins={len(errors)}",
            ))

    return metrics


def _eval_ranking_categorical(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
) -> list[ColumnMetric]:
    metrics = []
    group_col = meta["group_col"]
    value_col = meta["value_col"]

    real_map = dict(zip(real_df[group_col], real_df[value_col]))
    synth_map = dict(zip(synth_df[group_col], synth_df[value_col]))

    common_keys = set(real_map.keys()) & set(synth_map.keys())
    if len(common_keys) == 0:
        metrics.append(ColumnMetric(
            column=value_col,
            metric_type="categorical_accuracy",
            value=0.0,
            passed=False,
            detail="no overlapping groups",
        ))
        return metrics

    matches = sum(1 for k in common_keys if real_map[k] == synth_map[k])
    accuracy = matches / len(real_map)

    metrics.append(ColumnMetric(
        column=value_col,
        metric_type="categorical_accuracy",
        value=accuracy,
        passed=accuracy >= 0.5,
        detail=f"{matches}/{len(real_map)} countries correct "
               f"({len(common_keys)} overlapping)",
    ))

    return metrics


def _eval_ranking_numeric(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
    rho_tol: float = 0.5,
) -> list[ColumnMetric]:
    metrics = []
    item_col = meta["item_col"]
    value_col = meta["value_col"]
    group_col = meta.get("group_col")
    extra_group = meta.get("extra_group_col")

    if group_col is None and extra_group is None:
        real_items = set(real_df[item_col])
        synth_items = set(synth_df[item_col])
        overlap_items = real_items & synth_items

        item_jaccard = jaccard_similarity(real_items, synth_items)
        metrics.append(ColumnMetric(
            column="item_overlap",
            metric_type="jaccard",
            value=item_jaccard,
            passed=item_jaccard >= 0.3,
            detail=f"{len(overlap_items)}/{len(real_items)} items matched",
        ))

        if len(overlap_items) >= 2:
            real_sub = real_df[real_df[item_col].isin(overlap_items)].set_index(item_col)
            synth_sub = synth_df[synth_df[item_col].isin(overlap_items)].set_index(item_col)
            common = sorted(overlap_items)

            all_value_cols = [value_col]
            if "extra_value_cols" in meta:
                all_value_cols.extend(meta["extra_value_cols"])

            for vc in all_value_cols:
                if vc not in real_sub.columns or vc not in synth_sub.columns:
                    continue
                r_vals = np.array([float(real_sub.loc[i, vc]) for i in common])
                s_vals = np.array([float(synth_sub.loc[i, vc]) for i in common])
                rho = spearman_rho(r_vals, s_vals)
                metrics.append(ColumnMetric(
                    column=f"{vc}_rank_corr",
                    metric_type="spearman_rho",
                    value=rho,
                    passed=rho >= rho_tol,
                    detail=f"rho={rho:.4f}, n_items={len(common)}",
                ))
        return metrics

    groups_to_use = [group_col]
    if extra_group:
        groups_to_use.append(extra_group)

    valid_groups = [g for g in groups_to_use if g is not None]
    if not valid_groups:
        return metrics

    real_group_keys = set(real_df[valid_groups].apply(tuple, axis=1))
    synth_group_keys = set(synth_df[valid_groups].apply(tuple, axis=1))
    common_groups = real_group_keys & synth_group_keys

    if len(common_groups) == 0:
        metrics.append(ColumnMetric(
            column="group_coverage",
            metric_type="jaccard",
            value=0.0,
            passed=False,
            detail="no overlapping groups",
        ))
        return metrics

    rho_values = []
    jaccard_values = []

    for gk in common_groups:
        if len(valid_groups) == 1:
            real_g = real_df[real_df[valid_groups[0]] == gk[0]]
            synth_g = synth_df[synth_df[valid_groups[0]] == gk[0]]
        else:
            mask_r = (real_df[valid_groups[0]] == gk[0]) & (real_df[valid_groups[1]] == gk[1])
            mask_s = (synth_df[valid_groups[0]] == gk[0]) & (synth_df[valid_groups[1]] == gk[1])
            real_g = real_df[mask_r]
            synth_g = synth_df[mask_s]

        real_items_g = set(real_g[item_col])
        synth_items_g = set(synth_g[item_col])
        overlap_g = real_items_g & synth_items_g
        jaccard_values.append(jaccard_similarity(real_items_g, synth_items_g))

        if len(overlap_g) >= 2:
            r_sub = real_g.set_index(item_col).loc[sorted(overlap_g)]
            s_sub = synth_g.set_index(item_col).loc[sorted(overlap_g)]
            rho = spearman_rho(
                r_sub[value_col].values.astype(float),
                s_sub[value_col].values.astype(float),
            )
            rho_values.append(rho)

    avg_jaccard = float(np.mean(jaccard_values)) if jaccard_values else 0.0
    metrics.append(ColumnMetric(
        column="item_overlap",
        metric_type="mean_jaccard",
        value=avg_jaccard,
        passed=avg_jaccard >= 0.3,
        detail=f"avg Jaccard={avg_jaccard:.4f} across {len(common_groups)} groups",
    ))

    if rho_values:
        avg_rho = float(np.mean(rho_values))
        metrics.append(ColumnMetric(
            column=f"{value_col}_rank_corr",
            metric_type="mean_spearman_rho",
            value=avg_rho,
            passed=avg_rho >= rho_tol,
            detail=f"avg rho={avg_rho:.4f} across {len(rho_values)} groups",
        ))

    return metrics


def _eval_row_level(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    meta: dict,
) -> list[ColumnMetric]:
    metrics = []

    row_count_re = relative_error(len(real_df), len(synth_df))
    metrics.append(ColumnMetric(
        column="row_count",
        metric_type="relative_error",
        value=row_count_re,
        passed=row_count_re <= 0.50,
        detail=f"real={len(real_df)}, synth={len(synth_df)}",
    ))

    for col in meta.get("cat_cols", []):
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_dist = real_df[col].value_counts(normalize=True)
        synth_dist = synth_df[col].value_counts(normalize=True)
        all_vals = sorted(set(real_dist.index) | set(synth_dist.index))
        p = np.array([real_dist.get(v, 0.0) for v in all_vals])
        q = np.array([synth_dist.get(v, 0.0) for v in all_vals])
        tvd = total_variation_distance(p, q)
        metrics.append(ColumnMetric(
            column=f"{col}_distribution",
            metric_type="total_variation",
            value=tvd,
            passed=tvd <= 0.20,
            detail=f"TV={tvd:.4f}",
        ))

    for col in meta.get("metric_cols", []):
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_mean = real_df[col].mean()
        s_mean = synth_df[col].mean()
        re = relative_error(r_mean, s_mean)
        metrics.append(ColumnMetric(
            column=f"{col}_mean",
            metric_type="relative_error",
            value=re,
            passed=re <= 0.50,
            detail=f"real_mean={r_mean:.2f}, synth_mean={s_mean:.2f}",
        ))

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation interface
# ---------------------------------------------------------------------------


EVAL_DISPATCH = {
    "aggregate": _eval_aggregate,
    "distribution": _eval_distribution,
    "histogram": _eval_histogram,
    "ranking_categorical": _eval_ranking_categorical,
    "ranking_numeric": _eval_ranking_numeric,
    "row_level": _eval_row_level,
}


def evaluate_query(
    query_name: str,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    rel_tol: float = 0.25,
    tv_tol: float = 0.15,
    rho_tol: float = 0.5,
) -> QueryResult:
    if query_name not in QUERY_METADATA:
        return QueryResult(
            query_name=query_name,
            query_type="unknown",
            error=f"no metadata for query '{query_name}'",
        )

    meta = QUERY_METADATA[query_name]
    qtype = meta["type"]
    eval_fn = EVAL_DISPATCH.get(qtype)

    if eval_fn is None:
        return QueryResult(
            query_name=query_name,
            query_type=qtype,
            error=f"no evaluator for type '{qtype}'",
        )

    kwargs = {}
    if qtype == "aggregate":
        kwargs = {"rel_tol": rel_tol}
    elif qtype == "distribution":
        kwargs = {"tv_tol": tv_tol}
    elif qtype == "histogram":
        kwargs = {"tv_tol": tv_tol, "rel_tol": rel_tol}
    elif qtype == "ranking_numeric":
        kwargs = {"rho_tol": rho_tol}

    try:
        col_metrics = eval_fn(real_df, synth_df, meta, **kwargs)
    except Exception as e:
        return QueryResult(
            query_name=query_name,
            query_type=qtype,
            error=str(e),
        )

    n_total = len(col_metrics)
    n_passed = sum(1 for m in col_metrics if m.passed)
    score = n_passed / n_total if n_total > 0 else 0.0

    return QueryResult(
        query_name=query_name,
        query_type=qtype,
        metrics=col_metrics,
        overall_score=score,
        passed=score >= 0.5,
    )


def evaluate_all(
    real_dir: Path,
    synth_dir: Path,
    rel_tol: float = 0.25,
    tv_tol: float = 0.15,
    rho_tol: float = 0.5,
) -> list[QueryResult]:
    results = []
    for query_name in sorted(QUERY_METADATA.keys()):
        real_path = real_dir / f"{query_name}.csv"
        synth_path = synth_dir / f"{query_name}.csv"

        if not real_path.exists():
            results.append(QueryResult(
                query_name=query_name,
                query_type=QUERY_METADATA[query_name]["type"],
                error="real CSV not found",
            ))
            continue

        if not synth_path.exists():
            results.append(QueryResult(
                query_name=query_name,
                query_type=QUERY_METADATA[query_name]["type"],
                error="synth CSV not found",
            ))
            continue

        real_df = pd.read_csv(real_path)
        synth_df = pd.read_csv(synth_path)
        result = evaluate_query(
            query_name, real_df, synth_df,
            rel_tol=rel_tol, tv_tol=tv_tol, rho_tol=rho_tol,
        )
        results.append(result)

    return results


def results_to_dataframe(results: list[QueryResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.error:
            rows.append({
                "query": r.query_name,
                "type": r.query_type,
                "score": None,
                "passed": None,
                "n_metrics": 0,
                "n_passed": 0,
                "error": r.error,
            })
        else:
            rows.append({
                "query": r.query_name,
                "type": r.query_type,
                "score": r.overall_score,
                "passed": r.passed,
                "n_metrics": r.n_total,
                "n_passed": r.n_passed,
                "error": "",
            })
    return pd.DataFrame(rows)


def detailed_results_to_dataframe(results: list[QueryResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for m in r.metrics:
            rows.append({
                "query": r.query_name,
                "query_type": r.query_type,
                "column": m.column,
                "metric_type": m.metric_type,
                "value": m.value,
                "passed": m.passed,
                "detail": m.detail,
            })
    return pd.DataFrame(rows)
