"""Generate report figures data from all evaluation CSVs.

Reads evaluation CSVs for all methods and outputs the numbers
needed for the report's tikz figures and longtable.
"""
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("data/results")

METHODS = {
    "Wide-table DP-SGD": "evaluation_widetable",
    "Per-table DP-SGD": "evaluation_pertable",
    "MST baseline": "evaluation_mst",
    "Private Evolution": "evaluation_pe",
}

QUERY_TYPE_MAP = {
    "avg_platform_power_c0_freq_temp_by_chassis": "Agg+Join",
    "server_exploration_1": "Agg+Join",
    "mods_blockers_by_osname_and_codename": "Agg+Join",
    "top_mods_blocker_types_durations_by_osname_and_codename": "Agg+Join",
    "display_devices_connection_type_resolution_durations_ac_dc": "Agg+Join",
    "display_devices_vendors_percentage": "Agg+Join",
    "most_popular_browser_in_each_country_by_system_count": "Top-k",
    "userwait_top_10_wait_processes": "Top-k",
    "userwait_top_10_wait_processes_wait_type_ac_dc": "Top-k",
    "userwait_top_20_wait_processes_compare_ac_dc_unknown_durations": "Top-k",
    "top_10_applications_by_app_type_ranked_by_focal_time": "Top-k",
    "top_10_applications_by_app_type_ranked_by_system_count": "Top-k",
    "top_10_applications_by_app_type_ranked_by_total_detections": "Top-k",
    "Xeon_network_consumption": "Geo/Demo",
    "pkg_power_by_country": "Geo/Demo",
    "battery_power_on_geographic_summary": "Geo/Demo",
    "battery_on_duration_cpu_family_gen": "Geo/Demo",
    "ram_utilization_histogram": "Histogram",
    "popular_browsers_by_count_usage_percentage": "Histogram",
    "persona_web_cat_usage_analysis": "Pivot",
    "on_off_mods_sleep_summary_by_cpu_marketcodename_gen": "Pivot",
}

TYPE_ORDER = ["Agg+Join", "Top-k", "Geo/Demo", "Histogram", "Pivot"]


def load_method(name: str) -> pd.DataFrame | None:
    csv_path = RESULTS_DIR / f"{name}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return df[(df["error"].isna()) | (df["error"] == "")]


def main():
    summaries = {}
    for label, csv_name in METHODS.items():
        df = load_method(csv_name)
        if df is not None:
            summaries[label] = df
            n_pass = int(df["passed"].sum())
            n_eval = len(df)
            avg = df["score"].mean()
            med = df["score"].median()
            print(f"{label}: {n_pass}/{n_eval} passed, avg={avg:.3f}, med={med:.3f}")
        else:
            print(f"{label}: NOT AVAILABLE")

    print("\n=== Figure 1 data (overall comparison) ===")
    for label, df in summaries.items():
        n_pass = int(df["passed"].sum())
        avg = df["score"].mean()
        short = label.split()[0] if label != "Private Evolution" else "PE"
        print(f"  ({short},{n_pass})  ({short},{avg:.3f})")

    print("\n=== Figure 3 data (by query type) ===")
    for qtype in TYPE_ORDER:
        line = f"  {qtype}: "
        for label, df in summaries.items():
            type_queries = [q for q, t in QUERY_TYPE_MAP.items() if t == qtype]
            scores = []
            for _, row in df.iterrows():
                if row["query"] in type_queries:
                    scores.append(row["score"])
            avg = sum(scores) / len(scores) if scores else 0
            short = label.split()[0] if label != "Private Evolution" else "PE"
            line += f"{short}={avg:.3f}  "
        print(line)

    print("\n=== Table 4 data (per-query scores) ===")
    all_queries = sorted(QUERY_TYPE_MAP.keys())
    for q in all_queries:
        line = f"  {q[:50]:50s}"
        for label, df in summaries.items():
            match = df[df["query"] == q]
            if len(match) > 0:
                score = match.iloc[0]["score"]
                passed = score >= 0.5
                if score >= 0.5:
                    color = "scorehi"
                elif score >= 0.25:
                    color = "scoremid"
                else:
                    color = "scorelo"
                check = r"\,$\checkmark$" if passed else ""
                line += f"  \\cellcolor{{{color}}}{score:.3f}{check}"
            else:
                line += f"  \\cellcolor{{scorena}}---"
        print(line)


if __name__ == "__main__":
    main()
