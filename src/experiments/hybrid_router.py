from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import pandas as pd

from src.eval.compare import QUERY_METADATA


@dataclass
class RoutingRule:
    by_type: dict[str, str]


DEFAULT_ROUTING_RULE = RoutingRule(
    by_type={
        "aggregate": "pertable",
        "distribution": "mst",
        "histogram": "pertable",
        "ranking_categorical": "pertable",
        "ranking_numeric": "mst",
        "row_level": "mst",
    }
)


def route_query_name(query_name: str, rule: RoutingRule = DEFAULT_ROUTING_RULE) -> str:
    qtype = QUERY_METADATA[query_name]["type"]
    return rule.by_type[qtype]


def build_routed_results(
    source_result_dirs: dict[str, Path],
    output_dir: Path,
    rule: RoutingRule = DEFAULT_ROUTING_RULE,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    routed_from: dict[str, str] = {}

    for query_name in sorted(QUERY_METADATA.keys()):
        method = route_query_name(query_name, rule)
        src_dir = source_result_dirs.get(method)
        if src_dir is None:
            continue

        src = src_dir / f"{query_name}.csv"
        dst = output_dir / f"{query_name}.csv"
        if not src.exists():
            continue

        shutil.copy2(src, dst)
        routed_from[query_name] = method

    return routed_from


def summarize_routing(routed_from: dict[str, str]) -> pd.DataFrame:
    rows = [{"query": q, "method": m, "type": QUERY_METADATA[q]["type"]} for q, m in routed_from.items()]
    if not rows:
        return pd.DataFrame(columns=["query", "method", "type"])
    return pd.DataFrame(rows).sort_values(["method", "type", "query"])

