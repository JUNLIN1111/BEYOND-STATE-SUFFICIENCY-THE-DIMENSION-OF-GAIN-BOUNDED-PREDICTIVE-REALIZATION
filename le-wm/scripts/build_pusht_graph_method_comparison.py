from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _pick_best_quotient(rows: List[Dict[str, str]], method: str) -> Dict[str, str] | None:
    subset = [row for row in rows if row.get("method_label") == method or row.get("clustering_method") == method]
    if not subset:
        return None
    valid = [row for row in subset if _as_float(row.get("largest_connected_component_fraction")) >= 0.75]
    pool = valid or subset
    return min(pool, key=lambda row: abs(_as_float(row.get("num_clusters")) - 1024.0))


def _normalize_row(row: Dict[str, str] | None, method: str, semantics: str, similarity_edges: str, status: str = "available") -> Dict[str, object]:
    if row is None:
        return {
            "graph_method": method,
            "status": "missing",
            "construction_semantics": semantics,
            "similarity_directly_creates_transition_edges": similarity_edges,
        }
    return {
        "graph_method": method,
        "status": status,
        "construction_semantics": semantics,
        "similarity_directly_creates_transition_edges": similarity_edges,
        "d90": row.get("d90", ""),
        "d95": row.get("d95", ""),
        "d99": row.get("d99", ""),
        "positive_eigen_count": row.get("positive_eigen_count", row.get("d100", "")),
        "negative_energy_ratio": row.get("negative_energy_ratio", ""),
        "largest_connected_component_size": row.get("largest_connected_component_size", row.get("largest_component_size", "")),
        "largest_connected_component_fraction": row.get("largest_connected_component_fraction", ""),
        "num_connected_components": row.get("num_connected_components", row.get("components", "")),
        "num_edges_or_avg_degree": row.get("average_degree", row.get("average_degree_total", "")),
        "num_clusters": row.get("num_clusters", ""),
        "cluster_radius_median": row.get("cluster_radius_median", ""),
    }


def _write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# PushT Graph-Method Comparison",
        "",
        "This table separates graph-construction choices from the downstream spectral diagnostic.",
        "",
        "| method | status | similarity creates transition edges? | LCC | components | d90 | d95 | d99 | negative energy | semantics |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('graph_method')} | {row.get('status')} | {row.get('similarity_directly_creates_transition_edges')} | "
            f"{row.get('largest_connected_component_size', '')} | {row.get('num_connected_components', '')} | "
            f"{row.get('d90', '')} | {row.get('d95', '')} | {row.get('d99', '')} | "
            f"{row.get('negative_energy_ratio', '')} | {row.get('construction_semantics')} |"
        )
    lines.extend(
        [
            "",
            "## Reading the comparison",
            "",
            "- `temporal_only` is the cleanest transition graph but is usually disconnected across offline trajectories, so its spectrum can describe only small local components.",
            "- `temporal_plus_knn` and `knn_only` are sensitivity baselines: they allow state-space proximity to create transition edges, which is exactly the possible false-shortcut issue.",
            "- Quotient methods use state similarity only to aggregate nearby samples; all graph edges come from observed temporal transitions.",
            "- Exact D_plan values can move with graph construction. The useful question is whether conservative transition-based quotients remain far above raw physical state dimension and local intrinsic dimension.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PushT graph-method comparison table.")
    parser.add_argument("--graph_sensitivity_csv", default="rollout_results/plannable_dim_evidence/spectra/pusht_graph_sensitivity.csv")
    parser.add_argument("--quotient_csv", default="rollout_results/plannable_dim_evidence/spectra/pusht_quotient_fixed_radius_kcenter.csv")
    parser.add_argument("--kmeans_quotient_csv", default="rollout_results/plannable_dim_evidence/spectra/clustered_quotient_diagnostic.csv")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    args = parser.parse_args()

    graph_rows = _read_csv(Path(args.graph_sensitivity_csv))
    quotient_rows = _read_csv(Path(args.quotient_csv))
    kmeans_rows = _read_csv(Path(args.kmeans_quotient_csv))
    by_mode = {row.get("graph_mode"): row for row in graph_rows}

    rows: List[Dict[str, object]] = [
        _normalize_row(
            by_mode.get("temporal_only"),
            "temporal_only",
            "observed adjacent transitions only; no cross-trajectory stitching",
            "no",
        ),
        _normalize_row(
            by_mode.get("temporal_plus_knn_k5"),
            "temporal_plus_knn_k5",
            "temporal edges plus kNN state-similarity edges",
            "yes",
        ),
        _normalize_row(
            by_mode.get("temporal_plus_knn_k10"),
            "temporal_plus_knn_k10",
            "temporal edges plus kNN state-similarity edges",
            "yes",
        ),
        _normalize_row(
            by_mode.get("temporal_plus_knn_k20"),
            "temporal_plus_knn_k20",
            "temporal edges plus kNN state-similarity edges",
            "yes",
        ),
        _normalize_row(
            by_mode.get("knn_only_k10"),
            "knn_only_k10",
            "Isomap-style state-similarity graph; appendix/sensitivity baseline",
            "yes",
        ),
        _normalize_row(
            _pick_best_quotient(kmeans_rows, "kmeans"),
            "kmeans_quotient",
            "cluster states by k-means; quotient edges induced only by temporal transitions",
            "no",
        ),
        _normalize_row(
            _pick_best_quotient(quotient_rows, "fixed_radius"),
            "fixed_radius_quotient",
            "cluster states by radius cover; quotient edges induced only by temporal transitions",
            "no",
        ),
        _normalize_row(
            _pick_best_quotient(quotient_rows, "kcenter"),
            "kcenter_quotient",
            "cluster states by farthest-point centers; quotient edges induced only by temporal transitions",
            "no",
        ),
    ]

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "spectra" / "pusht_graph_method_comparison.csv"
    md_path = output_dir / "summaries" / "pusht_graph_method_comparison.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    missing = [str(row["graph_method"]) for row in rows if row.get("status") == "missing"]
    if missing:
        print(f"[graph_compare] missing rows: {', '.join(missing)}")
    print(f"[graph_compare] wrote {csv_path}")
    print(f"[graph_compare] wrote {md_path}")


if __name__ == "__main__":
    main()
