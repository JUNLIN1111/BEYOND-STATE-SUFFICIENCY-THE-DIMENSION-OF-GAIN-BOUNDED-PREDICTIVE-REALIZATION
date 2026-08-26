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


def _metric_table(rows: List[Dict[str, str]], max_rows: int = 20) -> str:
    if not rows:
        return "_Not available yet._"
    keys = list(rows[0].keys())
    preferred = [key for key in ["model", "graph_mode", "latent_dim", "metric", "mean", "ci95_low", "ci95_high", "d90", "d95", "d99", "largest_connected_component_size"] if key in keys]
    keys = preferred or keys[:6]
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    if len(rows) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | ... |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final plannable-dimension evidence summary markdown.")
    parser.add_argument("--input_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--output", default="rollout_results/plannable_dim_evidence/summaries/final_evidence_package.md")
    args = parser.parse_args()

    root = Path(args.input_dir)
    candidate = _read_csv(root / "summaries" / "candidate_future_metric_paper_table_n100.csv")
    graph_sensitivity = _read_csv(root / "spectra" / "pusht_graph_sensitivity.csv")
    convergence = _read_csv(root / "spectra" / "pusht_sampling_convergence.csv")
    pred_loss = _read_csv(root / "summaries" / "prediction_loss_control_table.csv")
    second_task = _read_csv(root / "spectra" / "second_task_spectral_table.csv")

    lines = [
        "# Plannable Latent Dimension Evidence Package",
        "",
        "## 1. Theory message",
        "",
        "Distance-based latent planners require transition structure to be represented in the latent metric, not merely decoded by downstream dynamics. If the planner cost is Euclidean distance to a latent goal, then transition-far candidate futures that become latent-near create false shortcuts.",
        "",
        "## 2. Diagnostic",
        "",
        "We build an empirical transition graph from feasible transition edges in offline trajectories, compute shortest-path transition/reachability distances, double-center the distance matrix, and inspect the positive MDS spectrum. `D_plan(q)` is a planner-facing spectral diagnostic: it estimates the Euclidean latent width needed to preserve q fraction of task-induced transition-distance energy.",
        "",
        "This is not claimed to be an exact optimal dimension.",
        "",
        "## 3. Difference from Isomap",
        "",
        "Classical Isomap uses observation-space similarity to build a kNN graph and recover data-manifold geodesics. Here, the primary edges are feasible transition edges from trajectories. kNN edges, when used, are a stitching / robustness device rather than the definition of transition geometry.",
        "",
        "## 4. Controlled toy",
        "",
        "The controlled toy shows a 2D physical layout with nearby states in adjacent corridors whose transition path must go through a junction. Physical proximity is therefore not transition proximity, and low-dimensional embeddings can create graph-far / Euclidean-near false shortcuts.",
        "",
        "Outputs: `toy/controlled_toy_summary.md` and `plots/controlled_toy_*`.",
        "",
        "## 5. PushT graph sensitivity",
        "",
        _metric_table(graph_sensitivity),
        "",
        "Interpretation: exact values can depend on graph construction, but the key question is whether reasonable transition graphs remain far above physical state dimension (7) and local intrinsic dimension (~2). `knn_only_k10` should be treated as an Isomap-style ablation.",
        "",
        "## 6. PushT sampling convergence",
        "",
        _metric_table(convergence),
        "",
        "If D_plan grows with sample size, it should be reported as a sampled empirical estimate. The diagnostic remains useful if values consistently stay far above physical/local dimension.",
        "",
        "## 7. Candidate mechanism validation",
        "",
        _metric_table(candidate, max_rows=30),
        "",
        "This is mechanism evidence. The intended claim is that too-small latent width can increase graph-far / score-near candidate errors. Confidence intervals should be respected; do not claim statistical significance if they overlap.",
        "",
        "## 8. Prediction-loss negative control",
        "",
        _metric_table(pred_loss),
        "",
        "Prediction accuracy is necessary but not sufficient. Comparable one-step latent prediction losses can coexist with different candidate-level false shortcuts, ranking fidelity, and closed-loop success.",
        "",
        "## 9. Optional second task",
        "",
        _metric_table(second_task),
        "",
        "The second-task spectrum is an offline diagnostic only; it is not closed-loop Cube evaluation.",
        "",
        "## 10. Limitations",
        "",
        "- Graph construction matters.",
        "- Offline coverage matters.",
        "- `D_plan(q)` is a tradeoff curve, not an exact optimal latent dimension.",
        "- The scope is distance-based latent planning; the claim should not be directly generalized to critic-based world models such as Dreamer.",
        "",
        "## Recommended wording",
        "",
        "> Plannable latent dimension is a planner-facing spectral diagnostic. It estimates the Euclidean latent width needed to represent task-induced transition geometry as a metric for distance-based planning.",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"[final_evidence] wrote {output}")


if __name__ == "__main__":
    main()
